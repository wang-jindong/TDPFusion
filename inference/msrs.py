import os
import argparse
import json
import numpy as np
from PIL import Image
import cv2
from inference.common import write_cv2_image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from transformers import CLIPTokenizer
from tqdm import tqdm  # 建议安装 pip install tqdm

# 引入你的模型定义
from model.Text_IF_model import Text_IF as create_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_prompts(json_path):
    """从 JSON 加载 Prompt 字典"""
    prompts = {}
    if json_path and os.path.exists(json_path):
        print(f"[Info] Loading prompts from {json_path}...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 兼容 List 和 Dict 格式
            if isinstance(data, list):
                for item in data:
                    k = item.get('file_name') or item.get('image') or item.get('id')
                    v = item.get('description') or item.get('caption') or item.get('text')
                    if k and v:
                        # 去掉后缀作为 Key (例如 00016N)
                        prompts[k.split('.')[0]] = v
            elif isinstance(data, dict):
                prompts = data
            print(f"[Info] Loaded {len(prompts)} prompts.")
        except Exception as e:
            print(f"[Warning] Failed to load JSON: {e}")
    return prompts


def process_image(ir_path, vis_path, device):
    """
    读取并预处理图片 (适配 [-1, 1] 范围)
    """
    img_ir = Image.open(ir_path).convert('L')
    img_vis = Image.open(vis_path).convert('RGB')

    # 转 Tensor [0, 1]
    img_ir = TF.to_tensor(img_ir).to(device)
    img_vis = TF.to_tensor(img_vis).to(device)

    # 归一化到 [-1, 1] (配合 Scheme 2)
    img_ir = (img_ir - 0.5) / 0.5
    img_vis = (img_vis - 0.5) / 0.5

    # 增加 Batch 维度
    img_ir = img_ir.unsqueeze(0)
    img_vis = img_vis.unsqueeze(0)

    # === 安全缓冲区策略 (消除边缘伪影) ===
    _, _, h, w = img_ir.shape
    factor = 16
    buffer = 32  # 安全边距

    # 计算对齐所需的 padding
    pad_h_need = (factor - h % factor) % factor
    pad_w_need = (factor - w % factor) % factor

    pad_left = buffer
    pad_right = pad_w_need + buffer
    pad_top = buffer
    pad_bottom = pad_h_need + buffer

    # Reflect Padding
    img_ir = F.pad(img_ir, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    img_vis = F.pad(img_vis, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

    return img_ir, img_vis, h, w, pad_top, pad_left


def load_weights_smart(model, weight_path, device):
    if not os.path.exists(weight_path):
        print(f"Error: 权重不存在 {weight_path}")
        return False
    print(f"Loading weights from: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)
    # 兼容带 'model' 键或直接存 state_dict 的情况
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        # 去掉分布式训练可能产生的 'module.' 前缀
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ Weights loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Weights loading failed: {e}")
        return False


def save_pic(outputpic, path, name):
    # [-1, 1] -> [0, 255]
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < -1.] = -1
    outputpic = ((outputpic * 0.5 + 0.5) * 255).astype(np.uint8)

    if outputpic.shape[2] == 3:
        outputpic = cv2.cvtColor(outputpic, cv2.COLOR_RGB2BGR)
    write_cv2_image(os.path.join(path, name), outputpic)


def save_mask(mask_tensor, path, name):
    if mask_tensor is None:
        return
    # Mask [1, 1, H, W] -> [H, W]
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    write_cv2_image(os.path.join(path, "mask_" + name), heatmap)


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 1. 加载 Tokenizer
    try:
        tokenizer = CLIPTokenizer.from_pretrained(args.clip_path)
    except:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # 2. 加载 Prompts (JSON)
    prompts_dict = load_prompts(args.prompt_path)
    default_prompt = args.default_text

    # 3. 扫描图片
    root_path = args.dataset_path
    # 自动探测目录结构
    if os.path.exists(os.path.join(root_path, "vi")):  # MSRS 常用结构
        vis_dir = os.path.join(root_path, "vi")
        ir_dir = os.path.join(root_path, "ir")
    elif os.path.exists(os.path.join(root_path, "Visible")):
        vis_dir = os.path.join(root_path, "Visible")
        ir_dir = os.path.join(root_path, "Infrared")
    else:
        vis_dir = root_path
        ir_dir = root_path

    vis_list = [x for x in os.listdir(vis_dir) if x.endswith(('.jpg', '.png', '.bmp', '.tif'))]
    pairs = []
    for v in vis_list:
        # 尝试匹配红外图 (兼容 jpg/png 后缀不一致的情况)
        ir_name = v
        if not os.path.exists(os.path.join(ir_dir, ir_name)):
            # 尝试换后缀
            name_stem = v.split('.')[0]
            for ext in ['.png', '.jpg', '.bmp']:
                if os.path.exists(os.path.join(ir_dir, name_stem + ext)):
                    ir_name = name_stem + ext
                    break

        if os.path.exists(os.path.join(ir_dir, ir_name)):
            pairs.append((os.path.join(ir_dir, ir_name), os.path.join(vis_dir, v), v))

    print(f"Found {len(pairs)} image pairs.")

    # 4. 初始化模型
    model = create_model(inp_A_channels=1, inp_B_channels=3).to(device)
    if not load_weights_smart(model, args.weights, device):
        return
    model.eval()

    print(f"Starting Inference on {device}...")

    # 使用 tqdm 显示进度
    for ir_p, vis_p, name in tqdm(pairs):
        # A. 准备 Prompt
        key = name.split('.')[0]
        # 优先用 JSON 里的，没有就用默认的
        text = prompts_dict.get(key, default_prompt)

        inputs = tokenizer(text, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # B. 预处理
        img_ir, img_vis, h_ori, w_ori, pad_top, pad_left = process_image(ir_p, vis_p, device)

        # C. 推理
        with torch.no_grad():
            output, pred_mask = model(img_ir, img_vis, input_ids, attention_mask)

        # D. 裁剪回原尺寸
        output = output[:, :, pad_top: pad_top + h_ori, pad_left: pad_left + w_ori]
        if pred_mask is not None:
            pred_mask = pred_mask[:, :, pad_top: pad_top + h_ori, pad_left: pad_left + w_ori]

        # E. 保存
        save_pic(output.cpu().squeeze(0).permute(1, 2, 0).numpy(), args.save_path, name)
        if args.save_mask:
            save_mask(pred_mask, args.save_path, name)

    print(f"\nDone! Results saved to {args.save_path}")
