import os
import argparse
import json
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from inference.common import load_tokenizer, write_cv2_image
from tqdm import tqdm

# 引入你的模型定义 (请确保路径正确)
from model.Text_IF_model import Text_IF as create_model

# 忽略环境警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_prompts(json_path):
    """从 JSON 加载 RoadScene 专属 Prompt 字典"""
    prompts = {}
    if json_path and os.path.exists(json_path):
        print(f"[Info] Loading RoadScene prompts from {json_path}...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    k = item.get('file_name') or item.get('id')
                    v = item.get('text') or item.get('description')
                    if k and v:
                        prompts[str(k).split('.')[0]] = v
            elif isinstance(data, dict):
                prompts = data
            print(f"[Info] Loaded {len(prompts)} prompts.")
        except Exception as e:
            print(f"[Warning] Failed to load JSON: {e}")
    return prompts


def process_image(ir_path, vis_path, device):
    """读取并预处理图片 (归一化到 [-1, 1])"""
    img_ir = Image.open(ir_path).convert('L')
    img_vis = Image.open(vis_path).convert('RGB')

    # 转 Tensor 并归一化到 [-1, 1]
    img_ir = (TF.to_tensor(img_ir).to(device) - 0.5) / 0.5
    img_vis = (TF.to_tensor(img_vis).to(device) - 0.5) / 0.5

    img_ir = img_ir.unsqueeze(0)
    img_vis = img_vis.unsqueeze(0)

    # 计算 Padding 以适配模型 (16的倍数)
    _, _, h, w = img_ir.shape
    factor = 16
    buffer = 32
    pad_h_need = (factor - h % factor) % factor
    pad_w_need = (factor - w % factor) % factor

    img_ir = F.pad(img_ir, (buffer, pad_w_need + buffer, buffer, pad_h_need + buffer), mode='reflect')
    img_vis = F.pad(img_vis, (buffer, pad_w_need + buffer, buffer, pad_h_need + buffer), mode='reflect')

    return img_ir, img_vis, h, w, buffer, buffer


def load_weights_smart(model, weight_path, device):
    if not os.path.exists(weight_path):
        print(f"❌ Error: 权重不存在 {weight_path}")
        return False
    print(f"�� Loading weights: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    # 移除分布式训练前缀
    new_state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict, strict=True)
    print("✅ Weights loaded successfully!")
    return True


def save_pic(output_tensor, path, name):
    # 1. 转为 Numpy 并限制范围
    outputpic = output_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

    # 2. 鲁棒性映射：先做百分位截断，再做线性拉伸
    # 这能有效压制红外高亮，同时拉开可见光纹理的对比度
    p_low, p_high = np.percentile(outputpic, (0.5, 99.5))
    outputpic = np.clip(outputpic, p_low, p_high)

    # 归一化到 [0, 1]
    outputpic = (outputpic - p_low) / (p_high - p_low + 1e-6)

    # 3. 映射到 [0, 255]
    outputpic = (outputpic * 255).astype(np.uint8)

    if outputpic.shape[2] == 3:
        outputpic = cv2.cvtColor(outputpic, cv2.COLOR_RGB2BGR)
    write_cv2_image(os.path.join(path, name), outputpic)

def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 1. 加载 Tokenizer
    tokenizer = load_tokenizer(args.clip_path)

    # 2. 加载 Prompts
    prompts_dict = load_prompts(args.prompt_path)

    # 3. 扫描 RoadScene 图片
    ir_dir = os.path.join(args.dataset_path, "cropinfrared")
    if not os.path.exists(ir_dir): ir_dir = os.path.join(args.dataset_path, "ir")
    vis_dir = os.path.join(args.dataset_path, "crop_HR_visible")
    if not os.path.exists(vis_dir): vis_dir = os.path.join(args.dataset_path, "vi")

    vis_list = sorted([x for x in os.listdir(vis_dir) if x.endswith(('.jpg', '.png', '.bmp'))])
    pairs = []
    for v in vis_list:
        name_stem = os.path.splitext(v)[0]
        ir_name = None
        for ext in ['.jpg', '.png', '.bmp']:
            if os.path.exists(os.path.join(ir_dir, name_stem + ext)):
                ir_name = name_stem + ext
                break
        if ir_name:
            pairs.append((os.path.join(ir_dir, ir_name), os.path.join(vis_dir, v), v))

    print(f"�� Found {len(pairs)} pairs. Starting inference...")

    # 4. 初始化模型
    model = create_model(inp_A_channels=1, inp_B_channels=3).to(device)
    if not load_weights_smart(model, args.weights, device):
        return
    model.eval()

    # 5. 推理
    with torch.no_grad():
        for ir_p, vis_p, name in tqdm(pairs):
            key = os.path.splitext(name)[0]
            text = prompts_dict.get(key, args.default_text)

            inputs = tokenizer(text, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            img_ir, img_vis, h_ori, w_ori, pt, pl = process_image(ir_p, vis_p, device)

            # 注意：如果你的模型输出是 tuple (fused, mask)，取第一个
            output = model(img_ir, img_vis, input_ids, attention_mask)
            fused = output[0] if isinstance(output, tuple) else output

            # 裁减回原始尺寸
            fused = fused[:, :, pt: pt + h_ori, pl: pl + w_ori]

            save_pic(fused, args.save_path, name)

    print(f"\n✨ 测试完成！结果路径: {args.save_path}")


