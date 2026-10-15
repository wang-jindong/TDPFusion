import os
import argparse
import json
import re
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer

# 导入你的模型
from model.Text_IF_model import Text_IF


def pad_tensor(img_tensor, multiple=16):
    """
    对图像张量进行 padding，使其高和宽成为 multiple 的整数倍。
    """
    _, _, h, w = img_tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    # F.pad 的格式是 (左, 右, 上, 下)
    img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return img_padded, h, w


def unpad_tensor(img_padded, orig_h, orig_w):
    """
    移除 padding，恢复图像原始尺寸。
    """
    return img_padded[:, :, :orig_h, :orig_w]


@torch.no_grad()
def main(args):
    # 1. 设置设备 (根据你的 gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"�� Using device: {device}")

    # 2. 读取 Prompt 字典
    prompt_map_id = {}
    if os.path.exists(args.prompt_path):
        try:
            with open(args.prompt_path, 'r', encoding='utf-8') as f:
                prompt_map_id = json.load(f)
            print(f"✅ Loaded prompt dictionary from {args.prompt_path}")
        except Exception as e:
            print(f"⚠️ Failed to load prompt JSON: {e}. Will use default text.")
    else:
        print(f"⚠️ Prompt path not found: {args.prompt_path}. Will use default text.")

    # 3. 初始化 CLIP Tokenizer
    try:
        tokenizer = CLIPTokenizer.from_pretrained(args.clip_path)
        print("✅ Local CLIP tokenizer loaded.")
    except:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        print("⚠️ Online CLIP tokenizer loaded.")

    # 4. 初始化模型并加载权重
    print("Loading Text_IF model...")
    model = Text_IF(inp_A_channels=1, inp_B_channels=3).to(device)

    if os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Weights loaded successfully!")
    else:
        raise FileNotFoundError(f"❌ Weights file not found: {args.weights}")

    model.eval()

    # 5. 准备数据转换 (直接转为 [0, 1] 的 Tensor，不加 Normalize)
    transform_ir = transforms.Compose([transforms.ToTensor()])
    transform_vis = transforms.Compose([transforms.ToTensor()])

    # 6. 遍历测试集
    os.makedirs(args.save_path, exist_ok=True)
    ir_files = sorted([f for f in os.listdir(args.ir_path) if f.endswith(('.png', '.jpg', '.bmp', '.tif'))])
    print(f"Found {len(ir_files)} images to fuse in {args.ir_path}.")

    for file_name in tqdm(ir_files, desc="Fusing Images"):
        ir_file_path = os.path.join(args.ir_path, file_name)
        vis_file_path = os.path.join(args.vis_path, file_name)

        if not os.path.exists(vis_file_path):
            print(f"⚠️ Warning: Missing VIS image for {file_name}. Skipping.")
            continue

        # --- 获取当前图片的 Prompt ---
        current_text = args.default_text
        nums = re.findall(r'\d+', file_name)
        if len(nums) > 0:
            f_id = nums[-1]  # 匹配训练代码提取 ID 的逻辑
            if f_id in prompt_map_id:
                current_text = prompt_map_id[f_id]

        # Tokenize
        inputs = tokenizer([current_text], padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # --- 读取图像 ---
        ir_img = Image.open(ir_file_path).convert('L')
        vis_img = Image.open(vis_file_path).convert('RGB')

        ir_tensor = transform_ir(ir_img).unsqueeze(0).to(device)
        vis_tensor = transform_vis(vis_img).unsqueeze(0).to(device)

        # Padding (使尺寸满足 16 的倍数)
        ir_tensor_padded, orig_h, orig_w = pad_tensor(ir_tensor, multiple=16)
        vis_tensor_padded, _, _ = pad_tensor(vis_tensor, multiple=16)

        # --- 前向传播 ---
        output = model(ir_tensor_padded, vis_tensor_padded, input_ids, attention_mask)
        fused_tensor_padded = output[0] if isinstance(output, tuple) else output

        # --- 后处理 ---
        # 1. Unpadding
        fused_tensor = unpad_tensor(fused_tensor_padded, orig_h, orig_w)

        # 2. 【核心修复】移除错误的 +1/2 映射，直接 clamp 到 [0, 1]
        # 因为你训练集用 ToTensor 读入的图就是 [0, 1]，网络输出已经是 [0, 1] 范围了
        fused_tensor = torch.clamp(fused_tensor, 0.0, 1.0)

        # 3. 保存图像
        fused_img_np = fused_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        fused_img_np = (fused_img_np * 255.0).astype('uint8')

        save_file_path = os.path.join(args.save_path, file_name)
        Image.fromarray(fused_img_np).save(save_file_path)

    print(f"�� All done! Results saved to: {args.save_path}")
