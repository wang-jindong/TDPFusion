import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPTokenizer
import torchvision.transforms.functional as TF

class RoadSceneDataset(Dataset):
    def __init__(self, file_list, ir_folder, vis_folder,
                 mask_folder=None, img_size=256, mode='train', prompt_json_path=None):
        super(RoadSceneDataset, self).__init__()
        self.file_list = file_list
        self.ir_folder = ir_folder
        self.vis_folder = vis_folder
        self.mask_folder = mask_folder  # 伪标签存放路径
        self.img_size = img_size
        self.mode = mode

        # 1. 初始化 Tokenizer
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(os.environ.get("TDPFUSION_CLIP_PATH", "./clip_weights"))
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # 2. 加载 RoadScene 专属 Prompt JSON
        self.prompts = {}
        if prompt_json_path and os.path.exists(prompt_json_path):
            with open(prompt_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        k = item.get('file_name') or item.get('id')
                        v = item.get('text') or item.get('description')
                        if k and v:
                            self.prompts[str(k).split('.')[0]] = v
                else:
                    self.prompts = data
            print(f"✅ Dataset: 成功加载了 {len(self.prompts)} 条文本描述")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        name = self.file_list[index]['file_name']
        ir_img = Image.open(os.path.join(self.ir_folder, name)).convert('L')
        vis_img = Image.open(os.path.join(self.vis_folder, name)).convert('RGB')

        # 加载 Mask
        mask_img = None
        if self.mask_folder is not None:
            mask_name = os.path.splitext(name)[0] + ".png"
            mask_path = os.path.join(self.mask_folder, mask_name)
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert('L')

        # 获取当前尺寸
        w, h = ir_img.size

        # --- [修正点]：保底缩放逻辑 ---
        # 如果原图比 256 还小，random.randint 会报错。先放大到 256 以上。
        if w < self.img_size or h < self.img_size:
            scale = max(self.img_size / w, self.img_size / h)
            new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
            ir_img = ir_img.resize((new_w, new_h), Image.BILINEAR)
            vis_img = vis_img.resize((new_w, new_h), Image.BILINEAR)
            if mask_img is not None:
                mask_img = mask_img.resize((new_w, new_h), Image.NEAREST)
            # 更新缩放后的尺寸
            w, h = ir_img.size

        # --- 绝对同步的数据增强 ---
        if self.mode == 'train':
            # 1. 随机裁剪
            th, tw = self.img_size, self.img_size
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

            ir_img = TF.crop(ir_img, i, j, th, tw)
            vis_img = TF.crop(vis_img, i, j, th, tw)
            if mask_img is not None:
                mask_img = TF.crop(mask_img, i, j, th, tw)

            # 2. 随机水平翻转
            if random.random() > 0.5:
                ir_img = TF.hflip(ir_img)
                vis_img = TF.hflip(vis_img)
                if mask_img is not None:
                    mask_img = TF.hflip(mask_img)
        else:
            # 验证模式：直接 Resize 到目标尺寸
            ir_img = ir_img.resize((self.img_size, self.img_size), Image.BILINEAR)
            vis_img = vis_img.resize((self.img_size, self.img_size), Image.BILINEAR)
            if mask_img is not None:
                mask_img = mask_img.resize((self.img_size, self.img_size), Image.NEAREST)

        # --- 归一化与转换为 Tensor ---
        ir_tensor = TF.normalize(TF.to_tensor(ir_img), [0.5], [0.5])
        vis_tensor = TF.normalize(TF.to_tensor(vis_img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        # 处理 Mask Tensor
        if mask_img is not None:
            mask_tensor = torch.from_numpy(np.array(mask_img)).long()
        else:
            # 如果没找到伪标签，返回忽略索引 255
            mask_tensor = torch.ones((self.img_size, self.img_size)).long() * 255

        # --- 文本处理 ---
        key = name.split('.')[0]
        prompt = self.prompts.get(key, "infrared and visible image fusion for road scene")
        tokens = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=77, return_tensors="pt")

        return {
            'ir': ir_tensor,
            'vis': vis_tensor,
            'mask': mask_tensor,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'file_name': name
        }