import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


# =================================================================
#  核心损失计算类 (基于你提供的代码)
# =================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from transformers import CLIPModel


# =================================================================
#  1. 新增：语义约束损失 (L_Semantic)
#  作用：强制融合图像的内容与 Text Prompt 匹配
# =================================================================
class L_Semantic(nn.Module):
    def __init__(self, device):
        super(L_Semantic, self).__init__()
        self.device = device
        print("正在加载 CLIP 模型用于计算语义损失 (可能需要一点时间)...")
        # 加载冻结的 CLIP 模型 (必须与你 Tokenizer 用的版本一致)
        self.clip_model = CLIPModel.from_pretrained(os.environ.get("TDPFUSION_CLIP_PATH", "./clip_weights")).to(device)

        # 冻结参数，只做评估用，不更新 CLIP 的权重
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        # CLIP 预处理所需的均值和方差
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def forward(self, image_fused, input_ids, attention_mask):
        # 1. 图像预处理：从 [-1, 1] 转换到 [0, 1]
        img = (image_fused + 1.0) / 2.0

        # 2. 如果是单通道(灰度)，强制转为3通道以喂给 CLIP
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)

        # 3. Resize 到 224x224 (CLIP 的输入标准)
        img_resized = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)

        # 4. 标准化 (Normalize)
        img_normalized = (img_resized - self.clip_mean) / self.clip_std

        # 5. 获取图像特征和文本特征
        vision_outputs = self.clip_model.vision_model(pixel_values=img_normalized)
        image_embeds = self.clip_model.visual_projection(vision_outputs.pooler_output)

        text_outputs = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = self.clip_model.text_projection(text_outputs.pooler_output)

        # 6. 归一化特征向量
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # 7. 计算余弦相似度损失 (1 - Cosine Similarity)
        # 我们希望相似度越高越好，所以 Loss = 1 - sim
        similarity = torch.matmul(image_embeds, text_embeds.t())
        # 取对角线元素 (即每张图对应它自己的 prompt)
        score = torch.diagonal(similarity)
        loss = 1.0 - score.mean()

        return loss


# =================================================================
#  2. 修改：核心损失计算类
# =================================================================
class fusion_loss(nn.Module):
    def __init__(self, device):
        super(fusion_loss, self).__init__()
        self.device = device
        self.loss_func_ssim = L_SSIM(window_size=11).to(device)
        self.loss_func_Grad = L_Grad_position().to(device)
        self.loss_func_Max = L_Intensity().to(device)
        self.loss_func_color = L_color().to(device)

        # 引入新的语义损失
        self.loss_func_Semantic = L_Semantic(device).to(device)

    def forward(self, image_visible, image_infrared, image_fused,
                input_ids, attention_mask,  # <--- 新增输入：文本 Token
                max_ratio=4, ssim_vis_ratio=1, ssim_ir_ratio=1,
                ssim_ratio=1, color_ratio=12, grad_ratio=5, text_ratio=50):  # <--- 调高 text_ratio

        image_visible = image_visible.to(self.device)
        image_infrared = image_infrared.to(self.device)
        image_fused = image_fused.to(self.device)

        image_visible_gray = self.rgb2gray(image_visible)
        image_infrared_gray = self.rgb2gray(image_infrared)
        image_fused_gray = self.rgb2gray(image_fused)

        # 1. 强度损失
        loss_max = max_ratio * self.loss_func_Max(image_visible, image_infrared, image_fused)

        # 2. SSIM 损失
        loss_ssim = ssim_ratio * (
                ssim_vis_ratio * self.loss_func_ssim(image_visible, image_fused) +
                ssim_ir_ratio * self.loss_func_ssim(image_infrared_gray, image_fused_gray)
        )

        # 3. 色彩损失
        loss_color = color_ratio * self.loss_func_color(image_visible, image_fused)

        # 4. 梯度损失 (原有的 text loss 其实是 grad loss)
        loss_grad = grad_ratio * self.loss_func_Grad(image_visible_gray, image_infrared_gray, image_fused_gray)

        # 5. [新] 语义/文本损失
        # 这里使用 text_ratio 进行强约束
        loss_semantic = text_ratio * self.loss_func_Semantic(image_fused, input_ids, attention_mask)

        # 总损失
        total_loss = loss_max + loss_ssim + loss_color + loss_grad + loss_semantic

        # 返回各项便于监控 (注意返回值数量变了，train.py 里接收时要注意)
        return total_loss, loss_ssim, loss_max, loss_color, loss_grad, loss_semantic

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray


# =================================================================
#  3. 修改：适配器类 PatentLoss
# =================================================================
class PatentLoss(nn.Module):
    def __init__(self, device):
        super(PatentLoss, self).__init__()
        self.device = device
        self.core_loss = fusion_loss(device)

    def forward(self, fused_img, ir_img, vis_img, input_ids, attention_mask):
        """
        注意：现在 forward 必须接收 input_ids 和 attention_mask
        """
        # 参数配置策略：
        # 1. 稍微降低像素级约束 (max_ratio, ssim)，给语义优化留出空间
        # 2. 极大提高 text_ratio (语义权重)，强迫图像向文本描述靠拢
        total_loss, l_ssim, l_max, l_col, l_grad, l_sem = self.core_loss(
            image_visible=vis_img,
            image_infrared=ir_img,
            image_fused=fused_img,
            input_ids=input_ids,  # 传入文本
            attention_mask=attention_mask,  # 传入 Mask

            max_ratio=5.0,  # 强度
            ssim_ratio=1.0,  # 结构
            color_ratio=8.0,  # 色彩
            grad_ratio=5.0,  # 梯度 (原 text_ratio)
            text_ratio=0.5  # [强约束] 语义权重，可以尝试 10.0 ~ 50.0
        )
        return total_loss


# =================================================================
#  (下方的辅助类 L_color, L_Intensity, L_Grad_position, L_SSIM 保持不变)
#  ... 请保留你原有的辅助类代码 ...
# =================================================================

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, image_visible, image_fused):
        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        loss_cb = F.l1_loss(cb_visible, cb_fused)
        loss_cr = F.l1_loss(cr_visible, cr_fused)

        loss_color = loss_cb + loss_cr
        return loss_color

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr_image = torch.stack((y, cb, cr), dim=1)
        return ycbcr_image


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused):
        # 确保通道一致，如果 vis 是 3 通道，ir 是 1 通道，需要处理
        if image_visible.size(1) == 3:
            gray_visible = 0.299 * image_visible[:, 0, :, :] + 0.587 * image_visible[:, 1, :,
                                                                       :] + 0.114 * image_visible[:, 2, :, :]
            gray_visible = gray_visible.unsqueeze(1)
        else:
            gray_visible = image_visible

        if image_infrared.size(1) == 3:
            gray_infrared = 0.299 * image_infrared[:, 0, :, :] + 0.587 * image_infrared[:, 1, :,
                                                                         :] + 0.114 * image_infrared[:, 2, :, :]
            gray_infrared = gray_infrared.unsqueeze(1)
        else:
            gray_infrared = image_infrared

        mask = (gray_infrared > gray_visible).float()

        # 这里的 image_infrared 可能是一通道，image_visible 是三通道
        # 为了相加，需要广播或者转换
        if image_infrared.size(1) != image_visible.size(1):
            # 简单策略：仅在灰度层面计算 intensity loss，或者将红外扩展为3通道
            ir_3c = image_infrared.repeat(1, 3, 1, 1) if image_infrared.size(1) == 1 else image_infrared
            vis_3c = image_visible
            fused_target = mask * ir_3c + (1 - mask) * vis_3c
        else:
            fused_target = mask * image_infrared + (1 - mask) * image_visible

        Loss_intensity = F.l1_loss(fused_target, image_fused)
        return Loss_intensity


class L_Grad_position(nn.Module):
    def __init__(self):
        super(L_Grad_position, self).__init__()
        # 移除 requires_grad=False 和 .cuda()，改为 register_buffer 自动管理设备
        sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse):
        gradient_A_x, gradient_A_y = self.gradient(image_A)
        gradient_B_x, gradient_B_y = self.gradient(image_B)
        gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)
        loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + F.l1_loss(gradient_fuse_y,
                                                                                             torch.max(gradient_A_y,
                                                                                                       gradient_B_y))
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)


# SSIM Utils
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = window_size // 2
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    # 避免除以0
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2 + 1e-8)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            self.window = window
            self.channel = channel

        return 1 - ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)