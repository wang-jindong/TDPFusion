"""Chapter 3 equations; discretization choices are documented in TRAINING.md."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel
from torchvision.models.segmentation import deeplabv3_resnet50


def sobel(image):
    kernel = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    channels = image.shape[1]
    image = F.pad(image, (1, 1, 1, 1), mode='replicate')
    return tuple(F.conv2d(image, k.expand(channels, 1, 3, 3), groups=channels)
                 for k in (kernel, kernel.T))


def ssim_loss(a, b):
    # Gaussian 11x11, sigma=1.5; inputs are in [-1,1], hence data range=2.
    x = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    g = torch.exp(-x.square() / (2 * 1.5 ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).expand(a.shape[1], 1, 11, 11)
    def average(t):
        return F.conv2d(F.pad(t, (5, 5, 5, 5), mode='replicate'), kernel, groups=a.shape[1])
    ma, mb = average(a), average(b)
    va, vb = average(a * a) - ma.square(), average(b * b) - mb.square()
    covariance = average(a * b) - ma * mb
    score = ((2 * ma * mb + 0.02 ** 2) * (2 * covariance + 0.06 ** 2)
             / ((ma.square() + mb.square() + 0.02 ** 2) * (va + vb + 0.06 ** 2)))
    return 1 - score.mean()


def pixel_losses(fused, ir, vi):
    ir = ir.expand_as(vi)
    gf, gi, gv = sobel(fused), sobel(ir), sobel(vi)
    intensity = F.l1_loss(fused, torch.maximum(ir, vi))
    gradient = sum(F.l1_loss(f.abs(), torch.maximum(i.abs(), v.abs()))
                   for f, i, v in zip(gf, gi, gv))
    color = sum(F.l1_loss(f, v) for f, v in zip(gf, gv))
    structure = ssim_loss(fused, vi) + ssim_loss(fused, ir)
    return {'intensity': intensity, 'gradient': gradient, 'structure': structure, 'color': color}


def judge_loss(judge, fused, mask):
    # Freeze parameters in the caller, but retain the gradient to the fused image.
    rgb = ((fused + 1) / 2).clamp(0, 1)
    mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    logits = judge((rgb - mean) / std)['out']
    if (mask != 255).any():
        return F.cross_entropy(logits, mask, ignore_index=255)
    return logits.sum() * 0


class Chapter3Loss(nn.Module):
    def __init__(self, clip_path, judge_weights=None):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(str(clip_path), local_files_only=True)
        self.clip.requires_grad_(False).eval()
        self.register_buffer('clip_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None])
        self.register_buffer('clip_std', torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None])
        self.judge = None
        if judge_weights:
            self.judge = deeplabv3_resnet50(weights=None, weights_backbone=None,
                                          num_classes=9, aux_loss=True)
            state = torch.load(judge_weights, map_location='cpu', weights_only=True)
            state = state.get('model', state)
            state = {k.removeprefix('module.'): v for k, v in state.items()}
            self.judge.load_state_dict(state, strict=True)
            self.judge.requires_grad_(False).eval()

    def forward(self, fused, batch):
        parts = pixel_losses(fused, batch['ir'], batch['vi'])
        pixels = F.interpolate((fused + 1) / 2, size=(224, 224), mode='bilinear', align_corners=False)
        image_embedding = self.clip.get_image_features((pixels - self.clip_mean) / self.clip_std)
        with torch.no_grad():
            text_embedding = self.clip.get_text_features(batch['input_ids'], batch['attention_mask'])
        parts['semantic'] = 1 - F.cosine_similarity(image_embedding, text_embedding).mean()
        parts['judge'] = judge_loss(self.judge, fused, batch['mask']) if self.judge is not None else fused.new_zeros(())
        total = sum(parts[k] for k in ('intensity', 'gradient', 'structure', 'color'))
        total = total + 0.1 * parts['semantic'] + 0.2 * parts['judge']
        return total, parts
