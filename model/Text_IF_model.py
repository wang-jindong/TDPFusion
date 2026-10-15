import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from einops import rearrange
from transformers import CLIPTextModel


# =========================================================================
#  Part 0: Restormer Block (MDTA + GDFN)
#  专为底层视觉设计：通道注意力机制，天生免疫高分辨率 OOM，且能保留高清边缘细节
# =========================================================================

class Attention(nn.Module):
    """
    Multi-Dconv Head Transposed Attention (MDTA)
    计算通道维度 (C) 的注意力，不会因为图像分辨率过大而 OOM
    """
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 1x1 卷积生成 Q, K, V
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 3x3 深度可分离卷积，提取局部结构信息 (非常适合融合任务)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 1x1 卷积输出投影
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # [B, C, H, W] -> [B, 3*C, H, W]
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape为通道注意力格式: [B, Head, C/Head, H*W]
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化 (Restormer 标准操作)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算通道注意力: (B, Head, C/Head, H*W) @ (B, Head, H*W, C/Head) -> (B, Head, C/Head, C/Head)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 注意力与 V 相乘
        out = (attn @ v)

        # 还原回图像维度
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FeedForward(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (GDFN)
    """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        # 1x1 卷积升维 (乘以2是为了后面的门控机制)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 3x3 深度可分离卷积
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        # 1x1 卷积降维
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # 门控机制 (Gated)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    """
    Restormer Block
    Norm -> MDTA -> Norm -> GDFN
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# =========================================================================
#  Part 1: 辅助模块 (CLIP, DWT, Modulations)
# =========================================================================

try:
    from model.modules_local import StructureAwareSemanticGatedAttention
except ImportError:
    # Fallback if specific module file is missing
    class StructureAwareSemanticGatedAttention(nn.Module):
        def __init__(self, dim, text_dim=512, num_heads=8):
            super().__init__()
            # 简单的辅助注意力生成 Mask
            self.mask_conv = nn.Sequential(
                nn.Conv2d(dim, dim // 2, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(dim // 2, 1, 1),
                nn.Sigmoid()
            )
            self.proj = nn.Conv2d(dim, dim, 1)

        def forward(self, x, text_tokens):
            mask = self.mask_conv(x)
            x = x * (1 + mask)
            return self.proj(x), mask


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2;
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2];
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2];
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4;
        x_LH = -x1 + x2 - x3 + x4;
        x_HH = x1 - x2 - x3 + x4
        x_High = torch.cat((x_HL, x_LH, x_HH), dim=1)
        return x_LL, x_High


class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x_LL, x_High):
        C = x_LL.shape[1]
        x_HL = x_High[:, 0:C, :, :];
        x_LH = x_High[:, C:2 * C, :, :];
        x_HH = x_High[:, 2 * C:3 * C, :, :]
        x1 = (x_LL - x_HL - x_LH + x_HH) / 2;
        x2 = (x_LL - x_HL + x_LH - x_HH) / 2
        x3 = (x_LL + x_HL - x_LH - x_HH) / 2;
        x4 = (x_LL + x_HL + x_LH + x_HH) / 2
        B, C, H, W = x_LL.shape
        x = torch.zeros(B, C, H * 2, W * 2).to(x_LL.device)
        x[:, :, 0::2, 0::2] = x1;
        x[:, :, 1::2, 0::2] = x2
        x[:, :, 0::2, 1::2] = x3;
        x[:, :, 1::2, 1::2] = x4
        return x


class FrequencyAwareModulation(nn.Module):
    def __init__(self, dim, text_dim=512):
        super().__init__()
        self.dwt = DWT();
        self.idwt = IDWT()
        self.enhance_LL = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, groups=dim), nn.BatchNorm2d(dim), nn.SiLU(),
                                        nn.Conv2d(dim, dim, 1))
        self.enhance_High = nn.Sequential(nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3), nn.BatchNorm2d(dim * 3),
                                          nn.SiLU(), nn.Conv2d(dim * 3, dim * 3, 1))
        self.low_freq_gate = nn.Sequential(nn.Linear(text_dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.Sigmoid())
        self.high_freq_gate = nn.Sequential(nn.Linear(text_dim, dim), nn.ReLU(), nn.Linear(dim, dim * 3), nn.Sigmoid())
        self.refine = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, img_feat, text_embed):
        x_LL, x_High = self.dwt(img_feat)
        x_LL = x_LL + self.enhance_LL(x_LL)
        x_High = x_High + self.enhance_High(x_High)
        w_low = self.low_freq_gate(text_embed).view(-1, x_LL.shape[1], 1, 1)
        w_high = self.high_freq_gate(text_embed).view(-1, x_High.shape[1], 1, 1)
        out = self.idwt(x_LL * w_low, x_High * w_high)
        return img_feat + self.refine(out)


class TextEncoder(nn.Module):
    def __init__(self, frozen=True):
        super(TextEncoder, self).__init__()
        local_path = os.environ.get("TDPFUSION_CLIP_PATH", "./clip_weights")
        print(f"Loading local CLIP weights from: {local_path} ...")

        try:
            self.clip = CLIPTextModel.from_pretrained(local_path)
            print("✅ Local CLIP model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Local CLIP load failed: {e}. Trying online...")
            self.clip = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        if frozen:
            for param in self.clip.parameters():
                param.requires_grad = False
        self.out_dim = 512

    def forward(self, input_ids, attention_mask):
        outputs = self.clip(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


# =========================================================================
#  Part 2: FeatureWiseAffine
# =========================================================================

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, text_dim=512, use_affine_level=True):
        super().__init__()
        self.use_affine_level = use_affine_level
        total_in_dim = text_dim + in_channels
        self.MLP = nn.Sequential(
            nn.Linear(total_in_dim, out_channels * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(out_channels * 2, out_channels * (1 + use_affine_level))
        )
        self.context_norm = nn.LayerNorm(total_in_dim)

    def forward(self, x, text_embed):
        if self.use_affine_level:
            B, C, H, W = x.shape
            img_context = torch.mean(x, dim=(2, 3))
            if text_embed.dim() == 3: text_embed = text_embed.mean(dim=1)
            combined_context = torch.cat([text_embed, img_context], dim=1)
            combined_context = self.context_norm(combined_context)
            params = self.MLP(combined_context)
            gamma, beta = params.view(B, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x


# =========================================================================
#  Part 3: 主模型 Text_IF (Transformer Backbone)
# =========================================================================

class TransformerEncoder(nn.Module):
    def __init__(self, inp_channels=3, dim=32, num_blocks=[2, 3, 3, 4], heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(TransformerEncoder, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.level1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)

        self.level2 = nn.Sequential(*[
            TransformerBlock(int(dim * 2), heads[1], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(int(dim * 2))

        self.level3 = nn.Sequential(*[
            TransformerBlock(int(dim * 4), heads[2], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(int(dim * 4))

        self.level4 = nn.Sequential(*[
            TransformerBlock(int(dim * 8), heads[3], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[3])
        ])

    def forward(self, x):
        x = self.patch_embed(x)
        out1 = self.level1(x)
        x = self.down1_2(out1)
        out2 = self.level2(x)
        x = self.down2_3(out2)
        out3 = self.level3(x)
        x = self.down3_4(out3)
        out4 = self.level4(x)
        return out4, out3, out2, out1


class Text_IF(nn.Module):
    def __init__(self, inp_A_channels=1, inp_B_channels=3, out_channels=3, dim=48,
                 num_blocks=[2, 2, 2, 2], num_refinement_blocks=4, heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(Text_IF, self).__init__()
        self.text_encoder = TextEncoder()
        text_dim = self.text_encoder.out_dim

        # Encoders
        self.encoder_A = TransformerEncoder(inp_A_channels, dim, num_blocks, heads, ffn_expansion_factor, bias,
                                            LayerNorm_type)
        self.encoder_B = TransformerEncoder(inp_B_channels, dim, num_blocks, heads, ffn_expansion_factor, bias,
                                            LayerNorm_type)

        # Bottleneck
        dim_l4 = dim * 8
        self.cross_attention = Cross_attention(dim_l4)
        self.feature_fusion_4 = Fusion_Embed(dim_l4)
        self.freq_modulation = FrequencyAwareModulation(dim_l4, text_dim)
        self.ssga_module = StructureAwareSemanticGatedAttention(dim_l4, text_dim, num_heads=8)
        self.dual_stream_fusion = nn.Conv2d(dim_l4 * 2, dim_l4, 1, bias=bias)

        # Decoder Level 4
        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(dim_l4, heads[3], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[3])
        ])

        # Decoder Level 3
        dim_l3 = dim * 4
        self.up4_3 = Upsample(dim_l4)
        self.reduce_chan_level3 = nn.Conv2d(dim_l4, dim_l3, 1, bias=bias)
        self.feature_fusion_3 = Fusion_Embed(dim_l3)
        self.prompt_guidance_3 = FeatureWiseAffine(in_channels=dim_l3, out_channels=dim_l3, text_dim=text_dim)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim_l3, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        # Decoder Level 2
        dim_l2 = dim * 2
        self.up3_2 = Upsample(dim_l3)
        self.reduce_chan_level2 = nn.Conv2d(dim_l3, dim_l2, 1, bias=bias)
        self.feature_fusion_2 = Fusion_Embed(dim_l2)
        self.prompt_guidance_2 = FeatureWiseAffine(in_channels=dim_l2, out_channels=dim_l2, text_dim=text_dim)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim_l2, heads[1], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        # Decoder Level 1
        dim_l1 = dim
        self.up2_1 = Upsample(dim_l2)
        self.reduce_chan_level1 = nn.Conv2d(dim_l1 * 2, dim_l1, 1, bias=bias)
        self.feature_fusion_1 = Fusion_Embed(dim_l1)
        self.prompt_guidance_1 = FeatureWiseAffine(in_channels=dim_l1, out_channels=dim_l1, text_dim=text_dim)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim_l1, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        # Refinement
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim_l1, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(dim_l1, out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp_img_A, inp_img_B, input_ids, attention_mask):
        text_tokens = self.text_encoder(input_ids, attention_mask)
        text_global = text_tokens.mean(dim=1)

        out4_A, out3_A, out2_A, out1_A = self.encoder_A(inp_img_A)
        out4_B, out3_B, out2_B, out1_B = self.encoder_B(inp_img_B)

        ca_A, ca_B = self.cross_attention(out4_A, out4_B)
        feat_base = self.feature_fusion_4(ca_A, ca_B)
        feat_global = self.freq_modulation(feat_base, text_global)
        feat_local, pred_mask = self.ssga_module(out4_A, text_tokens)

        feat_fused = self.dual_stream_fusion(torch.cat([feat_global, feat_local], dim=1))
        out_dec4 = self.decoder_level4(feat_fused)

        # Decode
        inp_dec3 = self.reduce_chan_level3(torch.cat(
            [self.prompt_guidance_3(self.up4_3(out_dec4), text_global), self.feature_fusion_3(out3_A, out3_B)], 1))
        out_dec3 = self.decoder_level3(inp_dec3)

        # Decoder Level 2
        inp_dec2 = self.reduce_chan_level2(torch.cat(
            [self.prompt_guidance_2(self.up3_2(out_dec3), text_global), self.feature_fusion_2(out2_A, out2_B)], 1))
        out_dec2 = self.decoder_level2(inp_dec2)

        # Decoder Level 1
        inp_dec1 = self.reduce_chan_level1(torch.cat(
            [self.prompt_guidance_1(self.up2_1(out_dec2), text_global), self.feature_fusion_1(out1_A, out1_B)], 1))
        out_dec1 = self.decoder_level1(inp_dec1)

        out_final = torch.tanh(self.output(self.refinement(out_dec1)))

        if pred_mask is not None:
            pred_mask = F.interpolate(pred_mask, size=out_final.shape[-2:], mode='bilinear', align_corners=False)

        return out_final, pred_mask


# =========================================================================
#  Part 4: Utils (Standard Blocks)
# =========================================================================

class Fusion_Embed(nn.Module):
    def __init__(self, dim, bias=False): super().__init__(); self.fusion_proj = nn.Conv2d(dim * 2, dim, 1, 1, bias=bias)

    def forward(self, x_A, x_B): return self.fusion_proj(torch.cat([x_A, x_B], dim=1))


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape): super().__init__(); self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x): return x / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape): super().__init__(); self.weight = nn.Parameter(
        torch.ones(normalized_shape)); self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x): return (x - x.mean(-1, keepdim=True)) / torch.sqrt(
        x.var(-1, keepdim=True, unbiased=False) + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type): super().__init__(); self.body = WithBias_LayerNorm(
        dim) if LayerNorm_type != 'BiasFree' else BiasFree_LayerNorm(dim)

    def forward(self, x): return to_4d(self.body(to_3d(x)), x.shape[-2], x.shape[-1])


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False): super().__init__(); self.proj = nn.Conv2d(in_c, embed_dim, 3,
                                                                                                    1, 1, bias=bias)

    def forward(self, x): return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat): super().__init__(); self.body = nn.Sequential(
        nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2))

    def forward(self, x): return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat): super().__init__(); self.body = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False))

    def forward(self, x): return self.body(x)


class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=8, norm_groups=16):
        super().__init__();
        self.n_head = n_head;
        self.norm_A = nn.GroupNorm(norm_groups, in_channel);
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False);
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)
        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False);
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        b, c, h, w = x_A.shape;
        head_dim = c // self.n_head
        qkv_A = self.qkv_A(self.norm_A(x_A)).view(b, self.n_head, head_dim * 3, h, w);
        q_A, k_A, v_A = qkv_A.chunk(3, dim=2)
        qkv_B = self.qkv_B(self.norm_B(x_B)).view(b, self.n_head, head_dim * 3, h, w);
        q_B, k_B, v_B = qkv_B.chunk(3, dim=2)

        # 考虑到这里分辨率很低 (Level 4)，标准 attention 没问题。
        attn_A = torch.softmax(
            torch.einsum("bnchw, bncyx -> bnhwyx", q_B, k_A).contiguous().view(b, self.n_head, h, w, -1) / math.sqrt(c),
            -1).view(b, self.n_head, h, w, h, w)
        out_A = self.out_A(torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, v_A).reshape(b, c, h, w)) + x_A
        attn_B = torch.softmax(
            torch.einsum("bnchw, bncyx -> bnhwyx", q_A, k_B).contiguous().view(b, self.n_head, h, w, -1) / math.sqrt(c),
            -1).view(b, self.n_head, h, w, h, w)
        out_B = self.out_B(torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, v_B).reshape(b, c, h, w)) + x_B
        return out_A, out_B


def to_3d(x): return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w): return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)