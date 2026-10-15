import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(SemanticLoss, self).__init__()
        # 使用标准的交叉熵损失
        # ignore_index=255 用于忽略 mask 中的未标注区域
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, seg_logits, gt_mask):
        """
        :param seg_logits: 裁判网络的输出 [B, 9, H, W] (9是类别数)
        :param gt_mask: 真实的语义标签 [B, H, W]
        """
        # 确保 gt_mask 是 Long 类型且去掉了 Channel 维度
        if len(gt_mask.shape) == 4:
            gt_mask = gt_mask.squeeze(1)

        return self.criteria(seg_logits, gt_mask)