# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""SegFormer neck + 语义分割分类头。

作为 FeatureDecoder 的可训练 module[1]，接收冻结 backbone（module[0]）输出的多层特征，
先经 SegFormerNeck 融合，再用 1x1 conv 分类。暴露 forward/predict 两个入口，与
LinearHead 的契约一致，因此 make_inference/slide_inference 无需改动（非 "m2f" 类型
一律按“返回 logits 张量”处理）。
"""

import torch.nn as nn
import torch.nn.functional as F

from dinov3.eval.segmentation.necks import SegFormerNeck


class SegFormerSegHead(nn.Module):
    def __init__(
        self,
        in_channels_list,
        num_classes: int,
        neck_dim: int = 256,
        out_stride: int = 4,
        patch_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.neck = SegFormerNeck(
            in_channels_list=in_channels_list,
            neck_dim=neck_dim,
            out_stride=out_stride,
            patch_size=patch_size,
        )
        self.dropout = nn.Dropout2d(dropout)
        self.conv_seg = nn.Conv2d(neck_dim, num_classes, kernel_size=1, padding=0, stride=1)
        nn.init.normal_(self.conv_seg.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv_seg.bias, 0)

    def forward(self, features):
        """训练/前向：带 dropout。features 为 backbone 输出的多层特征。"""
        x = self.neck(features)
        x = self.dropout(x)
        return self.conv_seg(x)

    def predict(self, features, rescale_to=(512, 512)):
        """评测：不使用 dropout，输出插值回 GT 尺寸以计算指标。"""
        x = self.neck(features)
        x = self.conv_seg(x)
        return F.interpolate(x, size=rescale_to, mode="bilinear", align_corners=False)
