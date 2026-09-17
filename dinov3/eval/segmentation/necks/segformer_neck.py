# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""SegFormer 风格的轻量 all-MLP neck。

ViT backbone 只输出单尺度 1/16 特征，直接接线性 head 表达力弱。本模块从冻结 ViT
取多个中间层特征，按 SegFormer decode head 的做法融合成一张更“厚”的特征图：

    每层 1x1 conv 投影到统一维度 neck_dim
    -> bilinear 上采样到目标 stride (1/out_stride)
    -> concat
    -> 融合模块 (DWConv3x3 + BN + ReLU + Conv1x1 + BN + ReLU)
    -> [B, neck_dim, H/out_stride, W/out_stride]


注意：本模块刻意放在 segmentation/necks/（而非 segmentation/models/necks/），
使其 import 链保持轻量——不会触发 segmentation/models/__init__.py 里的
torchmetrics 与 MSDeformAttn(需编译 CUDA 算子) 依赖，方便检测路径直接复用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DWConvBNReLU(nn.Module):
    """Depthwise 3x3 conv + BatchNorm + ReLU（SegFormer 融合模块的前半段）。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.dwconv(x)))


class SegFormerNeck(nn.Module):
    """把 ViT 的多个单尺度中间层特征融合成一张多通道特征图。

    Args:
        in_channels_list: 每个输入特征的通道数列表，例如 ViT-L 取 4 层时为 [1024]*4。
        neck_dim: 每层投影后的统一通道数，也是 neck 的输出通道数。
        out_stride: 输出特征相对原图的下采样倍率。分割用 4（更锐利），检测用 16
            （= patch_size，不上采样，保持 DETR 的几何不变）。
        patch_size: ViT 的 patch 大小（输入特征本身的 stride）。
    """

    def __init__(self, in_channels_list, neck_dim: int = 256, out_stride: int = 4, patch_size: int = 16):
        super().__init__()
        self.in_channels_list = list(in_channels_list)
        self.neck_dim = neck_dim
        self.out_stride = out_stride
        self.patch_size = patch_size

        # 1x1 投影：把每一层特征统一到 neck_dim
        self.linears = nn.ModuleList(
            [nn.Conv2d(c, neck_dim, kernel_size=1, bias=False) for c in self.in_channels_list]
        )

        fused_dim = neck_dim * len(self.in_channels_list)
        self.fusion = nn.Sequential(
            _DWConvBNReLU(fused_dim),
            nn.Conv2d(fused_dim, neck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(neck_dim),
            nn.ReLU(inplace=True),
        )

        # 从 1/patch_size 上采样到 1/out_stride 的整数倍；out_stride==patch_size 时为 1（不上采样）
        assert patch_size % out_stride == 0, (
            f"patch_size({patch_size}) 必须能被 out_stride({out_stride}) 整除"
        )
        self.upsample_factor = max(1, patch_size // out_stride)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """features: list/tuple，每个元素为 [B, in_ch_i, h, w] 的 ViT 中间层特征。"""
        features = list(features)
        assert len(features) == len(self.linears), (
            f"neck 期望 {len(self.linears)} 层特征，实际收到 {len(features)}"
        )
        _, _, h, w = features[0].shape
        target = (h * self.upsample_factor, w * self.upsample_factor)

        outs = []
        for linear, feat in zip(self.linears, features):
            x = linear(feat)
            if x.shape[-2:] != target:
                x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
            outs.append(x)

        return self.fusion(torch.cat(outs, dim=1))
