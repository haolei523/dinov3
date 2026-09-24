# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
import logging
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..util.misc import NestedTensor
from .position_encoding import build_position_encoding
from .utils import LayerNorm2D
from .windows import WindowsWrapper
from dinov3.eval.segmentation.necks import SegFormerNeck

logger = logging.getLogger("dinov3")


class DINOBackbone(nn.Module):
    def __init__(
        self,
        backbone_model: nn.Module,
        train_backbone: bool,
        blocks_to_train: Optional[List[str]] = None,
        layers_to_use: Union[int, List] = 1,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.backbone = backbone_model
        self.blocks_to_train = blocks_to_train
        self.patch_size = self.backbone.patch_size
        self.use_layernorm = use_layernorm

        for _, (name, parameter) in enumerate(self.backbone.named_parameters()):
            train_condition = any(f".{b}." in name for b in self.blocks_to_train) if self.blocks_to_train else True
            if (not train_backbone) or "mask_token" in name or (not train_condition):
                parameter.requires_grad_(False)

        self.strides = [self.backbone.patch_size]

        # get embed_dim for each intermediate output
        n_all_layers = self.backbone.n_blocks
        blocks_to_take = (
            range(n_all_layers - layers_to_use, n_all_layers) if isinstance(layers_to_use, int) else layers_to_use
        )

        # if models do not define embed_dims, repeat embed_dim n_blocks times
        embed_dims = getattr(self.backbone, "embed_dims", [self.backbone.embed_dim] * self.backbone.n_blocks)
        embed_dims = [embed_dims[i] for i in range(n_all_layers) if i in blocks_to_take]

        if self.use_layernorm:
            self.layer_norms = nn.ModuleList([LayerNorm2D(embed_dim) for embed_dim in embed_dims])

        self.num_channels = [sum(embed_dims)]
        self.layers_to_use = layers_to_use

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone.get_intermediate_layers(tensor_list.tensors, n=self.layers_to_use, reshape=True)
        if self.use_layernorm:
            xs = [ln(x).contiguous() for ln, x in zip(self.layer_norms, xs)]

        xs = [torch.cat(xs, axis=1)]

        out: list[NestedTensor] = []
        for x in xs:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(x, mask))
        return out


class NeckBackbone(nn.Module):
    """冻结 ViT + SegFormer 式 neck：取多个中间层特征，融合成单尺度特征图喂给 DETR。

    输出 stride 与 patch_size 一致（neck 内不做上采样，out_stride=patch_size），因此 DETR 的
    position encoding / proposal 几何与 DINOBackbone 完全一致；区别只是把“拼接多层通道”换成
    “neck 融合”。neck 可训练，ViT 冻结（前向在 no_grad 下跑，不存 ViT 激活以省显存）。
    """

    def __init__(self, backbone_model: nn.Module, layers_to_use: Union[int, List], neck_dim: int = 256):
        super().__init__()
        self.backbone = backbone_model
        # Important: we freeze the backbone
        self.backbone.requires_grad_(False)
        self.patch_size = self.backbone.patch_size
        self.layers_to_use = layers_to_use

        n_all_layers = self.backbone.n_blocks
        blocks_to_take = (
            range(n_all_layers - layers_to_use, n_all_layers) if isinstance(layers_to_use, int) else layers_to_use
        )
        embed_dims = getattr(self.backbone, "embed_dims", [self.backbone.embed_dim] * n_all_layers)
        in_channels_list = [embed_dims[i] for i in range(n_all_layers) if i in blocks_to_take]

        # out_stride=patch_size -> neck 不上采样，保持 DETR 几何不变
        self.neck = SegFormerNeck(
            in_channels_list=in_channels_list,
            neck_dim=neck_dim,
            out_stride=self.patch_size,
            patch_size=self.patch_size,
        )
        self.strides = [self.patch_size]
        self.num_channels = [neck_dim]

    def forward(self, tensor_list: NestedTensor):
        with torch.no_grad():  # backbone 冻结，无需存激活
            xs = self.backbone.get_intermediate_layers(tensor_list.tensors, n=self.layers_to_use, reshape=True)
        feat = self.neck(xs)

        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
        return [NestedTensor(feat, mask)]


class BackboneWithPositionEncoding(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        out: List[NestedTensor] = list(self[0](tensor_list))
        pos = [self[1][idx](x).to(x.tensors.dtype) for idx, x in enumerate(out)]
        return out, pos


def build_backbone(backbone_model, args):
    position_embedding = build_position_encoding(args)
    train_backbone = False
    if getattr(args, "use_neck", False):
        layers_to_use = args.neck_out_layers if args.neck_out_layers is not None else args.layers_to_use
        logger.info(f"Using SegFormerNeck backbone (neck_dim={args.neck_dim}, layers={layers_to_use})")
        backbone = NeckBackbone(backbone_model, layers_to_use, neck_dim=args.neck_dim)
    else:
        backbone = DINOBackbone(
            backbone_model, train_backbone, args.blocks_to_train, args.layers_to_use, args.backbone_use_layernorm
        )
    if args.n_windows_sqrt > 0:
        logger.info(f"Wrapping with {args.n_windows_sqrt} x {args.n_windows_sqrt} windows")
        backbone = WindowsWrapper(
            backbone, n_windows_w=args.n_windows_sqrt, n_windows_h=args.n_windows_sqrt, patch_size=backbone.patch_size
        )
    else:
        logger.info("Not wrapping with windows")

    return BackboneWithPositionEncoding(backbone, position_embedding)
