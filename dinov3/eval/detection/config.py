# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from dataclasses import dataclass, field
from enum import Enum

import torch
from omegaconf import MISSING

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dinov3.eval.setup import ModelConfig

from .models.position_encoding import PositionEncoding


# Detection images go through ToTensor ([0, 1]) then Normalize, so mean/std are
# the usual 0-1 ImageNet stats (NOT the 0-255 scale used by segmentation).
DEFAULT_MEAN = tuple(IMAGENET_DEFAULT_MEAN)
DEFAULT_STD = tuple(IMAGENET_DEFAULT_STD)


class ModelDtype(Enum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"

    @property
    def autocast_dtype(self):
        return {
            ModelDtype.BFLOAT16: torch.bfloat16,
            ModelDtype.FLOAT32: torch.float32,
        }[self]


@dataclass
class MatcherConfig:
    """Hungarian matcher costs (see criterion.HungarianMatcher)."""

    cost_class: float = 2.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0


@dataclass
class LossConfig:
    """SetCriterion weights (see criterion.SetCriterion)."""

    focal_alpha: float = 0.25
    cls_weight: float = 2.0
    bbox_weight: float = 5.0
    giou_weight: float = 2.0


@dataclass(kw_only=True)
class DetectionHeadConfig:
    num_classes: int = 91  # 91 classes in COCO
    # Deformable DETR tricks
    with_box_refine: bool = True
    two_stage: bool = True
    # DINO DETR tricks
    mixed_selection: bool = True
    look_forward_twice: bool = True  # was default False
    # Hybrid Matching tricks
    k_one2many: int = 6  # was 5
    lambda_one2many: float = 1.0
    num_queries_one2one: int = 300  # number of query slots for one_to_one matching
    num_queries_one2many: int = 1500  # was 0, number of query slots for one_to_many matching
    """
    Absolute coordinates & box regression reparameterization.
    If true, we use absolute coordindates & reparameterization for bounding boxes.
    """
    reparam: bool = True
    topk: int = 100

    # * Backbone
    # type of positional embedding to use on top of the image features
    position_embedding: PositionEncoding = PositionEncoding.SINE
    num_feature_levels: int = 1  # number of feature levels

    # * Transformer
    dec_layers: int = 6  # number of decoding layers in the transformer
    dim_feedforward: int = 2048  # intermediate size of the feedforward layers in the transformer blocks
    hidden_dim: int = 256  # size of the embeddings (dimension of the transformer)
    dropout: float = 0.0  # dropout applied in the transformer, was 0.1
    nheads: int = 8  # number of attention heads inside the transformer's attentions
    norm_type: str = "pre_norm"

    # Loss
    aux_loss: bool = True  # auxiliary decoding losses (loss at each layer)

    # * dev: proposals
    proposal_feature_levels: int = 4  # was 1
    proposal_min_size: int = 50
    # * dev decoder: global decoder
    decoder_type: str = "global_rpe_decomp"  # was deform
    decoder_use_checkpoint: bool = False
    decoder_rpe_hidden_dim: int = 512
    decoder_rpe_type: str = "linear"

    # Custom
    add_transformer_encoder: bool = True
    num_encoder_layers: int = 6
    layers_to_use: list[int] | None = None
    blocks_to_train: list[int] | None = None
    n_windows_sqrt: int = 0
    proposal_in_stride: int | None = None
    proposal_tgt_strides: list[int] | None = None
    backbone_use_layernorm: bool = False  # whether to use layernorm on each layer of the backbone's features

    # * Neck (SegFormer-style decoder fused on top of frozen ViT features)
    use_neck: bool = False  # True: 用 SegFormerNeck 融合多层特征再喂 DETR; False: 沿用 DINOBackbone 拼接
    neck_dim: int = 256  # neck 输出通道数 (DETR 的 input_proj 会自动适配)
    neck_out_layers: list[int] | None = None  # neck 取哪些 ViT 中间层; None 时复用 layers_to_use

    # Matcher / loss sub-configs live here so that build_model() and build_criterion()
    # share ONE source of truth for num_classes, dec_layers, two_stage, k_one2many, ...
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class OptimizerConfig:
    lr: float = 1e-4  # DETR heads train well with a small lr; backbone is frozen
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-4
    gradient_clip: float = 0.1  # max_norm for clip_grad_norm_


@dataclass
class SchedulerConfig:
    total_epochs: int = 12
    lr_drop_epochs: list[int] = field(default_factory=lambda: [10])  # MultiStepLR milestones
    gamma: float = 0.1


@dataclass
class DetectionDatasetConfig:
    root: str = MISSING  # dataset root; images + ann file live under <root>/<img_dir>/<ann>
    train_img_dir: str = "train"
    train_ann: str = "_annotations.coco.json"
    val_img_dir: str = "val"
    val_ann: str = "_annotations.coco.json"


@dataclass
class DetectionTransformConfig:
    scales: list[int] = field(default_factory=lambda: [480, 512, 544, 576, 608])
    max_size: int = 1024
    crop_min: int = 400
    crop_max: int = 600
    flip_prob: float = 0.5
    val_size: int = 512
    mean: tuple[float, ...] = DEFAULT_MEAN
    std: tuple[float, ...] = DEFAULT_STD


@dataclass
class DetectionEvalConfig:
    eval_interval: int = 2  # run COCOeval every N epochs
    topk: int = 100  # top-k boxes kept by PostProcess for evaluation


@dataclass
class DetectionTrainConfig:
    model: ModelConfig | None = None  # DINOv3 backbone, loaded via load_model_and_context
    head: DetectionHeadConfig = field(default_factory=DetectionHeadConfig)
    datasets: DetectionDatasetConfig = field(default_factory=DetectionDatasetConfig)
    transforms: DetectionTransformConfig = field(default_factory=DetectionTransformConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    eval: DetectionEvalConfig = field(default_factory=DetectionEvalConfig)

    bs: int = 1  # batch size per GPU; keep small for <=8GB
    num_workers: int = 2
    seed: int = 0
    model_dtype: ModelDtype = ModelDtype.BFLOAT16  # bf16 autocast saves memory

    output_dir: str | None = None
    load_from: str | None = None  # optional path to a pretrained head checkpoint
