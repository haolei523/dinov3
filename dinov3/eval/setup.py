# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from dataclasses import dataclass
from typing import Tuple, TypedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from dinov3.configs import DinoV3SetupArgs, setup_config
from dinov3.models import build_model_for_eval


@dataclass
class ModelConfig:
    # Loading a local file
    config_file: str | None = None
    pretrained_weights: str | None = None
    # Loading a DINOv3 or v2 model from torch.hub
    dino_hub: str | None = None


class BaseModelContext(TypedDict):
    """
    An object that contains the context of a model (autocast, description, ...)
    """

    autocast_dtype: torch.dtype  # default could be torch.float


def _load_dinov3_backbone_local(arch_name: str, weights_path: str) -> nn.Module:
    """离线加载 DINOv3 backbone：用仓库内的 hub 构建函数搭出与联网版逐参数一致的架构，
    再以 strict=True 加载本地权重。

    一套逻辑覆盖全部变体（vits16 / vits16plus / vitb16 / vitl16 / vitl16plus /
    vith16plus / vit7b16 及 convnext_*），无需为每个模型准备 config yaml；plus 系列
    （swiglu + ffn_ratio=6）也能正确构建（config_file 路径无法表达 ffn_ratio）。

    strict=True：键名不匹配会立即报错，避免离线加载被静默变成随机权重。
    """
    import dinov3.hub.backbones as backbones

    builder = getattr(backbones, arch_name, None)
    if builder is None:
        raise ValueError(f"未知的 backbone 名称: {arch_name!r}（可选见 dinov3/hub/backbones.py）")

    # pretrained=False：只搭架构 + 随机初始化，随后被本地权重完全覆盖，全程不联网。
    model = builder(pretrained=False)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    # 兼容训练 checkpoint：取出 teacher / model / state_dict 子字典。
    if isinstance(state_dict, dict):
        for key in ("teacher", "model", "state_dict"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    model.load_state_dict(state_dict, strict=True)
    return model


def load_model_and_context(model_config: ModelConfig, output_dir: str) -> tuple[torch.nn.Module, BaseModelContext]:
    if model_config.dino_hub is not None:
        assert model_config.config_file is None, "dino_hub 与 config_file 互斥，二选一"
        if model_config.pretrained_weights is not None:
            # 离线：本地 hub 函数搭架构 + 严格加载本地权重（适配所有变体，含 plus 系列）。
            model = _load_dinov3_backbone_local(
                arch_name=model_config.dino_hub,
                weights_path=model_config.pretrained_weights,
            )
            base_model_context = BaseModelContext(autocast_dtype=torch.bfloat16)
        else:
            # 联网：torch.hub 自动下载代码 + 官方权重。
            if "dinov3" in model_config.dino_hub:
                repo = "dinov3"
            elif "dinov2" in model_config.dino_hub:
                repo = "dinov2"
            else:
                raise ValueError
            model = torch.hub.load(f"facebookresearch/{repo}", model_config.dino_hub)
            base_model_context = BaseModelContext(autocast_dtype=torch.float)
    else:
        model, base_model_context = setup_and_build_model(
            config_file=model_config.config_file,
            pretrained_weights=model_config.pretrained_weights,
            output_dir=output_dir,
        )

    model.cuda()
    model.eval()
    return model, base_model_context


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.param_dtype
    if teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def setup_and_build_model(
    config_file: str,
    pretrained_weights: str | None = None,
    shard_unsharded_model: bool = False,
    output_dir: str = "",
    opts: list | None = None,
    **ignored_kwargs,
) -> Tuple[nn.Module, BaseModelContext]:
    cudnn.benchmark = True
    del ignored_kwargs
    setup_args = DinoV3SetupArgs(
        config_file=config_file,
        pretrained_weights=pretrained_weights,
        shard_unsharded_model=shard_unsharded_model,
        output_dir=output_dir,
        opts=opts or [],
    )
    config = setup_config(setup_args, strict_cfg=False)
    model = build_model_for_eval(config, setup_args.pretrained_weights)
    autocast_dtype = get_autocast_dtype(config)
    return model, BaseModelContext(autocast_dtype=autocast_dtype)
