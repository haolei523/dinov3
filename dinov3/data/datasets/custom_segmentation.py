# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

from .decoders import Decoder, DenseTargetDecoder, ImageDataDecoder
from .extended import ExtendedVisionDataset


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"

    @property
    def dirname(self) -> str:
        # 子目录名。若你的目录叫 training/validation，改这里即可。
        return self.value


# 允许的图片后缀（按需增减）
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _list_images(images_dir: str) -> List[str]:
    if not os.path.isdir(images_dir):
        raise RuntimeError(f'图片目录不存在: "{images_dir}"')
    names = [n for n in os.listdir(images_dir) if n.lower().endswith(_IMG_EXTS)]
    return sorted(names)


class CustomSegmentation(ExtendedVisionDataset):
    """自定义语义分割数据集（图像 + 逐像素掩码）。

    期望的目录结构（如与实际不同，改下方 IMAGES_DIR / MASKS_DIR / MASK_EXT / _Split.dirname）：

        <root>/
        ├── train/                 # _Split.TRAIN.dirname
        │   ├── images/  xxx.jpg   # 原始图片，支持多种后缀
        │   └── masks/   xxx.png   # 掩码，与图片“同名不同后缀”配对
        └── val/                   # _Split.VAL.dirname
            ├── images/
            └── masks/

    掩码要求：
        - 单通道（灰度）PNG，像素值 = 类别索引（0 .. num_classes-1），255 = ignore（不计损失）。
        - 若你的掩码是 RGB 调色板图（每种颜色一个类），需先转成上面的“索引图”，
          转换脚本见本文件末尾的注释。

    与仓库内置 ADE20K 数据集接口完全一致，因此可无缝复用分割训练/评测流程与 transforms。
    """

    Split = Union[_Split]

    IMAGES_DIR = "images"  # 图片子目录名
    MASKS_DIR = "masks"  # 掩码子目录名
    MASK_EXT = ".png"  # 掩码文件后缀

    def __init__(
        self,
        split: "CustomSegmentation.Split",
        root: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder: Decoder = ImageDataDecoder,
        target_decoder: Decoder = DenseTargetDecoder,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=image_decoder,
            target_decoder=target_decoder,
        )
        self.image_paths, self.target_paths = self._load_file_paths(root, split)

    def _load_file_paths(self, root: str, split: _Split) -> Tuple[List[str], List[str]]:
        images_dir = os.path.join(root, split.dirname, self.IMAGES_DIR)
        masks_dir = os.path.join(root, split.dirname, self.MASKS_DIR)
        image_paths, target_paths = [], []
        for name in _list_images(images_dir):
            stem = os.path.splitext(name)[0]
            mask_name = stem + self.MASK_EXT
            # 跳过没有对应掩码的图片，避免训练时报错
            if not os.path.isfile(os.path.join(masks_dir, mask_name)):
                continue
            image_paths.append(os.path.join(split.dirname, self.IMAGES_DIR, name))
            target_paths.append(os.path.join(split.dirname, self.MASKS_DIR, mask_name))
        if len(image_paths) == 0:
            raise RuntimeError(
                f'在 "{images_dir}" 下没有找到任何“图片-掩码”配对样本，请检查目录结构与文件后缀。'
            )
        return image_paths, target_paths

    def get_image_data(self, index: int) -> bytes:
        with open(os.path.join(self.root, self.image_paths[index]), mode="rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        with open(os.path.join(self.root, self.target_paths[index]), mode="rb") as f:
            return f.read()

    def __len__(self) -> int:
        return len(self.image_paths)


# --------------------------------------------------------------------------------------
# 附：把 RGB 调色板掩码转成“索引图”的一次性脚本（仅当你的掩码是彩色 RGB 时才需要）。
# 用法：改好 PALETTE / 路径后，单独运行本段（或复制到别的脚本里跑）。
#
#   import numpy as np
#   from PIL import Image
#   import glob, os
#   # 类别颜色表：index -> (R, G, B)。顺序即类别 id。
#   PALETTE = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]  # <== 改成你的调色板
#   color2id = {c: i for i, c in enumerate(PALETTE)}
#   for src in glob.glob("masks_rgb/*.png"):
#       rgb = np.array(Image.open(src).convert("RGB"))
#       idx = np.zeros(rgb.shape[:2], dtype=np.uint8)
#       for (r, g, b), i in color2id.items():
#           idx[(rgb[..., 0] == r) & (rgb[..., 1] == g) & (rgb[..., 2] == b)] = i
#       Image.fromarray(idx).save(os.path.join("masks", os.path.basename(src)))
# --------------------------------------------------------------------------------------
