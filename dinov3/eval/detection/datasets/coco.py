# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# ------------------------------------------------------------------------
# Deformable DETR / DINO
# COCO-format detection dataset.
# ------------------------------------------------------------------------
"""COCO detection dataset for training the DINO-DETR head.

__getitem__ returns (image_tensor, target) where target is a dict:
  - boxes:      [N, 4] normalized cxcywh in [0, 1]  (what the DETR criterion expects)
  - labels:     [N] contiguous class ids in 0..num_classes-1
  - image_id:   scalar tensor
  - orig_size:  [2] (h, w) of the original image
  - size:       [2] (h, w) after augmentation
  - area:       [N]
  - iscrowd:    [N]
"""
import os

import torch
import torch.utils.data
from PIL import Image

from ..util.box_ops import box_xyxy_to_cxcywh
from .transforms import make_coco_transforms

try:
    from pycocotools.coco import COCO
except ImportError:  # 让缺依赖时报错清晰，而不是 import 阶段就崩
    COCO = None


class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, img_folder: str, ann_file: str, transforms=None):
        if COCO is None:
            raise ImportError("读取 COCO 标注需要 pycocotools，请先安装：pip install pycocotools")
        if not os.path.isfile(ann_file):
            raise RuntimeError(f"标注文件不存在: {ann_file}")
        self.img_folder = img_folder
        self.coco = COCO(ann_file)  # 保留原始 COCO 对象，评测(COCOeval)时作为 GT
        self.ids = list(sorted(self.coco.imgs.keys()))
        if len(self.ids) == 0:
            raise RuntimeError(f"{ann_file} 里没有任何图片条目")
        self._transforms = transforms
        # COCO 的 category_id 可能不连续，映射到连续 label 0..C-1
        cat_ids = list(sorted(self.coco.getCatIds()))
        self.cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)

    def __len__(self):
        return len(self.ids)

    def _load_target(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annos = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        w, h = img_info["width"], img_info["height"]

        if len(annos) > 0:
            b = torch.as_tensor([a["bbox"] for a in annos], dtype=torch.float32).reshape(-1, 4)  # xywh abs
            b[:, 2:] += b[:, :2]  # -> xyxy abs
            b[:, 0::2].clamp_(min=0, max=w)
            b[:, 1::2].clamp_(min=0, max=h)
            boxes = b
            labels = torch.as_tensor([self.cat2label[a["category_id"]] for a in annos], dtype=torch.int64)
            area = torch.as_tensor([float(a.get("area", 0.0)) for a in annos], dtype=torch.float32)
            iscrowd = torch.as_tensor([int(a.get("iscrowd", 0)) for a in annos], dtype=torch.int64)
        else:  # 允许没有标注的图（criterion 会处理空 target）
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "image_id": torch.tensor(img_id),
            "orig_size": torch.tensor([h, w]),
            "size": torch.tensor([h, w]),
            "boxes": boxes,  # xyxy absolute；transforms 内部按此处理，__getitem__ 末尾转 normalized cxcywh
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
        }
        return target, img_info["file_name"]

    def __getitem__(self, index):
        img_id = self.ids[index]
        target, file_name = self._load_target(img_id)
        path = os.path.join(self.img_folder, file_name)
        img = Image.open(path).convert("RGB")

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # xyxy absolute -> normalized cxcywh（DETR criterion 期望的格式）
        boxes = target["boxes"]
        if boxes.numel() > 0:
            th, tw = target["size"].tolist()
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([tw, th, tw, th], dtype=torch.float32)
            target["boxes"] = boxes
        return img, target


def build_coco(image_set: str, config) -> CocoDetection:
    """Build a COCO dataset for `image_set` ("train"/"val") from config.

    Expected on-disk layout (matches the data-prep guide):
        <root>/<train_img_dir>/*.jpg + <root>/<train_img_dir>/<train_ann>
        <root>/<val_img_dir>/*.jpg   + <root>/<val_img_dir>/<val_ann>
    """
    ds = config.datasets
    if image_set == "train":
        img_folder = os.path.join(ds.root, ds.train_img_dir)
        ann_file = os.path.join(img_folder, ds.train_ann)
    else:
        img_folder = os.path.join(ds.root, ds.val_img_dir)
        ann_file = os.path.join(img_folder, ds.val_ann)

    transforms = make_coco_transforms(
        image_set,
        scales=list(config.transforms.scales),
        max_size=config.transforms.max_size,
        crop_min=config.transforms.crop_min,
        crop_max=config.transforms.crop_max,
        flip_prob=config.transforms.flip_prob,
        val_size=config.transforms.val_size,
        mean=tuple(config.transforms.mean),
        std=tuple(config.transforms.std),
    )
    return CocoDetection(img_folder, ann_file, transforms)
