# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# ------------------------------------------------------------------------
"""Detection data augmentation. Boxes are handled as ABSOLUTE xyxy inside the
transforms; conversion to normalized cxcywh happens in the dataset __getitem__.
"""
import random

import PIL
import torch
import torchvision.transforms.functional as F


def _get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return (oh, ow)


def _get_size(image_size, size, max_size=None):
    if isinstance(size, (list, tuple)):
        return size[::-1]  # (h, w) -> (w, h) for F.resize
    return _get_size_with_aspect_ratio(image_size, size, max_size)


def resize(image, target, size, max_size=None):
    size = _get_size(image.size, size, max_size)  # (w, h)
    rescaled_image = F.resize(image, size[::-1])  # F.resize wants (h, w)
    if target is None:
        return rescaled_image, None

    w, h = size
    ow, oh = image.size
    ratios = (float(w) / float(ow), float(h) / float(oh))
    if "boxes" in target:
        boxes = target["boxes"]  # xyxy absolute
        scaled_boxes = boxes * torch.as_tensor([ratios[0], ratios[1], ratios[0], ratios[1]], dtype=torch.float32)
        target["boxes"] = scaled_boxes
    if "area" in target:
        area = target["area"]
        target["area"] = area * (ratios[0] * ratios[1])
    target["size"] = torch.tensor([h, w])
    return rescaled_image, target


def crop(image, target, region):
    cropped_image = F.crop(image, *region)
    target = target.copy()
    i, j, h, w = region
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]
    if "boxes" in target:
        boxes = target["boxes"]  # xyxy absolute
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i], dtype=torch.float32)
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    # keep only boxes with non-zero area
    if "boxes" in target:
        cropped_boxes = target["boxes"].reshape(-1, 2, 2)
        keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        for field in fields:
            if field in target:
                target[field] = target[field][keep]
    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)
    w, h = image.size
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]  # xyxy absolute
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1], dtype=torch.float32) + torch.as_tensor(
            [w, 0, w, 0], dtype=torch.float32
        )
        target["boxes"] = boxes
    return flipped_image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        return self.__class__.__name__ + "(" + repr(self.transforms) + ")"


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize:
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = F.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class RandomSelect:
    """Randomly select between transforms1 and transforms2 with probability p for transforms2."""

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms2(img, target)
        return self.transforms1(img, target)


class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        return image, target


def make_coco_transforms(image_set, *, scales, max_size, crop_min, crop_max, flip_prob, val_size, mean, std):
    normalize = Compose([ToTensor(), Normalize(mean, std)])
    if image_set == "train":
        return Compose(
            [
                RandomHorizontalFlip(flip_prob),
                RandomSelect(
                    RandomResize(scales, max_size=max_size),
                    Compose([RandomSizeCrop(crop_min, crop_max), RandomResize(scales, max_size=max_size)]),
                ),
                normalize,
            ]
        )
    if image_set in ("val", "test"):
        return Compose([RandomResize([val_size], max_size=max_size), normalize])
    raise ValueError(f"unknown image_set {image_set}")
