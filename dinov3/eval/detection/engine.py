# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# ------------------------------------------------------------------------
# DINO / Deformable DETR
# Training loop + COCO evaluation.
# ------------------------------------------------------------------------
import copy
import math

import torch

import dinov3.distributed as distributed
from dinov3.logging import MetricLogger, SmoothedValue

from .util.misc import reduce_dict

try:
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCOeval = None


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    max_norm=0.0,
    autocast_dtype=None,
):
    model.train()
    criterion.train()
    # 冻结的 backbone 保持 eval（关掉 dropout/droppath，特征更稳定）
    unwrapped = model.module if hasattr(model, "module") else model
    if hasattr(unwrapped, "backbone"):
        unwrapped.backbone.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 50

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_value = losses.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training.\n{loss_dict}")

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        metric_logger.update(loss=losses_reduced_scaled, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class CocoEvaluator:
    """收集预测并用 pycocotools 跑 COCOeval（支持多卡结果聚合）。"""

    def __init__(self, coco_gt, label2cat=None):
        self.coco_gt = copy.deepcopy(coco_gt)
        self.label2cat = label2cat or {}
        self.results = []
        self.coco_eval = None
        self.stats = None

    def update(self, predictions):
        # predictions: {image_id: {"scores","labels","boxes"(xyxy absolute)}}
        for image_id, pred in predictions.items():
            scores = pred["scores"].tolist()
            labels = pred["labels"].tolist()
            boxes = pred["boxes"].tolist()
            for s, l, b in zip(scores, labels, boxes):
                self.results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(self.label2cat.get(l, l)),  # 连续 label -> COCO category_id
                        "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],  # xyxy -> xywh
                        "score": float(s),
                    }
                )

    def synchronize_between_processes(self):
        if distributed.is_enabled() and distributed.get_world_size() > 1:
            world = distributed.get_world_size()
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, self.results)
            merged = []
            for g in gathered:
                merged.extend(g)
            self.results = merged

    def accumulate(self):
        if COCOeval is None:
            raise ImportError("COCOeval 需要 pycocotools，请先安装：pip install pycocotools")
        if len(self.results) == 0:
            self.coco_eval = None
            return
        coco_dt = self.coco_gt.loadRes(self.results)
        self.coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()

    def summarize(self):
        if self.coco_eval is None:
            return {}
        self.coco_eval.summarize()
        s = self.coco_eval.stats
        self.stats = s
        return {"AP": s[0], "AP50": s[1], "AP75": s[2], "APs": s[3], "APm": s[4], "APl": s[5]}


@torch.no_grad()
def evaluate(model, postprocessor, data_loader, base_ds, device, autocast_dtype=None, label2cat=None):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    evaluator = CocoEvaluator(base_ds.coco, label2cat)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        # 用原图尺寸做后处理，输出原图坐标系的框，供 COCOeval 使用
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            outputs = model(samples)
        results = postprocessor(outputs, orig_sizes)
        res = {t["image_id"].item(): r for t, r in zip(targets, results)}
        evaluator.update(res)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    stats = evaluator.summarize()
    return stats
