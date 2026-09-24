# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# ------------------------------------------------------------------------
# DINO / Deformable DETR
# Hungarian matcher + set criterion, adapted to this repo's PlainDETR outputs.
# ------------------------------------------------------------------------
"""Loss for the DINO-DETR head.

The model (PlainDETR.forward) returns a dict with:
  pred_logits / pred_boxes                      -> one2one, final decoder layer
  aux_outputs                                   -> one2one, each earlier decoder layer
  enc_outputs                                   -> two-stage encoder proposal
  pred_logits_one2many / pred_boxes_one2many    -> one2many branch (DINO hybrid matching)
  aux_outputs_one2many                          -> one2many, each earlier layer
Classification uses sigmoid focal loss (no explicit background class), boxes are
normalized cxcywh and supervised with L1 + generalized IoU.
"""
import copy

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

import dinov3.distributed as distributed

from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2.0):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground truth (cost = class + L1 + GIoU)."""

    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0, focal_alpha: float = 0.25):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [bs*nq, C]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [bs*nq, 4]

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        alpha, gamma = self.focal_alpha, 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            if sizes[i] == 0:
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
            else:
                r, col = linear_sum_assignment(c[i])
                indices.append((torch.tensor(r, dtype=torch.int64), torch.tensor(col, dtype=torch.int64)))
        return indices


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: dict,
        focal_alpha: float = 0.25,
        k_one2many: int = 0,
        lambda_one2many: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs["pred_logits"]  # [bs, nq, C]
        device = src_logits.device
        idx = self._get_src_permutation_idx(indices)
        idx = (idx[0].to(device), idx[1].to(device))
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(device)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=device)
        if target_classes_o.numel() > 0:
            target_classes[idx] = target_classes_o

        onehot = torch.zeros(
            src_logits.shape[:2] + (src_logits.shape[2] + 1,), dtype=src_logits.dtype, device=device
        )
        onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_01 = onehot[..., : src_logits.shape[2]]  # drop the "background" column
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_01, num_boxes, alpha=self.focal_alpha, gamma=2.0)
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if src_boxes.numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0  # 保持计算图
            return {"loss_bbox": zero, "loss_giou": zero}
        src_boxes = src_boxes.to(target_boxes.device)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = loss_giou.sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def _loss_pair(self, outputs, targets, num_boxes):
        indices = self.matcher(outputs, targets)
        losses = self.loss_labels(outputs, targets, indices, num_boxes)
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses

    @staticmethod
    def _multiply_targets(targets, k):
        multi = []
        for t in targets:
            nt = {}
            for key, val in t.items():
                if isinstance(val, torch.Tensor) and val.dim() > 0 and key in ("labels", "boxes", "area", "iscrowd"):
                    nt[key] = val.repeat(*([k] + [1] * (val.dim() - 1)))
                else:
                    nt[key] = val
            multi.append(nt)
        return multi

    def forward(self, outputs, targets):
        device = outputs["pred_logits"].device
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if distributed.is_enabled():
            torch.distributed.all_reduce(num_boxes)
            num_boxes = num_boxes / distributed.get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        losses = {}

        # ---------- one2one ----------
        main_out = {"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]}
        losses.update(self._loss_pair(main_out, targets, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                d = self._loss_pair(aux, targets, num_boxes)
                losses.update({f"{k}_{i}": v for k, v in d.items()})

        if "enc_outputs" in outputs:
            bin_targets = copy.deepcopy(targets)
            d = self._loss_pair(outputs["enc_outputs"], bin_targets, num_boxes)
            losses.update({f"{k}_enc": v for k, v in d.items()})

        # ---------- one2many (DINO hybrid matching) ----------
        has_many = (
            self.k_one2many > 0
            and outputs.get("pred_logits_one2many") is not None
            and outputs["pred_logits_one2many"].shape[1] > 0
        )
        if has_many:
            multi_targets = self._multiply_targets(targets, self.k_one2many)
            num_boxes_many = num_boxes * self.k_one2many
            many_out = {
                "pred_logits": outputs["pred_logits_one2many"],
                "pred_boxes": outputs["pred_boxes_one2many"],
            }
            d = self._loss_pair(many_out, multi_targets, num_boxes_many)
            losses.update({f"{k}_one2many": v * self.lambda_one2many for k, v in d.items()})

            if "aux_outputs_one2many" in outputs:
                for i, aux in enumerate(outputs["aux_outputs_one2many"]):
                    d = self._loss_pair(aux, multi_targets, num_boxes_many)
                    losses.update({f"{k}_one2many_{i}": v * self.lambda_one2many for k, v in d.items()})

        return losses


def build_criterion(config, num_classes: int) -> SetCriterion:
    matcher = HungarianMatcher(
        cost_class=config.matcher.cost_class,
        cost_bbox=config.matcher.cost_bbox,
        cost_giou=config.matcher.cost_giou,
        focal_alpha=config.loss.focal_alpha,
    )

    base = {
        "loss_ce": config.loss.cls_weight,
        "loss_bbox": config.loss.bbox_weight,
        "loss_giou": config.loss.giou_weight,
    }
    weight_dict = dict(base)
    if config.aux_loss:
        for i in range(config.dec_layers + 1):  # 覆盖足够多的 decoder 层
            weight_dict.update({f"{k}_{i}": v for k, v in base.items()})
    if config.two_stage:
        weight_dict.update({f"{k}_enc": v for k, v in base.items()})
    if config.k_one2many > 0:
        weight_dict.update({f"{k}_one2many": v * config.lambda_one2many for k, v in base.items()})
        if config.aux_loss:
            for i in range(config.dec_layers + 1):
                weight_dict.update({f"{k}_one2many_{i}": v * config.lambda_one2many for k, v in base.items()})

    return SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=config.loss.focal_alpha,
        k_one2many=config.k_one2many,
        lambda_one2many=config.lambda_one2many,
    )
