# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# ------------------------------------------------------------------------
# DINO-DETR head training on a frozen DINOv3 backbone.
# ------------------------------------------------------------------------
"""Train the DINO-DETR detection head while keeping the DINOv3 backbone frozen.

The backbone is loaded by ``load_model_and_context`` (see eval/setup.py) and wrapped
by ``build_model`` -> ``build_backbone`` which hardcodes ``train_backbone=False``, so
every ViT parameter gets ``requires_grad_(False)``. Only the DETR head (input_proj,
transformer, class/box embeddings, query embeddings) is optimized.
"""
import logging
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler

import dinov3.distributed as distributed
from dinov3.eval.detection.criterion import build_criterion
from dinov3.eval.detection.datasets import build_coco
from dinov3.eval.detection.engine import evaluate, train_one_epoch
from dinov3.eval.detection.models.detr import PostProcess, build_model
from dinov3.eval.detection.util.misc import collate_fn

logger = logging.getLogger("dinov3")


def _build_detector(backbone, head_config):
    """Assemble the DINO-DETR head on top of a frozen DINOv3 backbone.

    Mirrors ``dinov3.hub.detectors._make_dinov3_detector``: the proposal strides and
    the intermediate layers to concatenate are derived from the backbone geometry.
    """
    head_config.proposal_in_stride = backbone.patch_size
    head_config.proposal_tgt_strides = [int(m * backbone.patch_size) for m in (0.5, 1, 2, 4)]
    if head_config.layers_to_use is None:
        # e.g. [5, 11, 17, 23] for a 24-block ViT-L, similar to the depth evaluation
        head_config.layers_to_use = [m * backbone.n_blocks // 4 - 1 for m in range(1, 5)]
    return build_model(backbone, head_config)


def _save_head(model, optimizer, epoch, ap, config, is_best):
    """Save ONLY the trainable head weights (drop the frozen backbone)."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    head_state = {k: v for k, v in model.state_dict().items() if k in trainable}
    payload = {"model": head_state, "optimizer": optimizer.state_dict(), "epoch": epoch, "AP": ap}
    torch.save(payload, os.path.join(config.output_dir, "detection_head_last.pth"))
    if is_best:
        torch.save(payload, os.path.join(config.output_dir, "detection_head_best.pth"))
    logger.info(f"Saved head checkpoint ({len(head_state)} tensors, AP={ap:.4f})")


def train_detection(backbone, config):
    head = config.head
    autocast_dtype = config.model_dtype.autocast_dtype
    local_device = torch.cuda.current_device()

    # 1- datasets (COCO format). num_classes is derived from the annotations so the
    #    class head matches the data (labels are remapped to contiguous 0..C-1).
    train_ds = build_coco("train", config)
    val_ds = build_coco("val", config)
    num_classes = train_ds.num_classes
    if val_ds.num_classes != num_classes:
        logger.warning(
            f"train/val category count mismatch ({num_classes} vs {val_ds.num_classes}); "
            "using the train mapping. Make sure both splits share the same categories."
        )
    logger.info(f"num_classes={num_classes} (contiguous labels from the COCO annotations)")
    head.num_classes = num_classes
    label2cat = {i: cat for cat, i in train_ds.cat2label.items()}  # for COCOeval category_id

    # 2- model: frozen backbone + trainable DETR head, wrapped in DDP.
    detector = _build_detector(backbone, head)
    n_train = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in detector.parameters())
    logger.info(f"Trainable parameters: {n_train:,} / {n_total:,} (backbone frozen)")
    detector = detector.to(local_device)
    if config.load_from:
        logger.info(f"Loading head weights from {config.load_from}")
        ckpt = torch.load(config.load_from, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = detector.load_state_dict(state, strict=False)
        logger.info(f"load_state_dict -> missing={len(missing)}, unexpected={len(unexpected)}")
    detector = torch.nn.parallel.DistributedDataParallel(
        detector, device_ids=[local_device], find_unused_parameters=True
    )

    criterion = build_criterion(head, num_classes).to(local_device)
    postprocessor = PostProcess(config.eval.topk, head.reparam)

    # 3- dataloaders (DETR collate_fn -> NestedTensor + list[target]).
    is_dist = distributed.is_enabled() and distributed.get_world_size() > 1
    train_sampler = (
        DistributedSampler(train_ds, shuffle=True, rank=distributed.get_rank(), num_replicas=distributed.get_world_size())
        if is_dist
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_sampler = (
        DistributedSampler(val_ds, shuffle=False, rank=distributed.get_rank(), num_replicas=distributed.get_world_size())
        if is_dist
        else None
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 4- optimizer + scheduler over the trainable head params only.
    params = [p for p in detector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(config.scheduler.lr_drop_epochs), gamma=config.scheduler.gamma
    )

    # 5- epoch loop.
    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)
    total_epochs = config.scheduler.total_epochs
    best_ap = -1.0
    for epoch in range(total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            detector,
            criterion,
            train_loader,
            optimizer,
            local_device,
            epoch,
            max_norm=config.optimizer.gradient_clip,
            autocast_dtype=autocast_dtype,
        )
        scheduler.step()
        logger.info(f"Epoch {epoch} train: {train_stats}")

        is_last = epoch == total_epochs - 1
        if (epoch + 1) % config.eval.eval_interval == 0 or is_last:
            coco_stats = evaluate(
                detector.module,
                postprocessor,
                val_loader,
                val_ds,
                local_device,
                autocast_dtype=autocast_dtype,
                label2cat=label2cat,
            )
            logger.info(f"Epoch {epoch} eval: {coco_stats}")
            ap = float(coco_stats.get("AP", -1.0)) if coco_stats else -1.0
            if distributed.is_main_process() and config.output_dir:
                is_best = ap > best_ap
                _save_head(detector.module, optimizer, epoch, ap, config, is_best)
                if is_best:
                    best_ap = ap
        elif distributed.is_main_process() and config.output_dir:
            _save_head(detector.module, optimizer, epoch, -1.0, config, is_best=False)

    logger.info(f"Training done. Best AP: {best_ap:.4f}")
    return {"AP": best_ap}
