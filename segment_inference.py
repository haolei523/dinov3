# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""对【无标注】图片做语义分割推理。

加载「冻结的 DINOv3 backbone + 你训练好的 linear head(model_final.pth)」，对任意图片
输出分割结果，每张图存三个文件：
    <name>_mask.png     索引图（像素值 = 类别 id，0..num_classes-1，可再用于后处理/评测）
    <name>_color.png    彩色 mask（每个类别一种颜色，便于肉眼查看）
    <name>_overlay.png  原图与彩色 mask 的叠加图

预处理与仓库训练/评测【完全一致】：短边缩放到 img_size、0-255 量纲用 ×255 的 ImageNet
mean/std 归一化、slide 滑窗(crop 512 / stride 341)、最后插值回原图尺寸。所以结果与
test_segmentation 的口径一致（只是这里不需要 GT、不算 mIoU）。

需要 GPU（load_model_and_context 内部会 .cuda()）。

用法（离线，单卡）：
    PYTHONPATH=. python3 segment_inference.py \
        --input /path/to/images_dir_or_one_image \
        --head-ckpt /path/to/model_final.pth \
        --output-dir /path/to/seg_out \
        --backbone-name dinov3_vitl16 \
        --backbone-weights /path/to/backbone.pth \
        --num-classes 2 \
        --backbone-out-layers LAST

    # backbone 也可用离线架构 yaml（非 plus 变体）：
    #   --config-file dinov3/configs/train/dinov3_vitl16_lvd1689m_distilled.yaml \
    #   --backbone-weights /path/to/backbone.pth
    # 联网环境可只给 --backbone-name（自动下载权重），省略 --backbone-weights。

关键：--num-classes / --backbone-out-layers / backbone 变体 必须与【训练时一致】，
否则 head 权重形状对不上（会直接报错）或 BN 统计量被静默跳过（结果变差）。
注：若训练时 reduce_zero_label=True，这里 argmax 得到的 id 需 +1 才是原始标签；
    你的 custom 配置是 reduce_zero_label=False，则 id 即原始标签，无需偏移。
"""
import argparse
import colorsys
import os
from functools import partial
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as Fv

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, make_normalize_transform
from dinov3.eval.segmentation.inference import make_inference
from dinov3.eval.segmentation.models import BackboneLayersSet, build_segmentation_decoder
from dinov3.eval.setup import ModelConfig, load_model_and_context
from dinov3.run.init import job_context

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args():
    p = argparse.ArgumentParser(description="DINOv3 分割推理（无标注图片，输出 mask/可视化）")
    p.add_argument("--input", required=True, help="图片路径，或包含图片的目录")
    p.add_argument("--head-ckpt", required=True, help="训练产出的 model_final.pth")
    p.add_argument("--output-dir", required=True, help="mask / 可视化输出目录")
    # ---- backbone（必须与训练一致）----
    p.add_argument("--backbone-name", default=None, help="hub 名称，如 dinov3_vitl16 / dinov3_vits16 ...")
    p.add_argument("--backbone-weights", default=None, help="本地 backbone .pth（离线必填；联网可省略）")
    p.add_argument("--config-file", default=None, help="离线架构 yaml（与 --backbone-name 二选一）")
    # ---- head（必须与训练一致）----
    p.add_argument("--num-classes", type=int, default=2, help="与训练时的 decoder_head.num_classes 一致")
    p.add_argument(
        "--backbone-out-layers",
        default="LAST",
        choices=[e.name for e in BackboneLayersSet],
        help="与训练时的 decoder_head.backbone_out_layers 一致",
    )
    # ---- 推理参数（默认与训练 eval 段一致）----
    p.add_argument("--mode", default="slide", choices=["slide", "whole"], help="slide=滑窗(推荐)，whole=整图缩放")
    p.add_argument("--img-size", type=int, default=512, help="短边缩放尺寸（对应 transforms.eval.img_size）")
    p.add_argument("--crop-size", type=int, default=512, help="滑窗窗口大小（对应 eval.crop_size）")
    p.add_argument("--stride", type=int, default=341, help="滑窗步长（对应 eval.stride）")
    p.add_argument("--model-dtype", default="float32", choices=["float32", "bfloat16"])
    return p.parse_args()


def collect_images(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        return [
            os.path.join(input_path, n)
            for n in sorted(os.listdir(input_path))
            if n.lower().endswith(IMG_EXTS)
        ]
    if os.path.isfile(input_path):
        return [input_path]
    raise FileNotFoundError(f"输入路径不存在: {input_path}")


def build_backbone(args) -> torch.nn.Module:
    """按训练时的方式加载 backbone（离线：本地 hub 名 + 本地权重；或 config_file + 本地权重）。"""
    if args.config_file is not None:
        model_config = ModelConfig(config_file=args.config_file, pretrained_weights=args.backbone_weights)
    elif args.backbone_name is not None:
        model_config = ModelConfig(dino_hub=args.backbone_name, pretrained_weights=args.backbone_weights)
    else:
        raise ValueError("必须提供 --backbone-name 或 --config-file 之一")
    backbone, _ = load_model_and_context(model_config, output_dir=args.output_dir)
    return backbone


def preprocess(pil_img: Image.Image, img_size: int, normalize) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """复刻 make_segmentation_eval_transforms(inference_mode='slide') 的图像预处理。

    返回：([1,C,h,w] 归一化张量, 原图 (H, W))。h/w 是短边缩放到 img_size 后的尺寸。
    """
    orig_w, orig_h = pil_img.size  # PIL.size = (W, H)
    # 短边缩放到 img_size，保持长宽比（与 FixedSideResize._resize 一致）
    if orig_h > orig_w:
        new_w = img_size
        new_h = int(img_size * orig_h / orig_w + 0.5)
    else:
        new_h = img_size
        new_w = int(img_size * orig_w / orig_h + 0.5)
    img = T.Resize(size=(new_h, new_w), interpolation=T.InterpolationMode.BILINEAR)(pil_img)
    x = Fv.pil_to_tensor(img).float()  # [C,H,W]，0-255
    x = normalize(x)  # (x - mean*255) / (std*255)
    return x.unsqueeze(0), (orig_h, orig_w)


def make_palette(num_classes: int) -> np.ndarray:
    """为每个类别生成确定性颜色；类别 0 视为背景（黑色）。"""
    palette = np.zeros((max(num_classes, 1), 3), dtype=np.uint8)
    for i in range(1, num_classes):
        hue = (i * 0.61803398875) % 1.0  # 黄金角，颜色分散
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        palette[i] = (int(r * 255), int(g * 255), int(b * 255))
    return palette


def load_head_state_dict(path: str):
    """读取 model_final.pth 里的 head 权重（兼容 {'model': ...} 与裸 state_dict）。"""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # 文件里若含优化器状态等非纯张量对象，weights_only 可能失败，这里回退一次
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    return ckpt


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("本脚本需要 GPU（backbone 加载内部会 .cuda()）。")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")
    autocast_dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32

    # 1) 冻结 backbone
    backbone = build_backbone(args)

    # 2) 组装 backbone + linear head（与训练同一函数，结构必然一致）
    seg_model = build_segmentation_decoder(
        backbone,
        BackboneLayersSet[args.backbone_out_layers],
        "segformer",
        num_classes=args.num_classes,
        autocast_dtype=autocast_dtype,
        dropout=0.0,
    ).to(device)

    # 3) 载入训练好的 head（strict=False：只覆盖 head，backbone 保持预训练权重）
    head_sd = load_head_state_dict(args.head_ckpt)
    missing, unexpected = seg_model.load_state_dict(head_sd, strict=False)
    head_missing = [k for k in missing if k.startswith("segmentation_model.1")]
    if head_missing or unexpected:
        raise RuntimeError(
            "head 权重未正确加载！\n"
            f"  缺失的 head 键: {head_missing}\n"
            f"  多余的键: {list(unexpected)}\n"
            "请确认 --num-classes / --backbone-out-layers / backbone 变体与训练时完全一致。"
        )
    seg_model.eval()

    # 4) 预处理常量（与训练一致：×255 的 ImageNet mean/std）
    normalize = make_normalize_transform(
        mean=[m * 255 for m in IMAGENET_DEFAULT_MEAN],
        std=[s * 255 for s in IMAGENET_DEFAULT_STD],
    )
    palette = make_palette(args.num_classes)
    softmax = partial(F.softmax, dim=1)

    images = collect_images(args.input)
    print(f"待推理图片数: {len(images)}；输出目录: {args.output_dir}")

    for path in images:
        pil_img = Image.open(path).convert("RGB")
        x, (orig_h, orig_w) = preprocess(pil_img, args.img_size, normalize)
        x = x.to(device)
        with torch.inference_mode():
            pred = make_inference(
                x,
                seg_model,
                inference_mode=args.mode,
                decoder_head_type="linear",
                rescale_to=(orig_h, orig_w),
                n_output_channels=args.num_classes,
                crop_size=(args.crop_size, args.crop_size),
                stride=(args.stride, args.stride),
                output_activation=softmax,
            )
        # pred: [1, num_classes, orig_h, orig_w] -> argmax 得类别 id 图
        mask = pred.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()  # [orig_h, orig_w]

        stem = os.path.splitext(os.path.basename(path))[0]
        # a) 索引 mask
        Image.fromarray(mask).save(os.path.join(args.output_dir, f"{stem}_mask.png"))
        # b) 彩色 mask
        color = palette[mask]  # [orig_h, orig_w, 3]
        Image.fromarray(color).save(os.path.join(args.output_dir, f"{stem}_color.png"))
        # c) 叠加可视化（原图尺寸与 mask 一致）
        orig = np.array(pil_img)
        overlay = (0.5 * orig + 0.5 * color).astype(np.uint8)
        Image.fromarray(overlay).save(os.path.join(args.output_dir, f"{stem}_overlay.png"))
        print(f"  {os.path.basename(path)} -> {stem}_mask/color/overlay.png")

    print("完成。")


def main():
    args = parse_args()
    # 与仓库 run.py 一致：先初始化分布式（单进程 world_size=1）与日志。
    # 走 --config-file 时，setup_config 内部会断言 distributed.is_enabled()，故必须先 setup_job。
    with job_context(output_dir=args.output_dir):
        run(args)


if __name__ == "__main__":
    main()
