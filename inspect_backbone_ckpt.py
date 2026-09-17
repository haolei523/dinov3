# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""诊断脚本：确认你的 backbone .pth 能否被本仓库直接加载（用完可删）。

背景：离线加载走 init_model_from_checkpoint_for_evals(..., strict=False)。
键名对不上时它【不会报错】，只会把 backbone 加载成随机权重，导致训练无效。
先跑这个脚本确认格式，再决定是否需要转换。

用法：
    PYTHONPATH=. python3 inspect_backbone_ckpt.py /path/to/your_backbone.pth
"""
import sys

import torch

from dinov3.hub.backbones import dinov3_vits16


def unwrap(sd):
    """模拟仓库加载逻辑：取 teacher/model/state_dict 子字典，并 strip 掉前缀。"""
    if isinstance(sd, dict):
        for k in ("teacher", "model", "state_dict"):
            if k in sd and isinstance(sd[k], dict):
                print(f"[info] 检测到嵌套 key '{k}'，取其内容")
                sd = sd[k]
                break
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    return sd


def main(path):
    ckpt = torch.load(path, map_location="cpu")
    sd = unwrap(ckpt)
    your_keys = {k for k in sd.keys() if hasattr(sd[k], "shape")}

    # 本仓库 ViT-L/16 期望的键名（pretrained=False 不联网，只搭结构）
    ref = dinov3_vits16(pretrained=False)
    ref_keys = set(ref.state_dict().keys())

    matched = ref_keys & your_keys
    missing = ref_keys - your_keys
    unexpected = your_keys - ref_keys
    ratio = len(matched) / max(len(ref_keys), 1)

    print("=" * 64)
    print(f"你的 ckpt 张量数 : {len(your_keys)}")
    print(f"仓库期望张量数   : {len(ref_keys)}")
    print(f"匹配             : {len(matched)}  ({ratio:.0%})")
    print(f"缺失(仓库要,你没有): {len(missing)}")
    print(f"多余(你有,仓库不要): {len(unexpected)}")
    print("=" * 64)
    print("你的键名样本(前 10):")
    for k in sorted(your_keys)[:10]:
        print("   ", k, tuple(sd[k].shape))

    tf_markers = ("embeddings.", "encoder.layer.", "layernorm.", "attention.attention.query", "intermediate.dense")
    looks_transformers = any(any(m in k for m in tf_markers) for k in your_keys)

    print("=" * 64)
    if ratio > 0.9:
        print(f"结论: [OK] 原生 DINOv3 格式，匹配率 {ratio:.0%}，可直接用作 pretrained_weights。")
        if missing:
            print(f"      (少量缺失多为运行时 buffer，如 bias_mask，属正常: {sorted(missing)[:5]})")
    elif looks_transformers:
        print(f"结论: [需转换] HuggingFace transformers 格式，匹配率仅 {ratio:.0%}，键名不兼容。")
        print("      典型多余键:", sorted(unexpected)[:8])
        print("      -> 需要用转换脚本把 q/k/v 合并成 qkv 并重命名，再训练。")
    else:
        print(f"结论: [存疑] 匹配率 {ratio:.0%}，既不像原生也不像标准 transformers。")
        print("      缺失键样本:", sorted(missing)[:8])
        print("      多余键样本:", sorted(unexpected)[:8])
        print("      -> 请把以上键名贴出来人工确认。")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: PYTHONPATH=. python3 inspect_backbone_ckpt.py /path/to/your_backbone.pth")
        sys.exit(1)
    main(sys.argv[1])
