# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import os
import sys
from typing import Any

from omegaconf import OmegaConf

import dinov3.distributed as distributed
from dinov3.eval.detection.config import DetectionTrainConfig
from dinov3.eval.detection.train import train_detection
from dinov3.eval.helpers import args_dict_to_dataclass, cli_parser, write_results
from dinov3.eval.setup import load_model_and_context
from dinov3.run.init import job_context

logger = logging.getLogger("dinov3")

RESULTS_FILENAME = "results-detection.csv"


def benchmark_launcher(eval_args: dict[str, object]) -> dict[str, Any]:
    """Distributed + logging must be initialized before calling this (see main)."""
    if "config" in eval_args:  # a config yaml was provided (the usual training path)
        base_config_path = eval_args.pop("config")
        output_dir = eval_args["output_dir"]
        base_config = OmegaConf.load(base_config_path)
        structured_config = OmegaConf.structured(DetectionTrainConfig)
        dataclass_config: DetectionTrainConfig = OmegaConf.to_object(
            OmegaConf.merge(structured_config, base_config, OmegaConf.create(eval_args))
        )
    else:  # defaults + a few CLI overrides
        dataclass_config, output_dir = args_dict_to_dataclass(
            eval_args=eval_args, config_dataclass=DetectionTrainConfig
        )

    assert dataclass_config.model is not None, (
        "A DINOv3 backbone is required. Set model.dino_hub=dinov3_vitl16 (downloads weights) "
        "or model.config_file + model.pretrained_weights (local weights) in the config/CLI."
    )
    backbone, _ = load_model_and_context(dataclass_config.model, output_dir=output_dir)
    dataclass_config.output_dir = output_dir

    logger.info(f"Detection Config:\n{OmegaConf.to_yaml(dataclass_config)}")
    if distributed.is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        OmegaConf.save(config=dataclass_config, f=os.path.join(output_dir, "detection_config.yaml"))

    results_dict = train_detection(backbone=backbone, config=dataclass_config)
    write_results(results_dict, output_dir, RESULTS_FILENAME)
    return results_dict


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    eval_args = cli_parser(argv)
    with job_context(output_dir=eval_args["output_dir"]):
        benchmark_launcher(eval_args=eval_args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
