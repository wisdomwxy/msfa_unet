#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation (VOC layout): writes eval_logs/detailed_metrics.txt and prints metrics + FPS to stdout."""
import os

import torch
from nets.unetFastV2 import Unet
from utils.callbacks import EvalCallback


class EvalConfig:
    VOCdevkit_path = r"dataset"
    model_path = r"weights\best.pth"
    num_classes = 6
    backbone = "mobilenetv3_small"
    input_shape = [512, 512]
    cuda = True

    miou_out_path = "miou_out"
    eval_flag = True
    dataset_path = VOCdevkit_path
    name_classes = ["_background_", "1", "2", "3", "4", "5"]


def main():
    log_dir = "eval_logs"
    os.makedirs(log_dir, exist_ok=True)

    test_txt = os.path.join(
        EvalConfig.VOCdevkit_path, "VOC2007/test.txt"
    )
    with open(test_txt, "r") as f:
        val_lines = [line.strip() for line in f.readlines()]

    device = torch.device(
        "cuda" if torch.cuda.is_available() and EvalConfig.cuda else "cpu"
    )
    model = Unet(num_classes=EvalConfig.num_classes, backbone=EvalConfig.backbone)

    print("Loading weights...")
    model_dict = model.state_dict()
    pretrained_dict = torch.load(EvalConfig.model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.to(device)
    model.eval()
    print("Weights loaded.")

    eval_callback = EvalCallback(
        model,
        EvalConfig.input_shape,
        EvalConfig.num_classes,
        val_lines,
        EvalConfig.dataset_path,
        log_dir,
        EvalConfig.cuda,
        EvalConfig.miou_out_path,
        True,
        1,
        EvalConfig.name_classes,
        metrics_txt_only=True,
    )

    print("Starting evaluation...")
    eval_callback.on_epoch_end(0, model)
    print("Evaluation finished.")


if __name__ == "__main__":
    main()
