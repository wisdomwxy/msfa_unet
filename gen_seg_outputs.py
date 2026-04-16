#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNet segmentation inference: blended overlay, color mask, and grayscale label maps.
Run from project root: python gen_seg_outputs.py ...
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from infer import load_model, predict_logits_to_class_map
import config as cfg

# Subfolder names under --out-dir for each output type
OUT_SUB = {"blended": "blended", "color": "color", "origin": "origin"}


def visualize_result(pr, original_img, color_map, alpha=0.5):
    vis_img = np.zeros((pr.shape[0], pr.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        vis_img[pr == class_id] = color
    blended = cv2.addWeighted(original_img, alpha, vis_img, 1 - alpha, 0)
    return blended


def calculate_metrics(pr, gt, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(pr.shape[0]):
        for j in range(pr.shape[1]):
            confusion_matrix[gt[i, j], pr[i, j]] += 1
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU = tp / (pos + res - tp + 1e-10)
    mean_IoU = np.nanmean(IoU)
    accuracy = np.sum(tp) / np.sum(confusion_matrix)
    return IoU, mean_IoU, accuracy, confusion_matrix


def get_image_list(read_mode, folder, val_txt, jpeg_dir):
    items = []
    if read_mode == "folder":
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                items.append((os.path.join(folder, fname), os.path.splitext(fname)[0]))
    elif read_mode == "val_txt":
        if not os.path.exists(val_txt):
            raise FileNotFoundError(val_txt)
        with open(val_txt, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        for name in names:
            found = False
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = (
                    os.path.join(jpeg_dir, name + ext)
                    if not name.lower().endswith((".jpg", ".jpeg", ".png"))
                    else os.path.join(jpeg_dir, name)
                )
                if os.path.exists(candidate):
                    items.append(
                        (candidate, os.path.splitext(os.path.basename(candidate))[0])
                    )
                    found = True
                    break
            if not found:
                print(f"Skip (missing image): {name}", file=sys.stderr)
    else:
        raise ValueError(read_mode)
    return items


def parse_args():
    p = argparse.ArgumentParser(
        description="UNet inference: write blended / color / origin under separate subfolders."
    )
    p.add_argument(
        "--read-mode",
        choices=["folder", "val_txt", "single"],
        default="val_txt",
        help="folder | val_txt | single (requires --image)",
    )
    p.add_argument("--image", default=None, help="Image path for read-mode single")
    p.add_argument("--folder", default=None, help="Image folder for read-mode folder")
    p.add_argument(
        "--val-txt",
        default=r"C:\workspace\unet\CMAT1200_mix\VOC2007\ImageSets\Segmentation\val.txt",
        dest="val_txt",
        help="val list (one basename per line) for val_txt mode",
    )
    p.add_argument(
        "--jpeg-dir",
        default=r"C:\workspace\unet\CMAT1200_mix\VOC2007\JPEGImages",
        dest="jpeg_dir",
        help="JPEGImages directory for val_txt mode",
    )
    p.add_argument(
        "--out-dir",
        default=cfg.DEFAULT_OUT_DIR,
        dest="out_dir",
        help="Root output directory (creates blended/, color/, origin/ as needed)",
    )
    p.add_argument(
        "--outputs",
        default="blended,color,origin",
        help="Comma-separated: blended, color, origin",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Weights path (default: config.DEFAULT_MODEL_PATH)",
    )
    p.add_argument(
        "--eval-gt",
        action="store_true",
        help="If {basename}_gt.png exists next to input, print mIoU and accuracy",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    _valid = {"blended", "color", "origin"}
    output_types = [
        x.strip() for x in args.outputs.split(",") if x.strip() and x.strip() in _valid
    ]
    if not output_types:
        print("Error: --outputs must include at least one of blended,color,origin", file=sys.stderr)
        sys.exit(1)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.cuda else "cpu"
    )

    model_path = args.model if args.model else cfg.DEFAULT_MODEL_PATH
    model, device = load_model(model_path=model_path, device=device, verbose=False)

    if args.read_mode == "single":
        if not args.image or not os.path.isfile(args.image):
            print("Error: single mode needs a valid --image", file=sys.stderr)
            sys.exit(1)
        image_items = [
            (args.image, os.path.splitext(os.path.basename(args.image))[0])
        ]
    elif args.read_mode == "folder":
        if not args.folder:
            print("Error: folder mode needs --folder", file=sys.stderr)
            sys.exit(1)
        image_items = get_image_list("folder", args.folder, None, None)
    else:
        if not args.val_txt or not args.jpeg_dir:
            print("Error: val_txt mode needs --val-txt and --jpeg-dir", file=sys.stderr)
            sys.exit(1)
        image_items = get_image_list("val_txt", None, args.val_txt, args.jpeg_dir)

    if not image_items:
        print("No images to process.", file=sys.stderr)
        return

    sub_paths = {k: os.path.join(out_root, OUT_SUB[k]) for k in output_types if k in OUT_SUB}
    for d in sub_paths.values():
        os.makedirs(d, exist_ok=True)

    color_map = cfg.COLOR_MAP
    n_ok = 0
    for image_path, base_name in image_items:
        image = Image.open(image_path)
        pr = predict_logits_to_class_map(image, model, device, cfg.input_shape)
        original_img = np.array(image.convert("RGB"))

        if "blended" in output_types:
            blended = visualize_result(pr, original_img, color_map)
            path = os.path.join(sub_paths["blended"], f"{base_name}.png")
            cv2.imwrite(path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        if "color" in output_types:
            color_mask = np.zeros((pr.shape[0], pr.shape[1], 3), dtype=np.uint8)
            for class_id, c in color_map.items():
                color_mask[pr == class_id] = c
            path = os.path.join(sub_paths["color"], f"{base_name}.png")
            Image.fromarray(color_mask).save(path)
        if "origin" in output_types:
            path = os.path.join(sub_paths["origin"], f"{base_name}.png")
            Image.fromarray(pr.astype(np.uint8)).save(path)

        n_ok += 1
        if args.eval_gt:
            gt_path = os.path.join(os.path.dirname(image_path), f"{base_name}_gt.png")
            if os.path.exists(gt_path):
                gt = np.array(Image.open(gt_path))
                _, mean_iou, acc, _ = calculate_metrics(pr, gt, cfg.num_classes)
                print(f"{base_name}: mIoU={mean_iou:.4f} acc={acc:.4f}")

    print(f"Done: {n_ok} image(s) -> {out_root}")


if __name__ == "__main__":
    main()
