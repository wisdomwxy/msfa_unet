#!/usr/bin/env python3
"""Copy JPEGImages listed in val.txt to dataset/img (same name resolution as gen_seg_outputs)."""
import argparse
import os
import shutil
import sys


def resolve_paths(names, jpeg_dir):
    items = []
    for name in names:
        found = False
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = (
                os.path.join(jpeg_dir, name + ext)
                if not name.lower().endswith((".jpg", ".jpeg", ".png"))
                else os.path.join(jpeg_dir, name)
            )
            if os.path.exists(candidate):
                items.append(candidate)
                found = True
                break
        if not found:
            print(f"Skip (missing): {name}", file=sys.stderr)
    return items


def main():
    p = argparse.ArgumentParser(description="Copy val.txt images to dataset/img")
    p.add_argument(
        "--val-txt",
        default=r"C:\workspace\unet\CMAT1200_mix\VOC2007\ImageSets\Segmentation\val.txt",
    )
    p.add_argument(
        "--jpeg-dir",
        default=r"C:\workspace\unet\CMAT1200_mix\VOC2007\SegmentationClass",
    )
    p.add_argument(
        "--dst",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "dataset", "img"
        ),
    )
    args = p.parse_args()

    if not os.path.isfile(args.val_txt):
        print(f"Error: not found {args.val_txt}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.jpeg_dir):
        print(f"Error: not found {args.jpeg_dir}", file=sys.stderr)
        sys.exit(1)

    with open(args.val_txt, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    os.makedirs(args.dst, exist_ok=True)
    paths = resolve_paths(names, args.jpeg_dir)
    for src in paths:
        shutil.copy2(src, os.path.join(args.dst, os.path.basename(src)))

    print(f"Copied {len(paths)} file(s) -> {args.dst}")


if __name__ == "__main__":
    main()
