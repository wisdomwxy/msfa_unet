#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare baseline (projection mask, in_set_A) vs. region growing (set_label A or D)
against PCD scalar ground truth (default field: Constant; positives: 1, 2).

Requires: numpy, pandas, openpyxl, scipy; open3d for non-ASCII PCD.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Project output directory (same folder as this script)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(_PROJECT_ROOT, "out")


def load_pcd_with_scalar(pcd_path, scalar_field="Constant"):
    """Load PCD; return (N,3) points and (N,) scalar. ASCII: parse header; else try Open3D tensor API."""
    with open(pcd_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith("#")]

    header = {}
    data_start = 0
    for i, line in enumerate(lines):
        if line.upper().startswith("DATA"):
            header["DATA"] = line.split()[1]
            data_start = i + 1
            break
        parts = line.split(maxsplit=1)
        if len(parts) >= 2:
            header[parts[0]] = parts[1]

    if header.get("DATA", "").lower() != "ascii":
        try:
            import open3d as o3d

            pcd = o3d.t.geometry.PointCloud(o3d.t.io.read_point_cloud(pcd_path))
            pts = np.asarray(pcd.point.positions.numpy())
            if scalar_field in pcd.point:
                scalar = np.asarray(pcd.point[scalar_field].numpy()).flatten()
            else:
                scalar = None
            return pts, scalar
        except Exception:
            raise RuntimeError(
                f"Expected ASCII DATA or readable Open3D PCD; got DATA={header.get('DATA')}"
            )

    fields = header.get("FIELDS", "").split()
    fields_lower = [f.lower() for f in fields]
    sf_lower = scalar_field.lower()
    if sf_lower not in fields_lower:
        raise ValueError(f"Scalar field '{scalar_field}' not in PCD FIELDS: {fields}")

    col_idx = fields_lower.index(sf_lower)
    n_cols = len(fields)
    rows = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < n_cols:
            continue
        rows.append([float(parts[j]) for j in range(n_cols)])

    arr = np.array(rows, dtype=np.float64)
    points = arr[:, :3]
    scalar = arr[:, col_idx]
    return points, scalar


def load_origin_excel(excel_path):
    """Baseline Excel: columns X, Y, Z, in_set_A."""
    df = pd.read_excel(excel_path)
    pts = df[["X", "Y", "Z"]].values.astype(np.float64)
    in_set_A = df["in_set_A"].values
    return pts, in_set_A


def load_region_grow_excel(excel_path):
    """Region-grow Excel: columns X, Y, Z, set_label."""
    df = pd.read_excel(excel_path)
    pts = df[["X", "Y", "Z"]].values.astype(np.float64)
    set_label = df["set_label"].values
    return pts, set_label


def points_identical(pts_a, pts_b, tol=1e-9):
    """True if same count and coordinates within tol (enables direct index alignment)."""
    if len(pts_a) != len(pts_b):
        return False
    return np.allclose(pts_a.astype(np.float64), pts_b.astype(np.float64), atol=tol, rtol=0)


def match_points_by_coord(pts_src, pts_ref, tol=1e-6):
    """Map each pts_src point to nearest pts_ref index; -1 if none within bound. Identity path skips KD-tree."""
    if len(pts_ref) == 0:
        return np.full(len(pts_src), -1, dtype=np.int64)
    if points_identical(pts_src, pts_ref, tol=tol * 10):
        return np.arange(len(pts_src), dtype=np.int64)
    tree = cKDTree(pts_ref.astype(np.float64))
    dist, idx = tree.query(pts_src.astype(np.float64), k=1, distance_upper_bound=tol * 10)
    valid = dist < tol * 10
    out = np.full(len(pts_src), -1, dtype=np.int64)
    out[valid] = idx[valid]
    return out


def compute_metrics(tp, fp, fn, tn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return prec, rec, f1, iou


def evaluate_single(
    origin_excel_path,
    region_grow_excel_path,
    label_pcd_path,
    gt_positive_values=(1, 2),
    scalar_field="Constant",
):
    """Evaluate one sample: baseline vs region_grow vs GT scalar on matched points."""
    pts_origin, in_set_A = load_origin_excel(origin_excel_path)
    pts_rg, set_label = load_region_grow_excel(region_grow_excel_path)
    pts_gt, scalar = load_pcd_with_scalar(label_pcd_path, scalar_field)

    # Match origin points to labeled PCD (same cloud, order may differ)
    match_orig_to_gt = match_points_by_coord(pts_origin, pts_gt, tol=1e-5)
    valid = match_orig_to_gt >= 0
    gt_label = np.zeros(len(pts_origin), dtype=np.int32)
    scalar_clean = np.nan_to_num(scalar, nan=0.0, posinf=0.0, neginf=0.0)
    gt_label[valid] = np.round(scalar_clean[match_orig_to_gt[valid]]).astype(np.int32)

    gt_pos = np.isin(gt_label, list(gt_positive_values))

    if not points_identical(pts_rg, pts_origin):
        match_rg_to_orig = match_points_by_coord(pts_rg, pts_origin, tol=1e-5)
        pred_rg = np.zeros(len(pts_origin), dtype=bool)
        for i, j in enumerate(match_rg_to_orig):
            if j >= 0 and set_label[i] in ("A", "D"):
                pred_rg[j] = True
    else:
        pred_rg = np.array([sl in ("A", "D") for sl in set_label], dtype=bool)

    pred_baseline = in_set_A == 1

    def _metrics(pred):
        tp = np.sum(pred & gt_pos)
        fp = np.sum(pred & ~gt_pos)
        fn = np.sum(~pred & gt_pos)
        tn = np.sum(~pred & ~gt_pos)
        d1 = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
        d2 = dict(zip(["precision", "recall", "f1", "iou"], compute_metrics(tp, fp, fn, tn)))
        return {**d1, **d2}

    return {
        "baseline": _metrics(pred_baseline),
        "region_grow": _metrics(pred_rg),
        "n_points": len(pts_origin),
        "n_gt_pos": int(np.sum(gt_pos)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate region growing vs baseline using labeled PCD scalars."
    )
    parser.add_argument(
        "--origin-dir",
        default=r"dataset\pcd_offset_compensation\seg_zy",
        help="Directory with baseline .xlsx (batch mode)",
    )
    parser.add_argument(
        "--label-dir",
        default=r"dataset\pcd_offset_compensation\seg_zy_label",
        help="Directory with labeled .pcd (batch mode)",
    )
    parser.add_argument(
        "--region-grow-dir",
        default=r"dataset\pcd_offset_compensation\seg_zy",
        help="Directory with *_region_grow.xlsx; default: same as --origin-dir",
    )
    parser.add_argument("--scalar-field", default="Constant", help="GT scalar field name in PCD")
    parser.add_argument(
        "--gt-positive",
        default="1,2",
        help="Comma-separated positive scalar values (e.g. 1,2)",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output folder for tables (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional CSV path (default: under --out-dir if only batch export name needed)",
    )
    parser.add_argument(
        "--out-excel",
        default=None,
        help="Excel path (default: <out-dir>/eval_region_grow_results.xlsx in batch mode)",
    )
    parser.add_argument(
        "--single",
        nargs=3,
        metavar=("ORIGIN_XLSX", "RG_XLSX", "LABEL_PCD"),
        help="Single-sample mode: three file paths",
    )
    args = parser.parse_args()

    gt_pos_vals = tuple(int(x.strip()) for x in args.gt_positive.split(","))
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.single:
        origin_xlsx, rg_xlsx, label_pcd = args.single
        for p in (origin_xlsx, rg_xlsx, label_pcd):
            if not os.path.exists(p):
                print(f"Error: file not found: {p}")
                sys.exit(1)
        res = evaluate_single(
            origin_xlsx,
            rg_xlsx,
            label_pcd,
            gt_positive_values=gt_pos_vals,
            scalar_field=args.scalar_field,
        )
        print("\n===== Single sample =====")
        print(f"Points: {res['n_points']}, GT positives: {res['n_gt_pos']}")
        print("Baseline (macro):", res["baseline"])
        print("Region grow (macro):", res["region_grow"])
        return

    if args.origin_dir is None or args.label_dir is None:
        print("Error: batch mode requires --origin-dir and --label-dir")
        sys.exit(1)

    origin_dir = args.origin_dir
    label_dir = args.label_dir
    rg_dir = args.region_grow_dir or origin_dir

    if not os.path.isdir(origin_dir) or not os.path.isdir(label_dir):
        print(f"Error: directory not found origin={origin_dir} label={label_dir}")
        sys.exit(1)

    # Baseline xlsx only (exclude companion *_region_grow.xlsx)
    xlsx_files = [
        f for f in os.listdir(origin_dir) if f.endswith(".xlsx") and "_region_grow" not in f
    ]
    results = []

    for xlsx in sorted(xlsx_files):
        base = os.path.splitext(xlsx)[0]
        origin_xlsx = os.path.join(origin_dir, xlsx)
        rg_xlsx = os.path.join(rg_dir, f"{base}_region_grow.xlsx")
        label_pcd = os.path.join(label_dir, f"{base}.pcd")

        if not os.path.exists(rg_xlsx):
            print(f"Skip (no region_grow file): {base}")
            continue
        if not os.path.exists(label_pcd):
            print(f"Skip (no labeled PCD): {label_pcd}")
            continue

        try:
            res = evaluate_single(
                origin_xlsx,
                rg_xlsx,
                label_pcd,
                gt_positive_values=gt_pos_vals,
                scalar_field=args.scalar_field,
            )
            res["base"] = base
            results.append(res)
            print(
                f"{base}: baseline F1={res['baseline']['f1']:.4f} | "
                f"region_grow F1={res['region_grow']['f1']:.4f}"
            )
        except Exception as e:
            print(f"Error {base}: {e}")
            import traceback

            traceback.print_exc()

    if not results:
        print("No valid samples")
        return

    n = len(results)

    base_prec = np.mean([r["baseline"]["precision"] for r in results])
    base_rec = np.mean([r["baseline"]["recall"] for r in results])
    base_f1 = np.mean([r["baseline"]["f1"] for r in results])
    base_iou = np.mean([r["baseline"]["iou"] for r in results])
    rg_prec = np.mean([r["region_grow"]["precision"] for r in results])
    rg_rec = np.mean([r["region_grow"]["recall"] for r in results])
    rg_f1 = np.mean([r["region_grow"]["f1"] for r in results])
    rg_iou = np.mean([r["region_grow"]["iou"] for r in results])

    print("\n===== Summary (mean over samples) =====")
    print(
        f"Baseline    Precision={base_prec:.4f}  Recall={base_rec:.4f}  "
        f"F1={base_f1:.4f}  IoU={base_iou:.4f}"
    )
    print(
        f"RegionGrow  Precision={rg_prec:.4f}  Recall={rg_rec:.4f}  "
        f"F1={rg_f1:.4f}  IoU={rg_iou:.4f}"
    )
    print(f"Delta F1: {rg_f1 - base_f1:+.4f}  Delta IoU: {rg_iou - base_iou:+.4f}")

    default_csv = os.path.join(out_dir, "eval_region_grow_results.csv")
    default_excel = os.path.join(out_dir, "eval_region_grow_results.xlsx")

    if args.out_csv:
        csv_path = os.path.abspath(args.out_csv)
    else:
        csv_path = None

    if args.out_excel:
        excel_path = os.path.abspath(args.out_excel)
    else:
        excel_path = default_excel if not csv_path else None

    if csv_path:
        rows = []
        for r in results:
            row = {
                "sample_id": r["base"],
                "n_points": r["n_points"],
                "n_gt_positive": r["n_gt_pos"],
                "baseline_precision": r["baseline"]["precision"],
                "baseline_recall": r["baseline"]["recall"],
                "baseline_f1": r["baseline"]["f1"],
                "baseline_iou": r["baseline"]["iou"],
                "region_grow_precision": r["region_grow"]["precision"],
                "region_grow_recall": r["region_grow"]["recall"],
                "region_grow_f1": r["region_grow"]["f1"],
                "region_grow_iou": r["region_grow"]["iou"],
            }
            rows.append(row)
        _d = os.path.dirname(csv_path)
        if _d:
            os.makedirs(_d, exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved CSV: {csv_path}")

    if excel_path:
        _write_result_excel(
            excel_path,
            results,
            base_prec,
            base_rec,
            base_f1,
            base_iou,
            rg_prec,
            rg_rec,
            rg_f1,
            rg_iou,
        )
        print(f"\nSaved Excel: {excel_path}")
    elif not csv_path:
        _write_result_excel(
            default_excel,
            results,
            base_prec,
            base_rec,
            base_f1,
            base_iou,
            rg_prec,
            rg_rec,
            rg_f1,
            rg_iou,
        )
        print(f"\nSaved Excel: {default_excel}")


def _write_result_excel(
    path,
    results,
    base_prec,
    base_rec,
    base_f1,
    base_iou,
    rg_prec,
    rg_rec,
    rg_f1,
    rg_iou,
):
    """Write per-sample sheet, summary sheet, and short analysis (all English)."""
    _dir = os.path.dirname(os.path.abspath(path))
    if _dir:
        os.makedirs(_dir, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        rows = []
        for r in results:
            row = {
                "sample_id": r["base"],
                "n_points": r["n_points"],
                "n_gt_positive": r["n_gt_pos"],
                "baseline_precision": round(r["baseline"]["precision"], 4),
                "baseline_recall": round(r["baseline"]["recall"], 4),
                "baseline_f1": round(r["baseline"]["f1"], 4),
                "baseline_iou": round(r["baseline"]["iou"], 4),
                "region_grow_precision": round(r["region_grow"]["precision"], 4),
                "region_grow_recall": round(r["region_grow"]["recall"], 4),
                "region_grow_f1": round(r["region_grow"]["f1"], 4),
                "region_grow_iou": round(r["region_grow"]["iou"], 4),
            }
            rows.append(row)
        pd.DataFrame(rows).to_excel(writer, sheet_name="PerSample", index=False)

        summary_rows = [
            {"metric": "Baseline Precision", "value": round(base_prec, 4)},
            {"metric": "Baseline Recall", "value": round(base_rec, 4)},
            {"metric": "Baseline F1", "value": round(base_f1, 4)},
            {"metric": "Baseline IoU", "value": round(base_iou, 4)},
            {"metric": "Region Grow Precision", "value": round(rg_prec, 4)},
            {"metric": "Region Grow Recall", "value": round(rg_rec, 4)},
            {"metric": "Region Grow F1", "value": round(rg_f1, 4)},
            {"metric": "Region Grow IoU", "value": round(rg_iou, 4)},
            {"metric": "Delta F1 (RG - Baseline)", "value": round(rg_f1 - base_f1, 4)},
            {"metric": "Delta IoU (RG - Baseline)", "value": round(rg_iou - base_iou, 4)},
        ]
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        f1_lift = rg_f1 - base_f1
        iou_lift = rg_iou - base_iou
        analysis_lines = [
            "[Conclusion]",
            f"Number of samples: {len(results)}",
            "",
            "[Overall]",
            f"Region grow vs baseline: Delta F1 = {f1_lift:+.4f}, Delta IoU = {iou_lift:+.4f}",
            "Compare precision/recall to assess filtering and region-growing vs projection mask.",
        ]
        analysis_lines.extend(
            [
                "",
                "[Note]",
                "Tune statistical filtering / region-growing hyperparameters to trade off precision vs recall.",
            ]
        )
        pd.DataFrame({"analysis": analysis_lines}).to_excel(writer, sheet_name="Analysis", index=False)


if __name__ == "__main__":
    main()
