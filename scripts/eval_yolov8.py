#!/usr/bin/env python3
"""Evaluate a trained YOLOv8 checkpoint on the test split (patient-level metrics).

Use this when training was done elsewhere (e.g. a Kaggle GPU commit) and you
only need to score an existing `best.pt` locally. It reuses the exact eval logic
from train_yolov8_baseline.py (patient-level aggregation, bootstrap AUC,
sensitivity at the M&E / TI&S TPP specificity targets) without re-training.

Usage:
    python scripts/eval_yolov8.py --weights /path/to/best.pt --contrast bf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# Allow importing the sibling training script's helpers.
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from schisto_mobile_ai.utils.io import ensure_dir

from train_yolov8_baseline import (
    aggregate_patient_scores,
    bootstrap_auc,
    patient_key_from_filename,
    patient_sens_at_spec,
    SPLIT_CSV,
)


def run(weights: str, contrast: str, imgsz: int, device: str, conf: float) -> None:
    from ultralytics import YOLO

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    data_root = REPO_ROOT / "data" / f"yolo_mar_to_nov_{contrast}"
    test_imgs_dir = data_root / "images" / "test"
    if not test_imgs_dir.exists():
        raise FileNotFoundError(f"Missing test tiles: {test_imgs_dir}")

    out_dir = ensure_dir(REPO_ROOT / "results" / f"yolov8_{contrast}_mar_to_nov")

    print(f"Weights:   {weights_path}")
    print(f"Contrast:  {contrast.upper()}")
    print(f"Test tiles:{test_imgs_dir}")
    print(f"Device:    {device}  imgsz={imgsz}  conf={conf}")

    model = YOLO(str(weights_path))

    detections: list[dict] = []
    results = model.predict(
        source=str(test_imgs_dir),
        imgsz=imgsz,
        conf=conf,
        device=device,
        stream=True,
        verbose=False,
    )
    n_seen = 0
    for r in results:
        n_seen += 1
        if n_seen % 5000 == 0:
            print(f"  ...processed {n_seen} tiles, {len(detections)} dets so far", flush=True)
        fname = Path(r.path).name
        pk = patient_key_from_filename(fname)
        if r.boxes is None:
            continue
        confs = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        for c, box in zip(confs, xyxy):
            detections.append({
                "tile":        fname,
                "patient_key": pk,
                "confidence":  float(c),
                "x1": float(box[0]), "y1": float(box[1]),
                "x2": float(box[2]), "y2": float(box[3]),
            })

    n_dets = len(detections)
    n_patients_with_dets = len(set(d["patient_key"] for d in detections))
    print(f"\n  Total detections: {n_dets}")
    print(f"  Patients with at least one detection: {n_patients_with_dets}")

    test_split = pd.read_csv(SPLIT_CSV)
    test_split = test_split[test_split["split"] == "test"]

    pat = aggregate_patient_scores(detections, test_split)
    n_pos = int(pat["target"].sum())
    n_neg = int(len(pat) - n_pos)
    print(f"  Patients: {len(pat)}  (pos={n_pos}, neg={n_neg})")

    auc, lo, hi = bootstrap_auc(pat["target"].values, pat["score"].values)
    me_sens, me_spec, me_thr = patient_sens_at_spec(pat, 0.965)
    tis_sens, tis_spec, tis_thr = patient_sens_at_spec(pat, 0.995)

    print(f"\n=== YOLOv8 {contrast.upper()} (mar->nov) — eval of {weights_path.name} ===")
    print(f"  AUC:           {auc:.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"  M&E target  (>=96.5% spec): sens={me_sens:.3f}  spec={me_spec:.3f}  thr={me_thr:.4f}")
    print(f"  TI&S target (>=99.5% spec): sens={tis_sens:.3f}  spec={tis_spec:.3f}  thr={tis_thr:.4f}")
    print()
    print("--- de Leon Derby reference (same direction, BF only) ---")
    print("  AUC (visual estimate from Fig 4B):  ~0.85")
    print("  M&E:  sens=0.76, TI&S: sens=0.53")

    pat.to_csv(out_dir / "patient_scores.csv", index=False)
    pd.DataFrame(detections).to_csv(out_dir / "tile_detections.csv", index=False)
    summary = {
        "contrast":      contrast,
        "weights":       str(weights_path),
        "patient_auc":   round(auc, 4),
        "auc_lo":        round(lo, 4),
        "auc_hi":        round(hi, 4),
        "me_sens":       round(me_sens, 4),
        "me_spec":       round(me_spec, 4),
        "me_threshold":  round(me_thr, 4),
        "tis_sens":      round(tis_sens, 4),
        "tis_spec":      round(tis_spec, 4),
        "tis_threshold": round(tis_thr, 4),
        "n_patients":    int(len(pat)),
        "n_positive":    n_pos,
        "n_detections":  n_dets,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_dir / 'summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained best.pt")
    ap.add_argument("--contrast", choices=["bf", "df"], default="bf")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps", help="mps | cpu | 0")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="Detection confidence floor for scoring (default 0.001)")
    args = ap.parse_args()
    run(args.weights, args.contrast, args.imgsz, args.device, args.conf)


if __name__ == "__main__":
    main()
