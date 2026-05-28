#!/usr/bin/env python3
"""Train YOLOv8 baseline on the schistosomiasis tiled dataset.

Replicates de Leon Derby et al. (PLOS NTDs 2025) methodology:
  - YOLOv8 fine-tuned from COCO 2017 weights
  - 640×640 input tiles
  - Patient-level classification by max detection confidence per patient

Evaluation matches their TPP framework:
  - Patient considered positive if any detection ≥ confidence threshold in any image
  - Sensitivity at 96.5% specificity (M&E TPP)
  - Sensitivity at 99.5% specificity (TI&S TPP)
  - Patient-level AUC with bootstrap 95% CI

Usage:
    python scripts/train_yolov8_baseline.py --contrast bf --epochs 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR   = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.reproducibility import seed_everything

SPLIT_CSV = REPO_ROOT / "splits" / "mar_to_nov_split.csv"
N_BOOT = 2000
SEED   = 42


def _wilcoxon_auc(t: np.ndarray, s: np.ndarray) -> float:
    pos = t == 1
    pc, nc = int(pos.sum()), int((~pos).sum())
    if pc == 0 or nc == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").values
    return float((ranks[pos].sum() - pc * (pc + 1) / 2.0) / (pc * nc))


def bootstrap_auc(t: np.ndarray, s: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    pt = _wilcoxon_auc(t, s)
    rng = np.random.default_rng(SEED)
    n = len(t)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = _wilcoxon_auc(t[idx], s[idx])
        if np.isfinite(a):
            boot.append(a)
    if len(boot) < 10:
        return pt, float("nan"), float("nan")
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return pt, float(lo), float(hi)


def patient_sens_at_spec(pat: pd.DataFrame, target_spec: float) -> tuple[float, float, float]:
    """Return (sens, spec, threshold) where threshold gives sens at >= target_spec."""
    thresholds = np.unique(np.concatenate([[0.0], np.sort(pat["score"].unique()), [1.0]]))
    best = (0.0, 0.0, 0.0)
    for thr in thresholds:
        pred = (pat["score"] >= thr).astype(int)
        tp = int(((pred == 1) & (pat["target"] == 1)).sum())
        fn = int(((pred == 0) & (pat["target"] == 1)).sum())
        tn = int(((pred == 0) & (pat["target"] == 0)).sum())
        fp = int(((pred == 1) & (pat["target"] == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if spec >= target_spec and sens > best[0]:
            best = (sens, spec, float(thr))
    return best


def patient_key_from_filename(name: str) -> str:
    """nov2021_001_2_0_tile_03.jpg → nov2021_001."""
    stem = Path(name).stem
    parts = stem.split("_")
    return f"{parts[0]}_{parts[1]}"


def aggregate_patient_scores(
    detections: list[dict],
    test_split: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate tile-level detection confidences to patient-level scores.

    For each patient: score = max(confidence) over all detections in all
    that patient's tiles (i.e. across all FOVs and all 30 tiles per FOV).
    Patients with no detections receive score = 0.
    """
    rows = []
    by_patient: dict[str, float] = {}
    for det in detections:
        pk = det["patient_key"]
        conf = det["confidence"]
        if pk not in by_patient or conf > by_patient[pk]:
            by_patient[pk] = conf

    pat_label = test_split.set_index("patient_key")["patient_label"].map(
        {"positive": 1, "negative": 0}
    )
    for pk, lbl in pat_label.items():
        rows.append({
            "patient_key": pk,
            "target": int(lbl),
            "score":  float(by_patient.get(pk, 0.0)),
        })
    return pd.DataFrame(rows)


def run(contrast: str, epochs: int, model_size: str, batch: int, image_size: int) -> None:
    seed_everything(SEED)
    from ultralytics import YOLO

    contrast_short = "bf" if contrast == "brightfield" else "df"
    data_root = REPO_ROOT / "data" / f"yolo_mar_to_nov_{contrast_short}"
    data_yaml = data_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing dataset: run build_yolo_dataset.py first ({data_yaml})")

    out_dir = ensure_dir(REPO_ROOT / "results" / f"yolov8_{contrast_short}_mar_to_nov")
    run_dir = ensure_dir(REPO_ROOT / "runs" / f"yolov8_{contrast_short}_mar_to_nov")

    print(f"Contrast:   {contrast} ({contrast_short.upper()})")
    print(f"Model:      yolov8{model_size}.pt (COCO pre-trained)")
    print(f"Epochs:     {epochs}")
    print(f"Image size: {image_size}")
    print(f"Batch:      {batch}")

    model = YOLO(f"yolov8{model_size}.pt")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device="mps",
        project=str(run_dir),
        name="train",
        exist_ok=True,
        patience=20,
        seed=SEED,
        save=True,
        verbose=True,
    )

    best_weights = run_dir / "train" / "weights" / "best.pt"
    print(f"\nLoading best weights: {best_weights}")
    model = YOLO(str(best_weights))

    test_imgs_dir = data_root / "images" / "test"
    print(f"Running inference on test tiles: {test_imgs_dir}")

    detections = []
    results = model.predict(
        source=str(test_imgs_dir),
        imgsz=image_size,
        conf=0.001,
        device="mps",
        stream=True,
        verbose=False,
    )
    for r in results:
        fname = Path(r.path).name
        pk = patient_key_from_filename(fname)
        if r.boxes is None:
            continue
        confs = r.boxes.conf.cpu().numpy()
        xyxy  = r.boxes.xyxy.cpu().numpy()
        for c, box in zip(confs, xyxy):
            detections.append({
                "tile":         fname,
                "patient_key":  pk,
                "confidence":   float(c),
                "x1":           float(box[0]),
                "y1":           float(box[1]),
                "x2":           float(box[2]),
                "y2":           float(box[3]),
            })

    n_dets = len(detections)
    n_patients_with_dets = len(set(d["patient_key"] for d in detections))
    print(f"  Total detections: {n_dets}")
    print(f"  Patients with at least one detection: {n_patients_with_dets}")

    test_split = pd.read_csv(SPLIT_CSV)
    test_split = test_split[test_split["split"] == "test"]

    pat = aggregate_patient_scores(detections, test_split)
    n_pos = int(pat["target"].sum())
    n_neg = int(len(pat) - n_pos)
    print(f"  Patients: {len(pat)}  (pos={n_pos}, neg={n_neg})")

    auc, lo, hi = bootstrap_auc(pat["target"].values, pat["score"].values)
    me_sens,  me_spec,  me_thr  = patient_sens_at_spec(pat, 0.965)
    tis_sens, tis_spec, tis_thr = patient_sens_at_spec(pat, 0.995)

    print(f"\n=== YOLOv8-{model_size} {contrast_short.upper()} (mar→nov) ===")
    print(f"  AUC:           {auc:.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"  M&E target  (≥96.5% spec): sens={me_sens:.3f}  spec={me_spec:.3f}  thr={me_thr:.4f}")
    print(f"  TI&S target (≥99.5% spec): sens={tis_sens:.3f}  spec={tis_spec:.3f}  thr={tis_thr:.4f}")
    print()
    print("--- de Leon Derby reference (same direction, BF only) ---")
    print("  AUC (visual estimate from Fig 4B):  ~0.85")
    print("  M&E:  sens=0.76, TI&S: sens=0.53")

    pat.to_csv(out_dir / "patient_scores.csv", index=False)
    pd.DataFrame(detections).to_csv(out_dir / "tile_detections.csv", index=False)
    summary = {
        "contrast":      contrast,
        "model":         f"yolov8{model_size}",
        "epochs":        epochs,
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
    ap.add_argument("--contrast", choices=["brightfield", "darkfield", "bf", "df"], default="bf")
    ap.add_argument("--epochs",   type=int, default=100)
    ap.add_argument("--model",    choices=["n", "s", "m"], default="s",
                    help="YOLOv8 size: n(ano), s(mall), m(edium)")
    ap.add_argument("--batch",    type=int, default=16)
    ap.add_argument("--imgsz",    type=int, default=640)
    args = ap.parse_args()

    contrast = {"bf": "brightfield", "df": "darkfield"}.get(args.contrast, args.contrast)
    run(contrast, args.epochs, args.model, args.batch, args.imgsz)


if __name__ == "__main__":
    main()
