#!/usr/bin/env python3
"""Error analysis visualization — following Delahunt's advice.

For cross-study failure: find worst misclassified mar2020 patients and
visualize their BF images with annotation overlays (egg locations).

Shows:
  - False negatives: positive patients the model scored lowest (missed eggs)
  - False positives: negative patients the model scored highest (phantom eggs)
  - True positives: correctly classified, for reference

Outputs:
  results/error_analysis/false_negatives.png
  results/error_analysis/false_positives.png
  results/error_analysis/comparison_panel.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.utils.io import ensure_dir

RAW_DIR   = REPO_ROOT / "data" / "raw"
META_CSV  = REPO_ROOT / "metadata" / "images.csv"
ANN_MAR   = RAW_DIR / "mar2020" / "mar2020_docs" / "mar2020_annotations_08262024.csv"
ANN_NOV   = RAW_DIR / "nov2021" / "nov2021_docs" / "nov2021_annotations_08262024.csv"
PREDS_CSV = REPO_ROOT / "runs" / "cross_study_pretrained" / "mobilenet_v2" / "test_predictions.csv"
OUT_DIR   = ensure_dir(REPO_ROOT / "results" / "error_analysis")

DISPLAY_WIDTH = 900   # px width for display (images are 4032×3024)
CIRCLE_RADIUS = 30    # annotation circle radius at full resolution


def load_patient_data() -> pd.DataFrame:
    meta = pd.read_csv(META_CSV)
    preds = pd.read_csv(PREDS_CSV)
    pat_eggs = (meta[meta["study_id"] == "mar2020"]
                .groupby("patient_key")["patient_eggs"].first().reset_index())

    pat = (preds.groupby("patient_key")
           .agg(target=("target", "max"), score=("prob", "max"))
           .reset_index()
           .merge(pat_eggs, on="patient_key", how="left"))
    return pat


def get_bf_images(patient_key: str, meta: pd.DataFrame) -> pd.DataFrame:
    return (meta[(meta["patient_key"] == patient_key) &
                 (meta["contrast"] == "brightfield")]
            .sort_values("frame_num")
            .reset_index(drop=True))


def load_annotations(image_name: str, ann: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (confirmed_coords, doubtful_coords) as Nx2 arrays."""
    rows = ann[ann["imageName"] == image_name]
    confirmed = rows[rows["objectType"] == "S.haematobium"][["xCoord", "yCoord"]].values
    doubtful  = rows[rows["objectType"] == "doubtful"][["xCoord", "yCoord"]].values
    return confirmed, doubtful


def draw_patient_row(axes, patient_key: str, score: float, patient_eggs: int,
                     meta: pd.DataFrame, ann: pd.DataFrame,
                     label: str, label_color: str) -> None:
    """Draw all BF images for a patient across a row of axes."""
    imgs = get_bf_images(patient_key, meta)
    n_frames = len(imgs)

    for ax_idx, ax in enumerate(axes):
        ax.axis("off")
        if ax_idx >= n_frames:
            continue

        row = imgs.iloc[ax_idx]
        img_path = RAW_DIR / row["relative_path"]
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        scale = DISPLAY_WIDTH / orig_w
        new_w = DISPLAY_WIDTH
        new_h = int(orig_h * scale)
        img_small = img.resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(img_small)

        confirmed, doubtful = load_annotations(row["image_name"], ann)
        n_eggs = int(row["number_eggs"])

        ax.imshow(arr)

        # Draw annotation circles (scaled)
        for (x, y) in confirmed:
            circle = plt.Circle((x * scale, y * scale),
                                 CIRCLE_RADIUS * scale,
                                 color="#ff4444", fill=False, linewidth=1.5)
            ax.add_patch(circle)
        for (x, y) in doubtful:
            circle = plt.Circle((x * scale, y * scale),
                                 CIRCLE_RADIUS * scale,
                                 color="#ffaa00", fill=False, linewidth=1.2,
                                 linestyle="--")
            ax.add_patch(circle)

        ax.set_title(f"frame {ax_idx}  ({n_eggs} eggs)",
                     fontsize=8, pad=3)

    # Row label on first axis
    axes[0].set_ylabel(
        f"{label}\n{patient_key}\nscore={score:.3f}\ntotal={patient_eggs} eggs",
        fontsize=8, color=label_color, rotation=0, ha="right", va="center",
        labelpad=60
    )


def make_panel(patients: list[dict], title: str, out_path: Path,
               meta: pd.DataFrame, ann: pd.DataFrame) -> None:
    n_rows = len(patients)
    max_frames = max(
        len(get_bf_images(p["patient_key"], meta)) for p in patients
    )
    max_frames = max(max_frames, 1)

    fig, axes = plt.subplots(n_rows, max_frames,
                             figsize=(max_frames * 4.5, n_rows * 3.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if max_frames == 1:
        axes = axes[:, np.newaxis]

    for r, p in enumerate(patients):
        draw_patient_row(
            axes[r], p["patient_key"], p["score"], p["patient_eggs"],
            meta, ann, p["label"], p["color"]
        )

    # Legend
    handles = [
        mpatches.Patch(color="#ff4444", label="S. haematobium (confirmed)"),
        mpatches.Patch(color="#ffaa00", label="Doubtful"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    meta = pd.read_csv(META_CSV)
    ann  = pd.read_csv(ANN_MAR)
    pat  = load_patient_data()

    # False negatives: positive patients, lowest scores — pick high-egg-count ones first
    fn = (pat[pat["target"] == 1]
          .sort_values(["score", "patient_eggs"], ascending=[True, False])
          .head(20))
    # Pick 4 with varying egg counts for visual variety
    fn_select = (fn.sort_values("patient_eggs", ascending=False)
                 .drop_duplicates("patient_eggs")
                 .head(4))

    # False positives: negative patients, highest scores
    fp = (pat[pat["target"] == 0]
          .sort_values("score", ascending=False)
          .head(4))

    # True positives: positive patients, highest scores (correct)
    tp = (pat[pat["target"] == 1]
          .sort_values("score", ascending=False)
          .head(3))

    print("=== Selected patients ===")
    print("\nFalse negatives (missed):")
    print(fn_select[["patient_key", "patient_eggs", "score"]].to_string(index=False))
    print("\nFalse positives (phantom):")
    print(fp[["patient_key", "patient_eggs", "score"]].to_string(index=False))
    print("\nTrue positives (correct):")
    print(tp[["patient_key", "patient_eggs", "score"]].to_string(index=False))

    # --- False negatives panel ---
    fn_patients = [
        {"patient_key": r["patient_key"], "score": r["score"],
         "patient_eggs": int(r["patient_eggs"]),
         "label": "FALSE NEG", "color": "#cc0000"}
        for _, r in fn_select.iterrows()
    ]
    make_panel(fn_patients,
               "False Negatives — Positive patients scored LOW by MobileNetV2\n"
               "(trained on nov2021, zero-shot on mar2020)\n"
               "Red circles = annotated S. haematobium eggs",
               OUT_DIR / "false_negatives.png", meta, ann)

    # --- False positives panel ---
    fp_patients = [
        {"patient_key": r["patient_key"], "score": r["score"],
         "patient_eggs": int(r["patient_eggs"]),
         "label": "FALSE POS", "color": "#0055cc"}
        for _, r in fp.iterrows()
    ]
    make_panel(fp_patients,
               "False Positives — Negative patients scored HIGH by MobileNetV2\n"
               "(no annotated eggs present)",
               OUT_DIR / "false_positives.png", meta, ann)

    # --- Comparison panel: 2 FN + 2 TP side by side ---
    comp_patients = []
    for _, r in fn_select.head(2).iterrows():
        comp_patients.append({
            "patient_key": r["patient_key"], "score": r["score"],
            "patient_eggs": int(r["patient_eggs"]),
            "label": "MISSED", "color": "#cc0000"
        })
    for _, r in tp.head(2).iterrows():
        comp_patients.append({
            "patient_key": r["patient_key"], "score": r["score"],
            "patient_eggs": int(r["patient_eggs"]),
            "label": "CORRECT", "color": "#007700"
        })
    make_panel(comp_patients,
               "Comparison: Missed Positives vs Correctly Detected Positives\n"
               "MobileNetV2 trained on nov2021, zero-shot on mar2020",
               OUT_DIR / "comparison_panel.png", meta, ann)

    print("\nDone. Review the images in results/error_analysis/")
    print("Look for visual differences between missed and detected patients.")


if __name__ == "__main__":
    main()
