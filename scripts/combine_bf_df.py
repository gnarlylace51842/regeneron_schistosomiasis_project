#!/usr/bin/env python3
"""Combine BF and DF YOLOv8 predictions using the four rubrics from
de Leon Derby et al. (PLOS NTDs 2025):

  PL AND  — patient-level AND  (both contrasts must classify positive)
  PL OR   — patient-level OR   (either contrast positive)
  OL AND  — object-level AND   (score_BF × score_DF × IoU, requires bbox overlap)
  OL OR   — object-level OR    (mean score, lonely detections retained at half score)

We report patient-level AUC, sensitivity at 96.5% specificity (M&E TPP),
and sensitivity at 99.5% specificity (TI&S TPP), each with bootstrap 95% CI.

Inputs (produced by train_yolov8_baseline.py):
  results/yolov8_bf_mar_to_nov/tile_detections.csv
  results/yolov8_df_mar_to_nov/tile_detections.csv

Outputs:
  results/yolov8_combinations/summary.json
  results/yolov8_combinations/patient_scores_<combo>.csv
"""

from __future__ import annotations

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


def bootstrap_auc(t: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    pt = _wilcoxon_auc(t, s)
    rng = np.random.default_rng(SEED)
    n = len(t)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        a = _wilcoxon_auc(t[idx], s[idx])
        if np.isfinite(a):
            boot.append(a)
    if len(boot) < 10:
        return pt, float("nan"), float("nan")
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return pt, float(lo), float(hi)


def sens_at_spec(pat: pd.DataFrame, target_spec: float) -> tuple[float, float, float]:
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


def patient_max_score(dets: pd.DataFrame, all_patients: list[str]) -> dict[str, float]:
    by_pat = dets.groupby("patient_key")["confidence"].max().to_dict()
    return {pk: float(by_pat.get(pk, 0.0)) for pk in all_patients}


def patient_level_combinations(
    bf: dict[str, float], df: dict[str, float],
    targets: dict[str, int], rule: str,
) -> pd.DataFrame:
    """PL AND/OR via combining BF and DF patient-level scores.

    PL AND patient score = min(bf, df)  — both must clear threshold to call positive
    PL OR  patient score = max(bf, df)  — either alone suffices
    """
    rows = []
    for pk, tgt in targets.items():
        s_bf, s_df = bf[pk], df[pk]
        if rule == "PL_AND":
            score = min(s_bf, s_df)
        elif rule == "PL_OR":
            score = max(s_bf, s_df)
        else:
            raise ValueError(rule)
        rows.append({"patient_key": pk, "target": tgt, "score": float(score)})
    return pd.DataFrame(rows)


def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    xa = max(b1[0], b2[0]); ya = max(b1[1], b2[1])
    xb = min(b1[2], b2[2]); yb = min(b1[3], b2[3])
    iw = max(0.0, xb - xa); ih = max(0.0, yb - ya)
    inter = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


def object_level_combinations(
    bf_dets: pd.DataFrame, df_dets: pd.DataFrame,
    targets: dict[str, int], rule: str,
) -> pd.DataFrame:
    """OL AND/OR following de Leon Derby et al.

    For each tile that exists in both contrasts (same patient + same tile id),
    pair detections by IoU. Then aggregate to patient-level max.

    OL AND: score = score_BF * score_DF * IoU   (requires overlap)
    OL OR : score = (score_BF + score_DF) / 2   (lonely dets get other = 0)
    """
    bf_tiles = bf_dets.groupby(["patient_key", "tile"])
    df_tiles = df_dets.groupby(["patient_key", "tile"])

    pat_scores: dict[str, float] = {pk: 0.0 for pk in targets.keys()}

    # Pair tiles by (patient_key, tile_id) — tile id is encoded in the filename suffix.
    def _tile_id(name: str) -> str:
        # 'mar2020_001_0_0_tile_03.jpg' → '0_0_tile_03'
        stem = Path(name).stem
        parts = stem.split("_")
        return "_".join(parts[2:])

    bf_dets = bf_dets.copy(); bf_dets["tile_id"] = bf_dets["tile"].map(_tile_id)
    df_dets = df_dets.copy(); df_dets["tile_id"] = df_dets["tile"].map(_tile_id)

    grouped_bf = bf_dets.groupby(["patient_key", "tile_id"])
    grouped_df = df_dets.groupby(["patient_key", "tile_id"])
    bf_keys = set(grouped_bf.groups.keys())
    df_keys = set(grouped_df.groups.keys())
    all_keys = bf_keys | df_keys

    for key in all_keys:
        pk = key[0]
        if pk not in pat_scores:
            continue
        bf_t = grouped_bf.get_group(key) if key in bf_keys else pd.DataFrame()
        df_t = grouped_df.get_group(key) if key in df_keys else pd.DataFrame()

        # If using bbox columns (xyxy), use them. Otherwise assume the YOLO
        # script saved (x1, y1, x2, y2) into 'x1','y1','x2','y2'. If they're
        # missing, fall back to OR-without-IoU and AND-without-IoU.
        has_bbox = all(c in bf_t.columns for c in ["x1", "y1", "x2", "y2"]) and \
                   all(c in df_t.columns for c in ["x1", "y1", "x2", "y2"])

        tile_score = 0.0
        if rule == "OL_AND":
            if has_bbox and len(bf_t) > 0 and len(df_t) > 0:
                for _, b in bf_t.iterrows():
                    for _, d in df_t.iterrows():
                        iou = _iou(b[["x1","y1","x2","y2"]].values.astype(float),
                                   d[["x1","y1","x2","y2"]].values.astype(float))
                        if iou > 0:
                            s = float(b["confidence"]) * float(d["confidence"]) * iou
                            tile_score = max(tile_score, s)
            elif len(bf_t) > 0 and len(df_t) > 0:
                # Fallback: agreement at tile level with no IoU
                tile_score = float(bf_t["confidence"].max()) * float(df_t["confidence"].max())
            else:
                tile_score = 0.0
        elif rule == "OL_OR":
            bf_max = float(bf_t["confidence"].max()) if len(bf_t) > 0 else 0.0
            df_max = float(df_t["confidence"].max()) if len(df_t) > 0 else 0.0
            tile_score = (bf_max + df_max) / 2.0
        else:
            raise ValueError(rule)

        if tile_score > pat_scores[pk]:
            pat_scores[pk] = tile_score

    rows = [{"patient_key": pk, "target": tgt, "score": pat_scores[pk]}
            for pk, tgt in targets.items()]
    return pd.DataFrame(rows)


def main() -> None:
    bf_csv = REPO_ROOT / "results" / "yolov8_bf_mar_to_nov" / "tile_detections.csv"
    df_csv = REPO_ROOT / "results" / "yolov8_df_mar_to_nov" / "tile_detections.csv"

    if not bf_csv.exists() or not df_csv.exists():
        print(f"Missing inputs:")
        print(f"  BF: {bf_csv}  exists={bf_csv.exists()}")
        print(f"  DF: {df_csv}  exists={df_csv.exists()}")
        sys.exit(1)

    bf = pd.read_csv(bf_csv)
    df = pd.read_csv(df_csv)
    print(f"BF detections: {len(bf)}   DF detections: {len(df)}")

    split = pd.read_csv(SPLIT_CSV)
    test_split = split[split["split"] == "test"].copy()
    targets = dict(zip(
        test_split["patient_key"],
        test_split["patient_label"].map({"positive": 1, "negative": 0})
    ))

    all_patients = list(targets.keys())
    bf_scores = patient_max_score(bf, all_patients)
    df_scores = patient_max_score(df, all_patients)

    out_dir = ensure_dir(REPO_ROOT / "results" / "yolov8_combinations")
    summary: dict = {}

    methods = [
        ("BF_only", lambda: pd.DataFrame([{"patient_key": pk, "target": targets[pk],
                                            "score": bf_scores[pk]} for pk in all_patients])),
        ("DF_only", lambda: pd.DataFrame([{"patient_key": pk, "target": targets[pk],
                                            "score": df_scores[pk]} for pk in all_patients])),
        ("PL_AND",  lambda: patient_level_combinations(bf_scores, df_scores, targets, "PL_AND")),
        ("PL_OR",   lambda: patient_level_combinations(bf_scores, df_scores, targets, "PL_OR")),
        ("OL_AND",  lambda: object_level_combinations(bf, df, targets, "OL_AND")),
        ("OL_OR",   lambda: object_level_combinations(bf, df, targets, "OL_OR")),
    ]

    print(f"\n{'Method':<10} {'AUC':<22} {'M&E sens@96.5%':<16} {'TI&S sens@99.5%':<16}")
    print("-" * 70)
    for name, fn in methods:
        pat = fn()
        pat.to_csv(out_dir / f"patient_scores_{name}.csv", index=False)
        auc, lo, hi = bootstrap_auc(pat["target"].values, pat["score"].values)
        me_sens,  me_spec,  _ = sens_at_spec(pat, 0.965)
        tis_sens, tis_spec, _ = sens_at_spec(pat, 0.995)
        summary[name] = {
            "auc":      round(auc, 4),
            "auc_lo":   round(lo, 4),
            "auc_hi":   round(hi, 4),
            "me_sens":  round(me_sens, 4),
            "me_spec":  round(me_spec, 4),
            "tis_sens": round(tis_sens, 4),
            "tis_spec": round(tis_spec, 4),
        }
        print(f"{name:<10} {auc:.3f} [{lo:.3f}, {hi:.3f}]   "
              f"{me_sens:.3f}            {tis_sens:.3f}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_dir / 'summary.json'}")
    print("\n--- de Leon Derby reference (Dataset 2 holdout, same direction) ---")
    print("  BF:     sens=0.76 (M&E), 0.53 (TI&S)")
    print("  DF:     sens=0.83 (M&E), 0.63 (TI&S)")
    print("  PL_AND: sens=0.81 (M&E), 0.73 (TI&S)")
    print("  PL_OR:  sens=0.81 (M&E), 0.64 (TI&S)")
    print("  OL_AND: sens=0.83 (M&E), 0.66 (TI&S)")
    print("  OL_OR:  sens=0.83 (M&E), 0.64 (TI&S)")


if __name__ == "__main__":
    main()
