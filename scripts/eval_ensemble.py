"""
Ensemble evaluation: average predictions from multiple BF models,
then run conditional inference against the existing best DF model.

Usage:
  python3 scripts/eval_ensemble.py

Outputs results to results/conditional_inference/byol_ensemble_val/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data import load_dual_contrast_data
from schisto_mobile_ai.data.paired_classification import PairedContrastDataset
from schisto_mobile_ai.models.simple_cnn import TinyConvClassifier
from schisto_mobile_ai.models.byol_ssl import CrossContrastBYOL
from schisto_mobile_ai.models.patient_aggregation import aggregate_patient_predictions
from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.logging import configure_logging
from schisto_mobile_ai.utils.reproducibility import resolve_device
from torch.utils.data import DataLoader

# ── Config ────────────────────────────────────────────────────────────────────

BF_MODELS = [
    REPO_ROOT / "runs/ssl/finetune/20260408_184036_finetune_ssl_byol_ft_bf_1p00/best_model.pt",
    REPO_ROOT / "runs/ssl/finetune/20260409_121501_finetune_ssl_byol_ft_bf_1p00_s1/best_model.pt",
    REPO_ROOT / "runs/ssl/finetune/20260409_121501_finetune_ssl_byol_ft_bf_1p00_s2/best_model.pt",
]
DF_MODEL_PATH = REPO_ROOT / "runs/ssl/finetune/20260408_230716_finetune_ssl_byol_ft_df_1p00/best_model.pt"
BYOL_WEIGHTS  = REPO_ROOT / "runs/ssl/pretrain_byol/20260408_105425_pretrain_byol_byol_pretrain_100ep/byol_model_weights.pt"

BASE_CHANNELS = 32
IMAGE_SIZE    = 224
BATCH_SIZE    = 16
N_TTA         = 8          # D4 TTA views
DEVICE        = torch.device("cpu")
OUT_DIR       = REPO_ROOT / "results/conditional_inference/byol_ensemble_val"

GATE_THRESHOLDS = np.linspace(0.0, 0.5, 200)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_auc(targets, probs) -> float:
    t = np.array(targets, dtype=float)
    p = np.array(probs, dtype=float)
    if len(np.unique(t)) < 2:
        return float("nan")
    pos_mask = t >= 0.5
    pc, nc = pos_mask.sum(), (~pos_mask).sum()
    if pc == 0 or nc == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").values
    return float((ranks[pos_mask].sum() - pc * (pc + 1) / 2.0) / (pc * nc))


def load_classifier(path: Path) -> TinyConvClassifier:
    m = TinyConvClassifier(base_channels=BASE_CHANNELS)
    state = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


def load_byol_alignment_model() -> CrossContrastBYOL:
    m = CrossContrastBYOL(base_channels=BASE_CHANNELS)
    state = torch.load(BYOL_WEIGHTS, map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


@torch.no_grad()
def score_pairs_ensemble(
    bf_models: list[TinyConvClassifier],
    df_model: TinyConvClassifier,
    eval_frame: pd.DataFrame,
    byol_model: CrossContrastBYOL,
) -> pd.DataFrame:
    """Score all pairs using ensemble BF (averaged over models + TTA) and single DF (TTA)."""

    n = len(eval_frame)
    bf_probs_acc = np.zeros(n, dtype=np.float64)   # accumulate over models × TTA views
    df_probs_acc = np.zeros(n, dtype=np.float64)
    align_acc    = np.zeros(n, dtype=np.float64)

    n_bf_passes = len(bf_models) * N_TTA
    n_df_passes = N_TTA

    # ── BF ensemble + TTA ────────────────────────────────────────────────────
    for model_idx, bf_model in enumerate(bf_models):
        bf_model.to(DEVICE)
        for tta_view in range(N_TTA):
            ds = PairedContrastDataset(eval_frame, image_size=IMAGE_SIZE, train=False,
                                       tta_view=tta_view)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            offset = 0
            for batch in loader:
                imgs = batch["brightfield_image"].to(DEVICE)
                logits = bf_model(imgs).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
                bs = len(probs)
                bf_probs_acc[offset:offset + bs] += probs
                offset += bs
        bf_model.cpu()

    bf_probs_acc /= n_bf_passes

    # ── BYOL alignment (clean pass, BF only) ─────────────────────────────────
    byol_model.to(DEVICE)
    ds_clean = PairedContrastDataset(eval_frame, image_size=IMAGE_SIZE, train=False, tta_view=None)
    loader_clean = DataLoader(ds_clean, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    offset = 0
    for batch in loader_clean:
        bf_imgs = batch["brightfield_image"].to(DEVICE)
        df_imgs = batch["darkfield_image"].to(DEVICE)
        z_bf = byol_model.online_encoder(bf_imgs)
        z_df = byol_model.online_encoder(df_imgs)
        import torch.nn.functional as F
        sim = F.cosine_similarity(z_bf, z_df, dim=1).cpu().numpy()
        bs = len(sim)
        align_acc[offset:offset + bs] = sim
        offset += bs
    byol_model.cpu()

    # ── DF single model + TTA ─────────────────────────────────────────────────
    df_model.to(DEVICE)
    for tta_view in range(N_TTA):
        ds = PairedContrastDataset(eval_frame, image_size=IMAGE_SIZE, train=False,
                                   tta_view=tta_view)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        offset = 0
        for batch in loader:
            imgs = batch["darkfield_image"].to(DEVICE)
            logits = df_model(imgs).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            bs = len(probs)
            df_probs_acc[offset:offset + bs] += probs
            offset += bs
    df_model.cpu()

    df_probs_acc /= n_df_passes

    # ── Assemble result frame ─────────────────────────────────────────────────
    # Metadata from clean pass
    meta_rows = []
    ds_meta = PairedContrastDataset(eval_frame, image_size=IMAGE_SIZE, train=False, tta_view=None)
    loader_meta = DataLoader(ds_meta, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    for batch in loader_meta:
        for i in range(len(batch["target"])):
            meta_rows.append({
                "pair_key":    batch["pair_key"][i],
                "patient_key": batch["patient_key"][i],
                "target":      float(batch["target"][i]),
            })
    meta_df = pd.DataFrame(meta_rows)

    meta_df["p_bf"]           = bf_probs_acc
    meta_df["p_df"]           = df_probs_acc
    meta_df["bf_confidence"]  = (meta_df["p_bf"] - 0.5).abs()
    meta_df["bf_df_alignment"]= align_acc
    meta_df["p_fused"]        = 0.6 * meta_df["p_bf"] + 0.4 * meta_df["p_df"]
    meta_df["p_weighted"]     = meta_df["p_fused"]
    return meta_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger = configure_logging()
    ensure_dir(OUT_DIR)

    bundle = load_dual_contrast_data(
        pairs_csv=REPO_ROOT / "metadata/pairs.csv",
        patients_csv=REPO_ROOT / "metadata/patients.csv",
        split_csv=REPO_ROOT / "splits/random_patient_split.csv",
        raw_dir=REPO_ROOT / "data/raw",
        label_source="image",
    )
    eval_frame = bundle.val_frame

    logger.info("Loading %d BF models + 1 DF model", len(BF_MODELS))
    bf_models = [load_classifier(p) for p in BF_MODELS]
    df_model  = load_classifier(DF_MODEL_PATH)
    byol_model = load_byol_alignment_model()

    logger.info("Scoring %d pairs (ensemble BF × %d TTA, DF × %d TTA)...",
                len(eval_frame), len(BF_MODELS) * N_TTA, N_TTA)
    pair_scores = score_pairs_ensemble(bf_models, df_model, eval_frame, byol_model)
    pair_scores.to_csv(OUT_DIR / "pair_scores.csv", index=False)

    # ── Baselines ─────────────────────────────────────────────────────────────
    def patient_auc(col: str) -> float:
        pat = pair_scores.groupby("patient_key").agg(
            score=(col, "max"), label=("target", "max")
        ).reset_index()
        return _safe_auc(pat["label"], pat["score"])

    bf_auc    = patient_auc("p_bf")
    df_auc    = patient_auc("p_df")
    fused_auc = patient_auc("p_fused")

    logger.info("Ensemble BF-only patient AUC: %.4f  (3-model × TTA=8)", bf_auc)
    logger.info("DF-only patient AUC:           %.4f  (TTA=8)", df_auc)
    logger.info("Always-fused AUC:              %.4f", fused_auc)

    # ── Conditional sweep ─────────────────────────────────────────────────────
    logger.info("Sweeping confidence gate...")
    curve_rows = []
    for gate in GATE_THRESHOLDS:
        uncertain = pair_scores["bf_confidence"] < gate
        pair_scores["p_use"] = np.where(uncertain, pair_scores["p_fused"], pair_scores["p_bf"])
        df_frac = float(uncertain.mean())
        pat = pair_scores.groupby("patient_key").agg(
            score=("p_use", "max"), label=("target", "max")
        ).reset_index()
        auc = _safe_auc(pat["label"], pat["score"])
        curve_rows.append({"threshold": gate, "df_fraction": df_frac, "patient_auc": auc})

    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(OUT_DIR / "tradeoff_curve.csv", index=False)

    peak_row = curve_df.loc[curve_df["patient_auc"].idxmax()]
    peak_auc  = float(peak_row["patient_auc"])
    peak_df   = float(peak_row["df_fraction"])
    peak_gate = float(peak_row["threshold"])

    logger.info("Conditional peak AUC: %.4f at df_fraction=%.2f (gate=%.3f)",
                peak_auc, peak_df, peak_gate)

    # ── Operating point metrics ───────────────────────────────────────────────
    uncertain_at_peak = pair_scores["bf_confidence"] < peak_gate
    pair_scores["p_use"] = np.where(uncertain_at_peak, pair_scores["p_fused"], pair_scores["p_bf"])
    pat_peak = pair_scores.groupby("patient_key").agg(
        score=("p_use", "max"), label=("target", "max")
    ).reset_index()

    from sklearn.metrics import roc_curve, f1_score  # type: ignore
    fpr, tpr, threshs = roc_curve(pat_peak["label"], pat_peak["score"])
    best_t = threshs[np.argmax(tpr - fpr)]
    preds_bin = (pat_peak["score"] >= best_t).astype(int)
    f1  = float(f1_score(pat_peak["label"], preds_bin, zero_division=0))
    sens = float(tpr[np.argmax(tpr - fpr)])
    spec = float(1 - fpr[np.argmax(tpr - fpr)])

    summary = {
        "n_bf_models": len(BF_MODELS),
        "n_tta": N_TTA,
        "bf_only_auc":    round(bf_auc, 4),
        "df_only_auc":    round(df_auc, 4),
        "always_fused_auc": round(fused_auc, 4),
        "conditional_peak_auc": round(peak_auc, 4),
        "peak_df_fraction": round(peak_df, 4),
        "peak_gate": round(peak_gate, 4),
        "peak_f1": round(f1, 4),
        "peak_sensitivity": round(sens, 4),
        "peak_specificity": round(spec, 4),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nEnsemble Evaluation Summary")
    print(f"  BF ensemble AUC (3 models × TTA=8): {bf_auc:.4f}  (solo best: 0.7165)")
    print(f"  DF AUC (TTA=8):                      {df_auc:.4f}  (solo best: 0.6385)")
    print(f"  Conditional peak AUC:                {peak_auc:.4f}  at {peak_df:.0%} DF")
    print(f"  Best single system (BYOL+TTA):       0.7630")
    print(f"  Gain from ensemble:                  {peak_auc - 0.7630:+.4f}")


if __name__ == "__main__":
    main()
