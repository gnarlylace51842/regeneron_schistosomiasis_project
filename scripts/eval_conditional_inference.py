#!/usr/bin/env python3
"""Evaluate the conditional dual-contrast inference pipeline.

This script generates the compute-sensitivity tradeoff curve — the second key
figure of the project. It answers: "How much compute can we save by routing
easy cases to BF-only, and what sensitivity do we sacrifice?"

Conditional pipeline:
    For each image pair in a patient:
        1. Run BF encoder → confidence score p_bf = sigmoid(logit_bf)
        2. If p_bf >= theta_high:  predict POSITIVE (no DF needed)
        3. If p_bf <= theta_low:   predict NEGATIVE (no DF needed)
        4. Else (uncertain):       run DF encoder, fuse scores, predict

    Patient prediction: max over pair predictions

Gating options (--gate-mode):
    "confidence"  : gate on BF prediction confidence |p_bf - 0.5|
                    High confidence (far from 0.5) → BF alone is sufficient
                    Low confidence (near 0.5)       → uncertain, request DF
    "alignment"   : gate on cross-contrast embedding cosine similarity
                    (requires --ssl-model-weights from BYOL/SimCLR)
                    High alignment → contrasts agree → BF alone is sufficient
                    Low alignment  → contrasts disagree → uncertain, request DF

Sweep:
    For each uncertainty_threshold in linspace(0, 0.5, 101):
        compute (% pairs needing DF, patient sensitivity, patient specificity)

    Plot: sensitivity@fixed_specificity vs % DF usage
    This is the compute-sensitivity Pareto curve.

Key comparisons on same plot:
    - BF-only (0% DF usage, fixed sensitivity)
    - DF-only (100% DF usage, fixed sensitivity)
    - Always-on dual (100% DF usage, dual model sensitivity)
    - Conditional (curve from 0% to 100% DF usage)

Outputs:
    tradeoff_curve.csv         — full sweep data
    tradeoff_curve.png         — the key figure
    operating_points.json      — specific operating points for the paper
    config.json                — reproducibility snapshot
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.paired_classification import (
    PairedContrastDataset, load_dual_contrast_data,
)
from schisto_mobile_ai.models.byol_ssl import CrossContrastBYOL
from schisto_mobile_ai.models.simple_cnn import TinyConvClassifier
from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.logging import configure_logging
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf-model", type=Path, required=True,
                        help="Path to best_model.pt from a BF fine-tuning run.")
    parser.add_argument("--df-model", type=Path, required=True,
                        help="Path to best_model.pt from a DF fine-tuning run.")
    parser.add_argument("--pairs-csv", type=Path,
                        default=REPO_ROOT / "metadata" / "pairs.csv")
    parser.add_argument("--patients-csv", type=Path,
                        default=REPO_ROOT / "metadata" / "patients.csv")
    parser.add_argument("--split-csv", type=Path,
                        default=REPO_ROOT / "splits" / "random_patient_split.csv")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--eval-split", type=str, default="val",
                        choices=("val", "test"),
                        help="Which split to evaluate on.")
    parser.add_argument("--gate-mode", type=str, default="confidence",
                        choices=("confidence", "alignment", "both"),
                        help="Gating signal. 'confidence' uses |p_bf - 0.5|. "
                             "'alignment' uses BYOL cross-contrast cosine similarity. "
                             "'both' runs both modes and plots both curves.")
    parser.add_argument("--byol-weights", type=Path, default=None,
                        help="Path to byol_model_weights.pt for alignment-based gating. "
                             "Required when --gate-mode is 'alignment' or 'both'.")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--target-specificities", type=float, nargs="+",
                        default=[0.80, 0.85, 0.90],
                        help="Specificity levels at which to report sensitivity.")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "results" / "conditional_inference")
    parser.add_argument("--run-name", type=str, default="sweep")
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _load_classifier(path: Path, *, base_channels: int, device: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # Auto-detect base_channels from checkpoint weight shape
    first_conv_key = "encoder.features.0.weight"
    if first_conv_key in state:
        detected = int(state[first_conv_key].shape[0])
        if detected != base_channels:
            base_channels = detected
    model = TinyConvClassifier(base_channels=base_channels)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _score_pairs(
    bf_model: torch.nn.Module,
    df_model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: str,
    byol_model: CrossContrastBYOL | None = None,
) -> pd.DataFrame:
    """Run both models on every pair, return per-pair score dataframe.

    If byol_model is provided, also computes bf_df_alignment (cosine similarity
    between BYOL online encoder embeddings) for alignment-based gating.
    High alignment = BF and DF agree = BF alone is sufficient.
    Low alignment  = contrasts disagree = uncertain, request DF.
    """
    rows: list[dict[str, Any]] = []
    for batch in loader:
        bf_imgs = batch["brightfield_image"].to(device)
        df_imgs = batch["darkfield_image"].to(device)

        bf_logits = bf_model(bf_imgs).squeeze(1)
        df_logits = df_model(df_imgs).squeeze(1)

        p_bf = torch.sigmoid(bf_logits).cpu().numpy()
        p_df = torch.sigmoid(df_logits).cpu().numpy()

        # Alignment-based gating: cosine similarity in BYOL encoder space
        if byol_model is not None:
            z_bf = F.normalize(byol_model.online_encoder(bf_imgs), dim=1)
            z_df = F.normalize(byol_model.online_encoder(df_imgs), dim=1)
            alignment = (z_bf * z_df).sum(dim=1).cpu().numpy()
        else:
            alignment = [float("nan")] * len(p_bf)

        for i in range(len(p_bf)):
            rows.append({
                "pair_key": batch["pair_key"][i],
                "patient_key": batch["patient_key"][i],
                "target": float(batch["target"][i].item()),
                "p_bf": float(p_bf[i]),
                "p_df": float(p_df[i]),
                # Confidence: how far from 0.5 (high = certain BF prediction)
                "bf_confidence": float(abs(p_bf[i] - 0.5)),
                # BYOL cross-contrast alignment (high = contrasts agree = BF sufficient)
                "bf_df_alignment": float(alignment[i]),
                # Fused score: simple average of BF and DF
                "p_fused": float((p_bf[i] + p_df[i]) / 2.0),
            })
    return pd.DataFrame(rows)


def _patient_label(group: pd.DataFrame) -> float:
    """Patient is positive if ANY pair is positive (max aggregation)."""
    return float(group["target"].max())


def _conditional_patient_score(
    pairs: pd.DataFrame,
    *,
    threshold: float,
    gate_col: str,
) -> pd.DataFrame:
    """For a given uncertainty threshold, compute patient scores and DF usage.

    A pair is routed to DF (uncertain) if its gate score < threshold.
    gate_col: 'bf_confidence' — low confidence means uncertain.
    """
    pairs = pairs.copy()
    pairs["use_df"] = pairs[gate_col] < threshold
    pairs["pair_score"] = np.where(
        pairs["use_df"],
        pairs["p_fused"],   # uncertain: use BF+DF fused score
        pairs["p_bf"],      # certain: BF score is sufficient
    )

    patient_rows = []
    for patient_key, grp in pairs.groupby("patient_key"):
        patient_rows.append({
            "patient_key": patient_key,
            "target": _patient_label(grp),
            "patient_score": float(grp["pair_score"].max()),
            "n_pairs": len(grp),
            "n_df_pairs": int(grp["use_df"].sum()),
            "df_fraction": float(grp["use_df"].mean()),
        })
    return pd.DataFrame(patient_rows)


def _auc(targets: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(targets)) < 2:
        return float("nan")
    pos_mask = targets == 1
    neg_mask = ~pos_mask
    pc, nc = int(pos_mask.sum()), int(neg_mask.sum())
    if pc == 0 or nc == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").values
    return float((ranks[pos_mask].sum() - pc * (pc + 1) / 2.0) / (pc * nc))


def _sensitivity_at_specificity(
    targets: np.ndarray,
    scores: np.ndarray,
    *,
    target_specificity: float,
    n_thresholds: int = 201,
) -> tuple[float, float]:
    """Find the threshold that achieves target_specificity, return (sensitivity, threshold)."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_sens = 0.0
    best_thresh = 0.5
    pos = targets == 1
    neg = ~pos
    nc = int(neg.sum())
    if nc == 0:
        return float("nan"), float("nan")
    for t in thresholds:
        preds = scores >= t
        spec = float((~preds[neg]).sum()) / nc if nc > 0 else 0.0
        if spec >= target_specificity:
            sens = float(preds[pos].sum()) / int(pos.sum()) if pos.sum() > 0 else 0.0
            if sens > best_sens:
                best_sens = sens
                best_thresh = float(t)
    return best_sens, best_thresh


def _build_tradeoff_curve(
    pair_scores: pd.DataFrame,
    *,
    gate_col: str,
    n_thresholds: int = 101,
    target_specificities: list[float],
) -> pd.DataFrame:
    """Sweep uncertainty threshold, compute tradeoff at each point.

    Threshold range is inferred from the gate column's actual data range so
    that both bf_confidence [0, 0.5] and bf_df_alignment [0, 1] are handled
    correctly. At threshold=col_min all pairs go to DF; at threshold=col_max
    no pairs go to DF.
    """
    rows = []
    gate_min = float(pair_scores[gate_col].min())
    gate_max = float(pair_scores[gate_col].max())
    thresholds = np.linspace(gate_min, gate_max, n_thresholds)

    for thresh in thresholds:
        patient_df = _conditional_patient_score(pair_scores, threshold=thresh, gate_col=gate_col)
        targets = patient_df["target"].values
        scores = patient_df["patient_score"].values
        df_frac = float(patient_df["df_fraction"].mean())
        auc = _auc(targets, scores)

        row: dict[str, Any] = {
            "threshold": float(thresh),
            "df_fraction": df_frac,
            "patient_auc": auc,
        }
        for sp in target_specificities:
            sens, t = _sensitivity_at_specificity(targets, scores, target_specificity=sp)
            sp_key = str(sp).replace(".", "p")
            row[f"sensitivity_at_spec{sp_key}"] = sens
            row[f"threshold_for_spec{sp_key}"] = t

        rows.append(row)
    return pd.DataFrame(rows)


def _plot_tradeoff(
    curve: pd.DataFrame,
    *,
    bf_only_auc: float,
    df_only_auc: float,
    target_specificities: list[float],
    output_path: Path,
    run_name: str = "",
    align_curve: pd.DataFrame | None = None,
) -> None:
    """Plot the compute-sensitivity Pareto curve.

    If align_curve is provided, overlays the alignment-gated curve in a
    distinct colour for direct comparison with confidence-gated.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: patient AUC vs DF fraction
    ax = axes[0]
    ax.plot(curve["df_fraction"] * 100, curve["patient_auc"],
            color="#2ca02c", linewidth=2.5, label="Confidence gate (sweep)")
    if align_curve is not None:
        ax.plot(align_curve["df_fraction"] * 100, align_curve["patient_auc"],
                color="#E91E63", linewidth=2.5, linestyle="-.",
                label="Alignment gate — BYOL (sweep)")
    ax.axhline(bf_only_auc, color="#1f77b4", linestyle="--", linewidth=1.5,
               label=f"BF-only (0% DF)  AUC={bf_only_auc:.3f}")
    ax.axhline(df_only_auc, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"DF-only (100% DF) AUC={df_only_auc:.3f}")
    ax.set_xlabel("% pairs routed to DF (compute cost)")
    ax.set_ylabel("Patient-level AUC")
    ax.set_title("Conditional Inference: AUC vs Compute Cost")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: sensitivity at fixed specificity vs DF fraction
    ax = axes[1]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for sp, col in zip(target_specificities, colors):
        sp_key = str(sp).replace(".", "p")
        col_name = f"sensitivity_at_spec{sp_key}"
        if col_name in curve.columns:
            ax.plot(curve["df_fraction"] * 100,
                    pd.to_numeric(curve[col_name], errors="coerce"),
                    color=col, linewidth=2.5, label=f"Conf @ spec≥{sp:.0%}")
        if align_curve is not None and col_name in align_curve.columns:
            ax.plot(align_curve["df_fraction"] * 100,
                    pd.to_numeric(align_curve[col_name], errors="coerce"),
                    color=col, linewidth=2.0, linestyle="-.",
                    label=f"Align @ spec≥{sp:.0%}")
    ax.set_xlabel("% pairs routed to DF (compute cost)")
    ax.set_ylabel("Patient-level sensitivity")
    ax.set_title("Conditional Inference: Sensitivity vs Compute Cost")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Resource-Aware Conditional Dual-Contrast Pipeline  {run_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)

    output_dir = ensure_dir(args.output_dir / args.run_name)
    existing = list(output_dir.glob("tradeoff_curve*"))
    if existing and not args.overwrite:
        print(f"ERROR: outputs exist in {output_dir}. Pass --overwrite.", file=sys.stderr)
        return 1

    seed_everything(args.seed)
    device = resolve_device(args.device)

    # Load models
    bf_model = _load_classifier(args.bf_model, base_channels=args.base_channels, device=device)
    df_model = _load_classifier(args.df_model, base_channels=args.base_channels, device=device)
    logger.info("Loaded BF model from %s", args.bf_model)
    logger.info("Loaded DF model from %s", args.df_model)

    # Load BYOL model for alignment-based gating if requested
    byol_model: CrossContrastBYOL | None = None
    use_alignment = args.gate_mode in ("alignment", "both")
    if use_alignment:
        if args.byol_weights is None:
            print("ERROR: --byol-weights required for alignment gating.", file=sys.stderr)
            return 1
        byol_ckpt = torch.load(args.byol_weights, map_location=device)
        byol_state = byol_ckpt.get("model_state_dict", byol_ckpt) if isinstance(byol_ckpt, dict) else byol_ckpt
        # Detect base_channels from BYOL online encoder
        byol_bc = 32
        key = "online_encoder.features.0.weight"
        if key in byol_state:
            byol_bc = int(byol_state[key].shape[0])
        byol_model = CrossContrastBYOL(base_channels=byol_bc).to(device)
        byol_model.load_state_dict(byol_state)
        byol_model.eval()
        logger.info("Loaded BYOL model from %s (base_channels=%d)", args.byol_weights, byol_bc)

    # Load paired eval data
    bundle = load_dual_contrast_data(
        pairs_csv=args.pairs_csv,
        patients_csv=args.patients_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        label_source="image",
    )
    if args.eval_split == "val":
        eval_frame = bundle.val_frame
    else:
        # Rebuild with test split — for now use val as test proxy
        eval_frame = bundle.val_frame
        logger.warning("Using val split as eval (test split not separately built)")

    dataset = PairedContrastDataset(eval_frame, image_size=args.img_size, train=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    logger.info("Scoring %d pairs on %s split...", len(eval_frame), args.eval_split)
    pair_scores = _score_pairs(bf_model, df_model, loader, device=device, byol_model=byol_model)
    pair_scores.to_csv(output_dir / "pair_scores.csv", index=False)

    # BF-only and DF-only baselines
    def _patient_auc_from_col(col: str) -> float:
        patient_rows = []
        for pk, grp in pair_scores.groupby("patient_key"):
            patient_rows.append({
                "target": _patient_label(grp),
                "score": float(grp[col].max()),
            })
        pf = pd.DataFrame(patient_rows)
        return _auc(pf["target"].values, pf["score"].values)

    bf_only_auc = _patient_auc_from_col("p_bf")
    df_only_auc = _patient_auc_from_col("p_df")
    fused_auc = _patient_auc_from_col("p_fused")
    logger.info("BF-only patient AUC: %.4f", bf_only_auc)
    logger.info("DF-only patient AUC: %.4f", df_only_auc)
    logger.info("Always-fused patient AUC: %.4f", fused_auc)

    # Tradeoff curve — confidence gate (always computed)
    logger.info("Building tradeoff curve (confidence gate)...")
    curve = _build_tradeoff_curve(
        pair_scores,
        gate_col="bf_confidence",
        target_specificities=args.target_specificities,
    )
    curve.to_csv(output_dir / "tradeoff_curve.csv", index=False)

    # Alignment gate curve (only when BYOL model loaded)
    align_curve: pd.DataFrame | None = None
    if use_alignment and "bf_df_alignment" in pair_scores.columns:
        logger.info("Building tradeoff curve (alignment gate)...")
        align_curve = _build_tradeoff_curve(
            pair_scores,
            gate_col="bf_df_alignment",
            target_specificities=args.target_specificities,
        )
        align_curve.to_csv(output_dir / "tradeoff_curve_alignment.csv", index=False)

    # Plot
    _plot_tradeoff(
        curve,
        bf_only_auc=bf_only_auc,
        df_only_auc=df_only_auc,
        target_specificities=args.target_specificities,
        output_path=output_dir / "tradeoff_curve.png",
        run_name=args.run_name,
        align_curve=align_curve,
    )

    # Operating points summary
    peak_idx = int(curve["patient_auc"].idxmax())
    peak_row = curve.iloc[peak_idx]
    operating_points: dict[str, Any] = {
        "baselines": {
            "bf_only_patient_auc": round(bf_only_auc, 4),
            "df_only_patient_auc": round(df_only_auc, 4),
            "always_fused_patient_auc": round(fused_auc, 4),
        },
        "confidence_gate_peak": {
            "df_fraction": round(float(peak_row["df_fraction"]), 4),
            "patient_auc": round(float(peak_row["patient_auc"]), 4),
            "auc_gain_over_bf_only": round(float(peak_row["patient_auc"]) - bf_only_auc, 4),
            "auc_gain_over_always_fused": round(float(peak_row["patient_auc"]) - fused_auc, 4),
            **{k: round(float(v), 4) for k, v in peak_row.items()
               if k.startswith("sensitivity_at_spec") and np.isfinite(float(v))},
        },
        "conditional_curve_summary": {},
    }

    for sp in args.target_specificities:
        sp_key = str(sp).replace(".", "p")
        sens_col = f"sensitivity_at_spec{sp_key}"
        if sens_col not in curve.columns:
            continue
        valid = curve.dropna(subset=[sens_col])
        if valid.empty:
            continue
        # Find point on curve that matches DF-only sensitivity at this specificity
        df_only_sens, _ = _sensitivity_at_specificity(
            pair_scores.groupby("patient_key").apply(
                lambda g: pd.Series({"target": _patient_label(g), "score": g["p_df"].max()}),
                include_groups=False,
            ).reset_index(drop=True)["target"].values,
            pair_scores.groupby("patient_key").apply(
                lambda g: pd.Series({"target": _patient_label(g), "score": g["p_df"].max()}),
                include_groups=False,
            ).reset_index(drop=True)["score"].values,
            target_specificity=sp,
        )
        # Find lowest DF fraction achieving that sensitivity on the curve
        close = valid[valid[sens_col] >= df_only_sens * 0.98].sort_values("df_fraction")
        if not close.empty:
            min_df = float(close.iloc[0]["df_fraction"])
            df_saving = 1.0 - min_df
        else:
            df_saving = float("nan")

        operating_points["conditional_curve_summary"][f"spec_{sp:.0%}"] = {
            "df_only_sensitivity": round(float(df_only_sens), 4),
            "min_df_fraction_to_match": round(min_df, 4) if not np.isnan(df_saving) else None,
            "df_saving_fraction": round(df_saving, 4) if not np.isnan(df_saving) else None,
            "compute_saving_percent": round(df_saving * 100, 1) if not np.isnan(df_saving) else None,
        }

    with open(output_dir / "operating_points.json", "w") as f:
        json.dump(operating_points, f, indent=2)

    config_snapshot = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "stage": "stage4_conditional_inference",
        "bf_model": str(args.bf_model),
        "df_model": str(args.df_model),
        "byol_weights": str(args.byol_weights) if args.byol_weights else None,
        "eval_split": args.eval_split,
        "gate_mode": args.gate_mode,
        "n_eval_pairs": len(eval_frame),
        "n_eval_patients": int(pair_scores["patient_key"].nunique()),
        "target_specificities": args.target_specificities,
        "outputs": {
            "pair_scores": str(output_dir / "pair_scores.csv"),
            "tradeoff_curve": str(output_dir / "tradeoff_curve.csv"),
            "tradeoff_plot": str(output_dir / "tradeoff_curve.png"),
            "operating_points": str(output_dir / "operating_points.json"),
        },
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print("Conditional Inference Evaluation Summary")
    print(f"  eval_pairs:          {len(eval_frame)}")
    print(f"  eval_patients:       {pair_scores['patient_key'].nunique()}")
    print(f"  BF-only AUC:         {bf_only_auc:.4f}")
    print(f"  DF-only AUC:         {df_only_auc:.4f}")
    print(f"  Always-fused AUC:    {fused_auc:.4f}")
    for sp in args.target_specificities:
        sp_key_j = f"spec_{sp:.0%}"
        info = operating_points["conditional_curve_summary"].get(sp_key_j, {})
        saving = info.get("compute_saving_percent")
        print(f"  Spec≥{sp:.0%}: DF saving = {saving}% compute to match DF-only sensitivity")
    print(f"  plot:                {output_dir / 'tradeoff_curve.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
