#!/usr/bin/env python3
"""
Cross-modal consistency fine-tuning.

Trains BF and DF classifiers jointly on paired images.
Loss = BCE(BF) + BCE(DF) + λ × cosine_distance(z_bf, z_df)

The consistency term anchors the supervised fine-tuning to preserve the
cross-modal alignment established during BYOL pre-training. Physical motivation:
BF and DF images of the same slide image the same physical object — their
encoder representations should remain similar even after supervised adaptation.

Expected benefit: DF model improvement (currently the weak link at 0.644 AUC),
because it gets to "borrow" signal from the stronger BF model's representations
via the consistency constraint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data import load_dual_contrast_data
from schisto_mobile_ai.data.paired_classification import PairedContrastDataset
from schisto_mobile_ai.models.simple_cnn import TinyConvClassifier
from schisto_mobile_ai.models.patient_aggregation import aggregate_patient_predictions
from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.logging import configure_logging
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder-weights", type=Path,
                   default=REPO_ROOT / "runs/ssl/pretrain_byol"
                           / "20260408_105425_pretrain_byol_byol_pretrain_100ep"
                           / "encoder_weights.pt")
    p.add_argument("--consistency-weight", type=float, default=0.1,
                   help="λ for cosine consistency loss (default 0.1)")
    p.add_argument("--pairs-csv", type=Path, default=REPO_ROOT / "metadata" / "pairs.csv")
    p.add_argument("--patients-csv", type=Path, default=REPO_ROOT / "metadata" / "patients.csv")
    p.add_argument("--split-csv", type=Path,
                   default=REPO_ROOT / "splits" / "random_patient_split.csv")
    p.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "runs" / "consistency_finetune")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cpu",
                   choices=("auto", "cpu", "mps"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--smoke", dest="smoke_test", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_auc(targets: list, probs: list) -> float:
    if len(set(targets)) < 2:
        return float("nan")
    frame = pd.DataFrame({"t": targets, "p": probs})
    pos = frame["t"] >= 0.5
    pc, nc = int(pos.sum()), int((~pos).sum())
    if pc == 0 or nc == 0:
        return float("nan")
    ranks = frame["p"].rank(method="average")
    return float((ranks[pos].sum() - pc * (pc + 1) / 2.0) / (pc * nc))


def _patient_auc(preds_df: pd.DataFrame, prob_col: str) -> float:
    pat = aggregate_patient_predictions(preds_df, patient_target_aggregation="max")
    return _safe_auc(
        pat["target"].tolist(),
        pat[prob_col].tolist(),
    )


# ── Training / eval loops ─────────────────────────────────────────────────────

def _train_epoch(
    bf_model: TinyConvClassifier,
    df_model: TinyConvClassifier,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    consistency_weight: float,
    device: torch.device,
) -> dict[str, float]:
    bf_model.train()
    df_model.train()

    total_loss = 0.0
    total_ce_bf = total_ce_df = total_cons = 0.0
    all_t_bf, all_p_bf = [], []
    all_t_df, all_p_df = [], []

    for batch in loader:
        bf_imgs = batch["brightfield_image"].to(device)
        df_imgs = batch["darkfield_image"].to(device)
        targets = batch["target"].float().to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass — get embeddings + logits for both modalities
        z_bf = bf_model.encoder(bf_imgs)                  # (B, feature_dim)
        z_df = df_model.encoder(df_imgs)
        logits_bf = bf_model.head(z_bf).squeeze(1)        # (B,)
        logits_df = df_model.head(z_df).squeeze(1)

        # Classification losses
        ce_bf = criterion(logits_bf, targets)
        ce_df = criterion(logits_df, targets)

        # Asymmetric consistency: only push DF toward BF (BF is the teacher).
        # z_bf.detach() stops gradients flowing back into the BF model,
        # so BF trains only on its classification loss — no degradation.
        # DF gets the extra signal: "your embedding should match the BF teacher."
        cos_sim = F.cosine_similarity(z_df, z_bf.detach(), dim=1).mean()
        cons_loss = 1.0 - cos_sim

        loss = ce_bf + ce_df + consistency_weight * cons_loss
        loss.backward()
        optimizer.step()

        bs = targets.size(0)
        total_loss   += float(loss.item())   * bs
        total_ce_bf  += float(ce_bf.item())  * bs
        total_ce_df  += float(ce_df.item())  * bs
        total_cons   += float(cons_loss.item()) * bs

        probs_bf = torch.sigmoid(logits_bf.detach()).cpu().numpy().tolist()
        probs_df = torch.sigmoid(logits_df.detach()).cpu().numpy().tolist()
        t = targets.detach().cpu().numpy().tolist()
        all_t_bf.extend(t); all_p_bf.extend(probs_bf)
        all_t_df.extend(t); all_p_df.extend(probs_df)

    n = max(sum(1 for _ in all_t_bf), 1)
    return {
        "loss":      total_loss / n,
        "ce_bf":     total_ce_bf / n,
        "ce_df":     total_ce_df / n,
        "cons_loss": total_cons / n,
        "auc_bf":    _safe_auc(all_t_bf, all_p_bf),
        "auc_df":    _safe_auc(all_t_df, all_p_df),
    }


def _eval_epoch(
    bf_model: TinyConvClassifier,
    df_model: TinyConvClassifier,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    bf_model.eval()
    df_model.eval()

    rows_bf, rows_df = [], []
    with torch.no_grad():
        for batch in loader:
            bf_imgs = batch["brightfield_image"].to(device)
            df_imgs = batch["darkfield_image"].to(device)
            targets = batch["target"].float().to(device)

            logits_bf = bf_model(bf_imgs).squeeze(1)
            logits_df = df_model(df_imgs).squeeze(1)

            probs_bf = torch.sigmoid(logits_bf).cpu().numpy().tolist()
            probs_df = torch.sigmoid(logits_df).cpu().numpy().tolist()
            t        = targets.cpu().numpy().tolist()

            for i in range(len(t)):
                meta = {
                    "patient_key": batch["patient_key"][i],
                    "pair_key":    batch["pair_key"][i],
                    "target":      float(t[i]),
                }
                rows_bf.append({**meta, "probability": float(probs_bf[i])})
                rows_df.append({**meta, "probability": float(probs_df[i])})

    preds_bf = pd.DataFrame(rows_bf)
    preds_df = pd.DataFrame(rows_df)

    pat_auc_bf = _patient_auc(preds_bf, "patient_probability_max")
    pat_auc_df = _patient_auc(preds_df, "patient_probability_max")

    metrics = {
        "val_patient_auc_bf": pat_auc_bf,
        "val_patient_auc_df": pat_auc_df,
        "val_patient_auc_mean": float(np.nanmean([pat_auc_bf, pat_auc_df])),
    }
    return metrics, preds_bf, preds_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = build_parser().parse_args()
    logger = configure_logging()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    epochs     = min(args.epochs, 2)  if args.smoke_test else args.epochs
    batch_size = min(args.batch_size, 4) if args.smoke_test else args.batch_size
    img_size   = min(args.img_size, 64) if args.smoke_test else args.img_size

    output_dir = ensure_dir(args.output_dir)
    done_flag  = output_dir / "done.json"
    if done_flag.exists() and not args.overwrite:
        logger.info("Output already exists at %s (use --overwrite)", output_dir)
        return 0

    # ── Data ──────────────────────────────────────────────────────────────────
    bundle = load_dual_contrast_data(
        pairs_csv=args.pairs_csv,
        patients_csv=args.patients_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        label_source="image",
        smoke_test=args.smoke_test,
    )

    train_frame = bundle.train_frame.copy()
    val_frame   = bundle.val_frame.copy()
    # target column is already 0.0/1.0 float from load_dual_contrast_data

    logger.info("Train: %d pairs (%d patients) | Val: %d pairs (%d patients)",
                len(train_frame), train_frame["patient_key"].nunique(),
                len(val_frame),   val_frame["patient_key"].nunique())

    train_ds = PairedContrastDataset(train_frame, image_size=img_size, train=True)
    val_ds   = PairedContrastDataset(val_frame,   image_size=img_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)

    # ── Models ────────────────────────────────────────────────────────────────
    def _make_model() -> TinyConvClassifier:
        m = TinyConvClassifier(base_channels=args.base_channels).to(device)
        if args.encoder_weights and args.encoder_weights.exists():
            state = torch.load(args.encoder_weights, map_location=device, weights_only=True)
            m.encoder.load_state_dict(state)
            logger.info("Loaded BYOL encoder from %s", args.encoder_weights)
        else:
            logger.warning("No encoder weights found — training from scratch")
        return m

    bf_model = _make_model()
    df_model = _make_model()

    # Prior bias initialisation
    pos_rate = float((train_frame["target"] == 1).sum()) / max(len(train_frame), 1)
    if 0.0 < pos_rate < 1.0:
        prior_bias = float(np.log(pos_rate / (1.0 - pos_rate)))
        with torch.no_grad():
            bf_model.head[-1].bias.fill_(prior_bias)
            df_model.head[-1].bias.fill_(prior_bias)

    # Weighted loss for class imbalance
    pos = float((train_frame["target"] == 1).sum())
    neg = float(len(train_frame) - pos)
    pos_weight = torch.tensor([(neg / pos) ** 0.5], device=device) if pos > 0 else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        list(bf_model.parameters()) + list(df_model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_bf_preds = best_df_preds = pd.DataFrame()

    for epoch in range(1, epochs + 1):
        train_m = _train_epoch(
            bf_model, df_model, train_loader,
            optimizer=optimizer, criterion=criterion,
            consistency_weight=args.consistency_weight, device=device,
        )
        val_m, preds_bf, preds_df = _eval_epoch(
            bf_model, df_model, val_loader,
            criterion=criterion, device=device,
        )

        row = {"epoch": epoch, **train_m, **val_m}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        score = val_m["val_patient_auc_mean"]
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_bf_preds = preds_bf.copy()
            best_df_preds = preds_df.copy()
            torch.save(bf_model.state_dict(), output_dir / "best_bf_model.pt")
            torch.save(df_model.state_dict(), output_dir / "best_df_model.pt")

        logger.info(
            "Epoch %d/%d | loss=%.4f  ce_bf=%.4f  ce_df=%.4f  cons=%.4f | "
            "val_auc_bf=%.4f  val_auc_df=%.4f",
            epoch, epochs,
            train_m["loss"], train_m["ce_bf"], train_m["ce_df"], train_m["cons_loss"],
            val_m["val_patient_auc_bf"], val_m["val_patient_auc_df"],
        )

    if not best_bf_preds.empty:
        best_bf_preds.to_csv(output_dir / "best_bf_val_preds.csv", index=False)
        best_df_preds.to_csv(output_dir / "best_df_val_preds.csv", index=False)

    history_df = pd.DataFrame(history)
    best_idx = int(history_df["val_patient_auc_mean"].fillna(-1).idxmax())
    best_bf_auc = float(history_df.loc[best_idx, "val_patient_auc_bf"])
    best_df_auc = float(history_df.loc[best_idx, "val_patient_auc_df"])

    summary = {
        "consistency_weight": args.consistency_weight,
        "best_epoch":   best_idx + 1,
        "best_bf_auc":  best_bf_auc,
        "best_df_auc":  best_df_auc,
        "best_mean_auc": float(history_df.loc[best_idx, "val_patient_auc_mean"]),
        "bf_model": str(output_dir / "best_bf_model.pt"),
        "df_model": str(output_dir / "best_df_model.pt"),
    }
    with open(done_flag, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nConsistency Fine-Tuning Summary")
    print(f"  consistency_weight: {args.consistency_weight}")
    print(f"  best_epoch:         {best_idx + 1}/{epochs}")
    print(f"  best_bf_auc:        {best_bf_auc:.4f}  (baseline: 0.692)")
    print(f"  best_df_auc:        {best_df_auc:.4f}  (baseline: 0.644)")
    print(f"  output_dir:         {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
