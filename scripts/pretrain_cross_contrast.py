#!/usr/bin/env python3
"""Cross-contrast self-supervised pre-training for schistosomiasis egg detection.

Uses BF/DF image pairs of the same slide as natural positive pairs in a
SimCLR-style contrastive learning framework. No labels are used. The shared
encoder is forced to learn representations that are stable across optical
regimes (brightfield absorption vs darkfield scattering), which biases it
toward egg morphology rather than illumination artefacts.

After pre-training, only the encoder weights are saved. The projection head
is discarded. Use finetune_ssl.py to attach a classification head and fine-tune
on progressively smaller fractions of the labelled data.

Key outputs:
    encoder_weights.pt    — encoder state dict for fine-tuning
    projector_weights.pt  — projection head weights (kept for reference)
    history.csv           — per-epoch training loss
    config.json           — full reproducibility snapshot
    embedding_stats.json  — alignment score distribution on val pairs (diagnostic)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.ssl_pairs import load_ssl_pairs, build_ssl_loader
from schisto_mobile_ai.models.cross_contrast_ssl import (
    CrossContrastSSLModel,
    nt_xent_loss,
    cross_contrast_alignment_score,
)
from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.logging import configure_logging
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything
from schisto_mobile_ai.utils.script_base import resolve_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path,
                        default=REPO_ROOT / "metadata" / "pairs.csv")
    parser.add_argument("--split-csv", type=Path,
                        default=REPO_ROOT / "splits" / "random_patient_split.csv")
    parser.add_argument("--raw-dir", type=Path,
                        default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="ssl_pretrain")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of pre-training epochs. 50 is a good default.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size. NT-Xent benefits from larger batches (more negatives).")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--base-channels", type=int, default=32,
                        help="Encoder base channel width. Must match downstream fine-tuning.")
    parser.add_argument("--projection-dim", type=int, default=128,
                        help="SSL projection head output dimension.")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="NT-Xent temperature. Lower = sharper distribution.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", dest="smoke_test", action="store_true",
                        help="Tiny run for environment verification.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _train_one_epoch(
    model: CrossContrastSSLModel,
    loader: torch.utils.data.DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: str,
    temperature: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_align = 0.0
    n_batches = 0

    for batch in loader:
        bf = batch["brightfield"].to(device)
        df = batch["darkfield"].to(device)

        optimizer.zero_grad(set_to_none=True)
        z_bf, z_df = model(bf, df)
        loss = nt_xent_loss(z_bf, z_df, temperature=temperature)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Track mean cosine alignment as a diagnostic (should increase with training)
            align = cross_contrast_alignment_score(z_bf, z_df).mean().item()
        total_loss += float(loss.item())
        total_align += float(align)
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "ssl_loss": total_loss / n,
        "mean_bf_df_alignment": total_align / n,
    }


@torch.no_grad()
def _eval_alignment(
    model: CrossContrastSSLModel,
    loader: torch.utils.data.DataLoader,
    *,
    device: str,
) -> dict[str, float]:
    """Compute alignment score distribution on the eval set.

    The alignment score (cosine similarity between BF and DF embeddings of the
    same pair) is the key inference-time gating signal for the conditional
    pipeline. We track its distribution here to monitor pre-training progress
    and to establish the threshold sweep range for Stage 4.
    """
    model.eval()
    scores: list[float] = []
    for batch in loader:
        bf = batch["brightfield"].to(device)
        df = batch["darkfield"].to(device)
        z_bf = model.project(bf)
        z_df = model.project(df)
        batch_scores = cross_contrast_alignment_score(z_bf, z_df).cpu().tolist()
        scores.extend(batch_scores)

    if not scores:
        return {}
    arr = np.array(scores, dtype=np.float64)
    return {
        "n_pairs": len(arr),
        "mean_alignment": float(arr.mean()),
        "std_alignment": float(arr.std()),
        "min_alignment": float(arr.min()),
        "p25_alignment": float(np.percentile(arr, 25)),
        "median_alignment": float(np.median(arr)),
        "p75_alignment": float(np.percentile(arr, 75)),
        "max_alignment": float(arr.max()),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)

    args.subset_size = None
    output_dir = resolve_output_dir(
        args=args,
        task_name="pretrain_cross_contrast",
        default_output_subdir="runs/ssl/pretrain_cross_contrast",
    )

    seed_everything(args.seed)
    device = resolve_device(args.device)

    epochs = min(args.epochs, 2) if args.smoke_test else args.epochs
    batch_size = min(args.batch_size, 16) if args.smoke_test else args.batch_size
    img_size = min(args.img_size, 128) if args.smoke_test else args.img_size

    # Load unlabelled pairs — train split only for strict separation
    bundle = load_ssl_pairs(
        pairs_csv=args.pairs_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        splits_to_include=("train",),
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
    logger.info(
        "SSL pre-training pairs: %d pairs from %d patients (splits: train only)",
        bundle.metadata["n_pairs"],
        bundle.metadata["n_patients"],
    )

    # Also build a small eval loader from val pairs for alignment diagnostics
    # Note: we only compute alignment scores here, no labels are used
    from schisto_mobile_ai.data.ssl_pairs import load_ssl_pairs as _load
    val_bundle = _load(
        pairs_csv=args.pairs_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        splits_to_include=("val",),
        smoke_test=args.smoke_test,
        seed=args.seed,
    )

    train_loader = build_ssl_loader(
        bundle,
        image_size=img_size,
        batch_size=batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = build_ssl_loader(
        val_bundle,
        image_size=img_size,
        batch_size=batch_size,
        num_workers=args.num_workers,
        seed=args.seed + 1,
    )

    model = CrossContrastSSLModel(
        base_channels=args.base_channels,
        projection_dim=args.projection_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Cosine LR schedule: smoothly decays to 0 over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=args.lr * 0.01,
    )

    history_rows: list[dict[str, Any]] = []
    history_path = output_dir / "history.csv"
    best_alignment = float("-inf")

    for epoch in range(1, epochs + 1):
        train_metrics = _train_one_epoch(
            model, train_loader,
            optimizer=optimizer, device=device, temperature=args.temperature,
        )
        scheduler.step()

        val_alignment = _eval_alignment(model, val_loader, device=device)

        row: dict[str, Any] = {
            "epoch": epoch,
            "ssl_loss": train_metrics["ssl_loss"],
            "train_mean_alignment": train_metrics["mean_bf_df_alignment"],
            "val_mean_alignment": val_alignment.get("mean_alignment", float("nan")),
            "val_median_alignment": val_alignment.get("median_alignment", float("nan")),
            "val_std_alignment": val_alignment.get("std_alignment", float("nan")),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        current_align = val_alignment.get("mean_alignment", float("-inf"))
        if current_align > best_alignment:
            best_alignment = current_align
            torch.save(model.encoder.state_dict(), output_dir / "encoder_weights.pt")
            torch.save(model.projector.state_dict(), output_dir / "projector_weights.pt")
            torch.save(model.state_dict(), output_dir / "ssl_model_weights.pt")

        logger.info(
            "Epoch %d/%d | ssl_loss=%.4f | train_align=%.4f | val_align=%.4f | lr=%.6f",
            epoch, epochs,
            train_metrics["ssl_loss"],
            train_metrics["mean_bf_df_alignment"],
            val_alignment.get("mean_alignment", float("nan")),
            float(scheduler.get_last_lr()[0]),
        )

    # Save final embedding stats on val set
    final_stats = _eval_alignment(model, val_loader, device=device)
    with open(output_dir / "embedding_stats.json", "w") as f:
        json.dump(final_stats, f, indent=2)

    # Save reproducibility snapshot
    config_snapshot = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "stage": "stage2_ssl_pretrain",
        "model": {
            "base_channels": args.base_channels,
            "projection_dim": args.projection_dim,
            "feature_dim": model.feature_dim,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "temperature": args.temperature,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_schedule": "cosine_annealing",
            "device": device,
            "seed": args.seed,
        },
        "data": {
            "pairs_csv": str(args.pairs_csv),
            "split_csv": str(args.split_csv),
            "raw_dir": str(args.raw_dir),
            "splits_used_for_pretrain": ["train"],
            "n_train_pairs": bundle.metadata["n_pairs"],
            "n_train_patients": bundle.metadata["n_patients"],
            "n_val_pairs_eval": val_bundle.metadata["n_pairs"],
        },
        "outputs": {
            "output_dir": str(output_dir),
            "encoder_weights": str(output_dir / "encoder_weights.pt"),
            "projector_weights": str(output_dir / "projector_weights.pt"),
            "ssl_model_weights": str(output_dir / "ssl_model_weights.pt"),
            "history_csv": str(history_path),
            "embedding_stats_json": str(output_dir / "embedding_stats.json"),
        },
        "ssl_method": "cross_contrast_simclr",
        "ssl_description": (
            "BF/DF pairs of the same slide as natural positive pairs. "
            "Shared encoder forced to learn physics-invariant (illumination-invariant) "
            "representations. NT-Xent loss with temperature=%.2f." % args.temperature
        ),
        "best_val_alignment": best_alignment,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print("Cross-Contrast SSL Pre-Training Summary")
    print(f"  device:           {device}")
    print(f"  train_pairs:      {bundle.metadata['n_pairs']}")
    print(f"  epochs:           {epochs}")
    print(f"  temperature:      {args.temperature}")
    print(f"  best_val_align:   {best_alignment:.4f}")
    print(f"  encoder_weights:  {output_dir / 'encoder_weights.pt'}")
    print(f"  history_csv:      {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
