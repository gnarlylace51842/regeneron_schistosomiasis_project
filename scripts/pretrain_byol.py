#!/usr/bin/env python3
"""Cross-contrast BYOL pre-training for schistosomiasis egg detection.

BYOL fallback for when SimCLR shows poor transfer due to small batch / small
dataset constraints. BYOL requires no negative pairs, making it effective with
batch_size=16 or 32 on datasets of ~1500 pairs.

Key difference from SimCLR: uses a momentum (EMA) target network and a predictor
head to prevent collapse, rather than contrasting against explicit negatives.
Works well on small datasets where SimCLR's need for many negatives is a bottleneck.

Usage (if SimCLR fine-tuning underperforms scratch):
    python scripts/pretrain_byol.py --run-name byol_pretrain_100ep --epochs 100
    python scripts/finetune_ssl.py --encoder-weights runs/ssl/byol/.../encoder_weights.pt ...
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.ssl_pairs import load_ssl_pairs, build_ssl_loader
from schisto_mobile_ai.models.byol_ssl import CrossContrastBYOL, cross_contrast_alignment_score_byol
from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.logging import configure_logging
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything
from schisto_mobile_ai.utils.script_base import resolve_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, default=REPO_ROOT / "metadata" / "pairs.csv")
    parser.add_argument("--split-csv", type=Path,
                        default=REPO_ROOT / "splits" / "random_patient_split.csv")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="byol_pretrain")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="BYOL works well with smaller batches than SimCLR.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--ema-decay", type=float, default=0.996,
                        help="EMA decay for target network. 0.996 is standard BYOL.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", dest="smoke_test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _train_one_epoch(
    model: CrossContrastBYOL,
    loader: torch.utils.data.DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_align = 0.0
    n_batches = 0

    for batch in loader:
        bf = batch["brightfield"].to(device)
        df = batch["darkfield"].to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = model(bf, df)
        loss.backward()
        optimizer.step()
        model.update_target()  # EMA update after each step

        with torch.no_grad():
            align = cross_contrast_alignment_score_byol(model, bf, df).mean().item()

        total_loss += float(loss.item())
        total_align += float(align)
        n_batches += 1

    n = max(n_batches, 1)
    return {"byol_loss": total_loss / n, "mean_bf_df_alignment": total_align / n}


@torch.no_grad()
def _eval_alignment(
    model: CrossContrastBYOL,
    loader: torch.utils.data.DataLoader,
    *,
    device: str,
) -> dict[str, float]:
    model.eval()
    scores: list[float] = []
    for batch in loader:
        bf = batch["brightfield"].to(device)
        df = batch["darkfield"].to(device)
        batch_scores = cross_contrast_alignment_score_byol(model, bf, df).cpu().tolist()
        scores.extend(batch_scores)
    if not scores:
        return {}
    arr = np.array(scores, dtype=np.float64)
    return {
        "n_pairs": len(arr),
        "mean_alignment": float(arr.mean()),
        "std_alignment": float(arr.std()),
        "median_alignment": float(np.median(arr)),
        "min_alignment": float(arr.min()),
        "max_alignment": float(arr.max()),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.subset_size = None
    logger = configure_logging(quiet=args.quiet)

    output_dir = resolve_output_dir(
        args=args,
        task_name="pretrain_byol",
        default_output_subdir="runs/ssl/pretrain_byol",
    )

    seed_everything(args.seed)
    device = resolve_device(args.device)

    epochs = min(args.epochs, 2) if args.smoke_test else args.epochs
    batch_size = min(args.batch_size, 16) if args.smoke_test else args.batch_size
    img_size = min(args.img_size, 128) if args.smoke_test else args.img_size

    bundle = load_ssl_pairs(
        pairs_csv=args.pairs_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        splits_to_include=("train",),
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
    val_bundle = load_ssl_pairs(
        pairs_csv=args.pairs_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        splits_to_include=("val",),
        smoke_test=args.smoke_test,
        seed=args.seed,
    )

    train_loader = build_ssl_loader(bundle, image_size=img_size, batch_size=batch_size,
                                    num_workers=args.num_workers, seed=args.seed)
    val_loader = build_ssl_loader(val_bundle, image_size=img_size, batch_size=batch_size,
                                  num_workers=args.num_workers, seed=args.seed + 1)

    logger.info("BYOL pre-training: %d train pairs, %d val pairs",
                bundle.metadata["n_pairs"], val_bundle.metadata["n_pairs"])

    model = CrossContrastBYOL(
        base_channels=args.base_channels,
        projection_dim=args.projection_dim,
        ema_decay=args.ema_decay,
    ).to(device)

    # Only online network parameters go into the optimizer; target is EMA only
    online_params = (
        list(model.online_encoder.parameters())
        + list(model.online_projector.parameters())
        + list(model.predictor.parameters())
    )
    optimizer = torch.optim.AdamW(online_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=args.lr * 0.01
    )

    history_rows: list[dict[str, Any]] = []
    history_path = output_dir / "history.csv"
    best_alignment = float("-inf")

    for epoch in range(1, epochs + 1):
        train_m = _train_one_epoch(model, train_loader, optimizer=optimizer, device=device)
        scheduler.step()
        val_align = _eval_alignment(model, val_loader, device=device)

        row: dict[str, Any] = {
            "epoch": epoch,
            "byol_loss": train_m["byol_loss"],
            "train_mean_alignment": train_m["mean_bf_df_alignment"],
            "val_mean_alignment": val_align.get("mean_alignment", float("nan")),
            "val_median_alignment": val_align.get("median_alignment", float("nan")),
            "val_std_alignment": val_align.get("std_alignment", float("nan")),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        current_align = val_align.get("mean_alignment", float("-inf"))
        if current_align > best_alignment:
            best_alignment = current_align
            torch.save(model.online_encoder.state_dict(), output_dir / "encoder_weights.pt")
            torch.save(model.state_dict(), output_dir / "byol_model_weights.pt")

        logger.info(
            "Epoch %d/%d | byol_loss=%.4f | train_align=%.4f | val_align=%.4f",
            epoch, epochs,
            train_m["byol_loss"],
            train_m["mean_bf_df_alignment"],
            val_align.get("mean_alignment", float("nan")),
        )

    final_stats = _eval_alignment(model, val_loader, device=device)
    with open(output_dir / "embedding_stats.json", "w") as f:
        json.dump(final_stats, f, indent=2)

    config_snapshot = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "stage": "stage2_byol_pretrain",
        "ssl_method": "cross_contrast_byol",
        "model": {"base_channels": args.base_channels, "projection_dim": args.projection_dim,
                  "ema_decay": args.ema_decay, "feature_dim": model.feature_dim},
        "training": {"epochs": epochs, "batch_size": batch_size, "img_size": img_size,
                     "lr": args.lr, "weight_decay": args.weight_decay, "device": device},
        "data": {"n_train_pairs": bundle.metadata["n_pairs"],
                 "n_val_pairs": val_bundle.metadata["n_pairs"]},
        "best_val_alignment": best_alignment,
        "outputs": {
            "encoder_weights": str(output_dir / "encoder_weights.pt"),
            "history_csv": str(history_path),
        },
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print("BYOL Pre-Training Summary")
    print(f"  device:          {device}")
    print(f"  train_pairs:     {bundle.metadata['n_pairs']}")
    print(f"  epochs:          {epochs}")
    print(f"  best_val_align:  {best_alignment:.4f}")
    print(f"  encoder_weights: {output_dir / 'encoder_weights.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
