#!/usr/bin/env python3
"""Cross-contrast pseudo-supervision: two modalities teach each other.

The key idea:
    A highly confident BF prediction on a well-aligned BF/DF pair is a
    reliable signal about whether eggs are present — independent of whether
    we have a human annotation for that image. We exploit this by using
    confident BF predictions as SOFT TEACHER LABELS for DF fine-tuning, and
    vice versa. This creates a self-reinforcing cross-modal training loop
    that effectively doubles the supervision signal without any additional
    annotations.

Selection criterion (both must hold to trust a pseudo-label):
    1. High BF classification confidence: |p_bf - 0.5| >= conf_threshold
       (the BF model is certain about its prediction)
    2. High BYOL cross-contrast alignment: cosine_sim(z_bf, z_df) >= align_threshold
       (BF and DF agree on the representation — the pair is consistent)

When BOTH conditions hold, the BF prediction is a reliable proxy for the
ground truth label, even for the DF image — and we use it as such.

Training procedure:
    Round 0: Train BF model with human labels (done — loaded from checkpoint)
              Train DF model with human labels (done — loaded from checkpoint)
    Round 1: Generate BF pseudo-labels for DF training
             Re-train DF model with: human_labels + weighted BF pseudo-labels
             Generate DF pseudo-labels for BF training
             Re-train BF model with: human_labels + weighted DF pseudo-labels
    (Optionally: Round 2 with updated models)

The pseudo-labels are blended with the human labels using a soft weight
(pseudo_label_weight < 1.0) to prevent over-reliance on noisy teacher signal.

Usage:
    python scripts/cross_modal_pseudo_supervision.py \\
        --bf-model runs/ssl/finetune/.../best_model.pt \\
        --df-model runs/ssl/finetune/.../best_model.pt \\
        --byol-weights runs/ssl/pretrain_byol/.../byol_model_weights.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.classification import (
    load_single_contrast_data, MetadataImageDataset, build_image_transform,
)
from schisto_mobile_ai.data.paired_classification import load_dual_contrast_data, PairedContrastDataset
from schisto_mobile_ai.models.byol_ssl import CrossContrastBYOL
from schisto_mobile_ai.models.simple_cnn import TinyConvClassifier
from schisto_mobile_ai.models.patient_aggregation import aggregate_patient_predictions
from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.logging import configure_logging
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf-model", type=Path, required=True)
    parser.add_argument("--df-model", type=Path, required=True)
    parser.add_argument("--byol-weights", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path,
                        default=REPO_ROOT / "metadata" / "pairs.csv")
    parser.add_argument("--patients-csv", type=Path,
                        default=REPO_ROOT / "metadata" / "patients.csv")
    parser.add_argument("--images-csv", type=Path,
                        default=REPO_ROOT / "metadata" / "images.csv")
    parser.add_argument("--split-csv", type=Path,
                        default=REPO_ROOT / "splits" / "random_patient_split.csv")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "runs" / "pseudo_supervised")
    parser.add_argument("--conf-threshold", type=float, default=0.35,
                        help="Min |p - 0.5| to trust a pseudo-label (0=all, 0.5=only certain).")
    parser.add_argument("--align-threshold", type=float, default=0.45,
                        help="Min BYOL cosine similarity to trust a pseudo-label.")
    parser.add_argument("--pseudo-weight", type=float, default=0.6,
                        help="Weight of pseudo-labels vs human labels in loss (0-1).")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Lower LR than initial training — we are refining, not starting fresh.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of pseudo-supervision rounds.")
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", dest="smoke_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _load_classifier(path: Path, *, base_channels: int, device: str) -> TinyConvClassifier:
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    key = "encoder.features.0.weight"
    if key in state:
        base_channels = int(state[key].shape[0])
    model = TinyConvClassifier(base_channels=base_channels)
    model.load_state_dict(state)
    return model.to(device).eval()


def _load_byol(path: Path, *, device: str) -> CrossContrastBYOL:
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    bc = 32
    key = "online_encoder.features.0.weight"
    if key in state:
        bc = int(state[key].shape[0])
    model = CrossContrastBYOL(base_channels=bc)
    model.load_state_dict(state)
    return model.to(device).eval()


def _safe_auc(targets: list[float], probs: list[float]) -> float:
    if len(set(targets)) < 2:
        return float("nan")
    frame = pd.DataFrame({"t": targets, "p": probs})
    pos = frame["t"] >= 0.5
    pc, nc = int(pos.sum()), int((~pos).sum())
    if pc == 0 or nc == 0:
        return float("nan")
    ranks = frame["p"].rank(method="average")
    return float((ranks[pos].sum() - pc * (pc + 1) / 2.0) / (pc * nc))


@torch.no_grad()
def generate_pseudo_labels(
    teacher_bf: TinyConvClassifier,
    teacher_df: TinyConvClassifier,
    byol: CrossContrastBYOL,
    paired_loader: DataLoader,
    *,
    device: str,
    conf_threshold: float,
    align_threshold: float,
) -> pd.DataFrame:
    """Generate pseudo-labels for all training pairs.

    Returns DataFrame with columns:
        pair_key, patient_key, target (human),
        p_bf, p_df, alignment,
        bf_pseudo_for_df, df_pseudo_for_bf,
        bf_trusted, df_trusted  (bool — whether pseudo-label passes both gates)
    """
    import torch.nn.functional as F
    rows = []
    for batch in paired_loader:
        bf_imgs = batch["brightfield_image"].to(device)
        df_imgs = batch["darkfield_image"].to(device)

        p_bf = torch.sigmoid(teacher_bf(bf_imgs).squeeze(1)).cpu().numpy()
        p_df = torch.sigmoid(teacher_df(df_imgs).squeeze(1)).cpu().numpy()

        z_bf = F.normalize(byol.online_encoder(bf_imgs), dim=1)
        z_df = F.normalize(byol.online_encoder(df_imgs), dim=1)
        alignment = (z_bf * z_df).sum(dim=1).cpu().numpy()

        for i in range(len(p_bf)):
            conf_bf = float(abs(p_bf[i] - 0.5))
            conf_df = float(abs(p_df[i] - 0.5))
            align = float(alignment[i])
            # BF pseudo-label trusted for DF training
            bf_trusted = conf_bf >= conf_threshold and align >= align_threshold
            # DF pseudo-label trusted for BF training
            df_trusted = conf_df >= conf_threshold and align >= align_threshold
            rows.append({
                "pair_key": batch["pair_key"][i],
                "patient_key": batch["patient_key"][i],
                "target": float(batch["target"][i].item()),
                "p_bf": float(p_bf[i]),
                "p_df": float(p_df[i]),
                "alignment": align,
                "bf_confidence": conf_bf,
                "df_confidence": conf_df,
                "bf_pseudo_for_df": float(p_bf[i]),  # BF pred → DF pseudo-label
                "df_pseudo_for_bf": float(p_df[i]),  # DF pred → BF pseudo-label
                "bf_trusted": bf_trusted,
                "df_trusted": df_trusted,
            })
    return pd.DataFrame(rows)


class PseudoLabelDataset(Dataset):
    """Single-contrast dataset that blends human labels with cross-modal pseudo-labels.

    For samples where the teacher's pseudo-label is trusted, the training target
    is a convex combination:
        target = (1 - pseudo_weight) * human_label + pseudo_weight * pseudo_label

    For untrusted samples, target = human_label (pseudo-label ignored).
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        pseudo_df: pd.DataFrame,
        *,
        pseudo_col: str,
        trusted_col: str,
        pseudo_weight: float,
        image_size: int,
    ) -> None:
        merged = frame.merge(
            pseudo_df[["pair_key", pseudo_col, trusted_col]],
            on="pair_key", how="left",
        )
        merged[pseudo_col] = merged[pseudo_col].fillna(merged["target"])
        merged[trusted_col] = merged[trusted_col].fillna(False)
        # Soft target: blend human label with pseudo-label
        merged["soft_target"] = np.where(
            merged[trusted_col],
            (1.0 - pseudo_weight) * merged["target"] + pseudo_weight * merged[pseudo_col],
            merged["target"].astype(float),
        )
        self.frame = merged.reset_index(drop=True)
        self.transform = build_image_transform(image_size=image_size, train=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from pathlib import Path
        from PIL import Image, ImageOps
        row = self.frame.iloc[index]
        path = Path(row["image_path"])
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
        return {
            "image": self.transform(img),
            "target": torch.tensor(float(row["soft_target"]), dtype=torch.float32),
            "hard_target": torch.tensor(float(row["target"]), dtype=torch.float32),
            "image_id": str(row.get("image_id", "")),
            "patient_key": str(row["patient_key"]),
            "contrast": str(row.get("contrast", "")),
            "split": str(row.get("split", "train")),
            "pair_key": str(row["pair_key"]),
        }


def _train_with_pseudo(
    model: TinyConvClassifier,
    train_ds: Dataset,
    val_ds: MetadataImageDataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    pos_weight: float | None,
    logger: Any,
    tag: str,
) -> tuple[TinyConvClassifier, float, pd.DataFrame]:
    pw = torch.tensor([pos_weight], dtype=torch.float32).to(device) if pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    best_auc = float("-inf")
    best_preds = pd.DataFrame()
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            imgs = batch["image"].to(device)
            targets = batch["target"].to(device)  # soft targets
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(imgs).squeeze(1), targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Eval on val with hard labels
        model.eval()
        rows = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                logits = model(imgs).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
                for i in range(len(probs)):
                    rows.append({
                        "image_id": batch["image_id"][i],
                        "patient_key": batch["patient_key"][i],
                        "target": float(batch["target"][i].item()),
                        "probability": float(probs[i]),
                        "contrast": batch["contrast"][i],
                        "split": "val",
                    })
        preds = pd.DataFrame(rows)
        patient_frame = aggregate_patient_predictions(preds, patient_target_aggregation="max")
        auc = _safe_auc(
            patient_frame["target"].tolist(),
            patient_frame["patient_probability_max"].tolist(),
        )
        logger.info("[%s] Epoch %d/%d | val_patient_auc=%.4f", tag, epoch, epochs, auc)

        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_preds = preds.copy()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc, best_preds


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    epochs = min(args.epochs, 2) if args.smoke_test else args.epochs
    batch_size = min(args.batch_size, 8) if args.smoke_test else args.batch_size
    img_size = min(args.img_size, 128) if args.smoke_test else args.img_size

    out_dir = ensure_dir(args.output_dir)

    # Load paired data for pseudo-label generation
    paired_bundle = load_dual_contrast_data(
        pairs_csv=args.pairs_csv,
        patients_csv=args.patients_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        label_source="image",
        smoke_test=args.smoke_test,
        seed=args.seed,
    )

    # Load single-contrast data for BF and DF fine-tuning
    bf_data = load_single_contrast_data(
        images_csv=args.images_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        contrast="bf",
        label_source="image",
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
    df_data = load_single_contrast_data(
        images_csv=args.images_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        contrast="df",
        label_source="image",
        smoke_test=args.smoke_test,
        seed=args.seed,
    )

    # Add pair_key to single-contrast frames by joining on image path
    # pair_key is used to join pseudo-labels
    pairs_df = paired_bundle.train_frame[["pair_key", "brightfield_path", "darkfield_path",
                                          "patient_key"]].copy()

    bf_train = bf_data.train_frame.copy()
    df_train = df_data.train_frame.copy()
    # Join pair_key via path matching
    bf_train = bf_train.merge(
        pairs_df[["pair_key", "brightfield_path"]].rename(columns={"brightfield_path": "image_path"}),
        on="image_path", how="left",
    )
    df_train = df_train.merge(
        pairs_df[["pair_key", "darkfield_path"]].rename(columns={"darkfield_path": "image_path"}),
        on="image_path", how="left",
    )

    # Load models
    bf_model = _load_classifier(args.bf_model, base_channels=args.base_channels, device=device)
    df_model = _load_classifier(args.df_model, base_channels=args.base_channels, device=device)
    byol = _load_byol(args.byol_weights, device=device)

    # Pos weight
    pos_bf = float(bf_train["target"].sum())
    neg_bf = float(len(bf_train) - pos_bf)
    pos_df = float(df_train["target"].sum())
    neg_df = float(len(df_train) - pos_df)
    pw_bf = (neg_bf / pos_bf) ** 0.5 if pos_bf > 0 else None
    pw_df = (neg_df / pos_df) ** 0.5 if pos_df > 0 else None

    val_bf_ds = MetadataImageDataset(bf_data.val_frame, image_size=img_size, train=False)
    val_df_ds = MetadataImageDataset(df_data.val_frame, image_size=img_size, train=False)

    results: list[dict] = []

    for round_idx in range(1, args.rounds + 1):
        logger.info("=== Pseudo-supervision round %d/%d ===", round_idx, args.rounds)

        # Generate pseudo-labels from current teacher models
        paired_ds = PairedContrastDataset(paired_bundle.train_frame,
                                          image_size=img_size, train=False)
        paired_loader = DataLoader(paired_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        pseudo_df = generate_pseudo_labels(
            bf_model, df_model, byol, paired_loader,
            device=device,
            conf_threshold=args.conf_threshold,
            align_threshold=args.align_threshold,
        )
        pseudo_path = out_dir / f"pseudo_labels_round{round_idx}.csv"
        pseudo_df.to_csv(pseudo_path, index=False)

        trusted_bf = int(pseudo_df["bf_trusted"].sum())
        trusted_df = int(pseudo_df["df_trusted"].sum())
        total = len(pseudo_df)
        logger.info("Pseudo-labels: BF trusted=%d/%d (%.0f%%), DF trusted=%d/%d (%.0f%%)",
                    trusted_bf, total, 100 * trusted_bf / max(total, 1),
                    trusted_df, total, 100 * trusted_df / max(total, 1))

        # Re-train DF model with BF pseudo-labels
        logger.info("Re-training DF model with BF pseudo-labels...")
        df_pseudo_ds = PseudoLabelDataset(
            df_train, pseudo_df,
            pseudo_col="bf_pseudo_for_df",
            trusted_col="bf_trusted",
            pseudo_weight=args.pseudo_weight,
            image_size=img_size,
        )
        df_model, df_auc, df_preds = _train_with_pseudo(
            df_model, df_pseudo_ds, val_df_ds,
            epochs=epochs, batch_size=batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            device=device, pos_weight=pw_df,
            logger=logger, tag=f"DF-r{round_idx}",
        )
        torch.save(df_model.state_dict(), out_dir / f"df_model_round{round_idx}.pt")
        logger.info("DF model after pseudo-supervision: patient AUC=%.4f", df_auc)

        # Re-train BF model with DF pseudo-labels
        logger.info("Re-training BF model with DF pseudo-labels...")
        bf_pseudo_ds = PseudoLabelDataset(
            bf_train, pseudo_df,
            pseudo_col="df_pseudo_for_bf",
            trusted_col="df_trusted",
            pseudo_weight=args.pseudo_weight,
            image_size=img_size,
        )
        bf_model, bf_auc, bf_preds = _train_with_pseudo(
            bf_model, bf_pseudo_ds, val_bf_ds,
            epochs=epochs, batch_size=batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            device=device, pos_weight=pw_bf,
            logger=logger, tag=f"BF-r{round_idx}",
        )
        torch.save(bf_model.state_dict(), out_dir / f"bf_model_round{round_idx}.pt")
        logger.info("BF model after pseudo-supervision: patient AUC=%.4f", bf_auc)

        results.append({
            "round": round_idx,
            "n_bf_trusted": trusted_bf,
            "n_df_trusted": trusted_df,
            "bf_patient_auc": round(bf_auc, 4),
            "df_patient_auc": round(df_auc, 4),
        })
        bf_preds.to_csv(out_dir / f"bf_val_preds_round{round_idx}.csv", index=False)
        df_preds.to_csv(out_dir / f"df_val_preds_round{round_idx}.csv", index=False)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nCross-Modal Pseudo-Supervision Results")
    for r in results:
        print(f"  Round {r['round']}: BF AUC={r['bf_patient_auc']:.4f}  "
              f"DF AUC={r['df_patient_auc']:.4f}  "
              f"(BF pseudo trusted: {r['n_bf_trusted']}, DF pseudo trusted: {r['n_df_trusted']})")
    print(f"  BF model: {out_dir / f'bf_model_round{args.rounds}.pt'}")
    print(f"  DF model: {out_dir / f'df_model_round{args.rounds}.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
