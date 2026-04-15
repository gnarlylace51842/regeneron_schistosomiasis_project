#!/usr/bin/env python3
"""
Knowledge distillation: MobileNetV2 (DF teacher) → TinyConv (DF student).

Step 1: Train MobileNetV2 on DF images (strong teacher ~0.78+ AUC expected).
Step 2: Use teacher soft labels to train TinyConv with BYOL encoder weights.

Loss = α × BCE(student, hard_label) + (1-α) × KL(student_logits, teacher_logits / T)

Temperature T=4 softens the teacher distribution so the student learns from
probability ratios, not just the winning class. α=0.3 means 70% of the signal
comes from the teacher — appropriate when teacher is substantially better.

Key insight: unlike cross-modal pseudo-supervision (which failed because the
BF teacher at 0.692 was barely better than the DF student at 0.644),
MobileNetV2 with ImageNet pre-training should reach ~0.75-0.80 AUC on DF —
a meaningfully stronger signal that can actually teach.
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
try:
    from torchvision import models
except ImportError:
    print("ERROR: torchvision required. pip install torchvision", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.classification import load_single_contrast_data, MetadataImageDataset
from schisto_mobile_ai.models.simple_cnn import TinyConvClassifier
from schisto_mobile_ai.models.patient_aggregation import aggregate_patient_predictions
from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.logging import configure_logging
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything


BYOL_ENCODER = REPO_ROOT / "runs/ssl/pretrain_byol/20260408_105425_pretrain_byol_byol_pretrain_100ep/encoder_weights.pt"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-csv", type=Path, default=REPO_ROOT / "metadata/images.csv")
    p.add_argument("--split-csv", type=Path, default=REPO_ROOT / "splits/random_patient_split.csv")
    p.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw")
    p.add_argument("--teacher-epochs", type=int, default=25,
                   help="Epochs to train MobileNetV2 teacher (default 25)")
    p.add_argument("--student-epochs", type=int, default=30,
                   help="Epochs to train TinyConv student (default 30)")
    p.add_argument("--temperature", type=float, default=4.0,
                   help="Distillation temperature (default 4.0)")
    p.add_argument("--alpha", type=float, default=0.3,
                   help="Weight on hard BCE loss; (1-alpha) on KL distillation (default 0.3)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--teacher-lr", type=float, default=1e-4)
    p.add_argument("--student-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--contrast", type=str, default="df", choices=("bf", "df"),
                   help="Which contrast to distill (default: df)")
    p.add_argument("--teacher-output-dir", type=Path, default=None,
                   help="Teacher checkpoint dir (default: runs/baselines/mobilenet_v2_<contrast>)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Student output dir (default: runs/distillation/tinyconv_<contrast>_from_mobilenet)")
    p.add_argument("--device", type=str, default="cpu", choices=("auto", "cpu", "mps"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--smoke", dest="smoke_test", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-teacher", action="store_true",
                   help="Skip teacher training, load from --teacher-output-dir")
    return p


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _safe_auc(targets, probs) -> float:
    t, p = list(targets), list(probs)
    if len(set(t)) < 2:
        return float("nan")
    frame = pd.DataFrame({"t": t, "p": p})
    pos = frame["t"] >= 0.5
    pc, nc = int(pos.sum()), int((~pos).sum())
    if pc == 0 or nc == 0:
        return float("nan")
    ranks = frame["p"].rank(method="average")
    return float((ranks[pos].sum() - pc * (pc + 1) / 2.0) / (pc * nc))


def _patient_auc_max(preds_df: pd.DataFrame) -> float:
    pat = aggregate_patient_predictions(preds_df, patient_target_aggregation="max")
    return _safe_auc(pat["target"], pat["patient_probability_max"])


def _pos_weight(frame: pd.DataFrame, device: torch.device) -> torch.Tensor | None:
    pos = float(frame["target"].sum())
    neg = float(len(frame) - pos)
    return torch.tensor([(neg / pos) ** 0.5], dtype=torch.float32).to(device) if pos > 0 else None


# ── STEP 1: Train MobileNetV2 teacher on DF ────────────────────────────────────

def build_mobilenet_df(num_classes: int = 1) -> nn.Module:
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(m.last_channel, num_classes),
    )
    return m


def train_teacher(args, data, device, logger) -> Path:
    out = ensure_dir(args.teacher_output_dir)
    ckpt = out / "best_model.pt"

    if ckpt.exists() and not args.overwrite and not args.smoke_test:
        logger.info("Teacher checkpoint found at %s — skipping training", ckpt)
        return ckpt

    epochs = 2 if args.smoke_test else args.teacher_epochs
    bs     = min(args.batch_size, 4) if args.smoke_test else args.batch_size
    sz     = min(args.img_size, 64) if args.smoke_test else args.img_size

    train_ds = MetadataImageDataset(data.train_frame, image_size=sz, train=True)
    val_ds   = MetadataImageDataset(data.val_frame,   image_size=sz, train=False)
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=args.num_workers)
    vl = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=args.num_workers)

    model = build_mobilenet_df().to(device)

    # Prior bias
    pos_rate = float(data.train_frame["target"].sum()) / max(len(data.train_frame), 1)
    if 0 < pos_rate < 1:
        with torch.no_grad():
            model.classifier[-1].bias.fill_(float(np.log(pos_rate / (1 - pos_rate))))

    pw = _pos_weight(data.train_frame, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.teacher_lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_auc = float("-inf")
    history = []
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for batch in tl:
            imgs = batch["image"].to(device)
            tgts = batch["target"].float().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(imgs).squeeze(1), tgts)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Eval
        model.eval()
        rows = []
        with torch.no_grad():
            for batch in vl:
                imgs = batch["image"].to(device)
                probs = torch.sigmoid(model(imgs).squeeze(1)).cpu().numpy()
                for i, prob in enumerate(probs):
                    rows.append({"patient_key": batch["patient_key"][i],
                                 "target": float(batch["target"][i]),
                                 "probability": float(prob)})
        preds = pd.DataFrame(rows)
        auc = _patient_auc_max(preds)
        history.append({"epoch": epoch, "val_patient_auc": auc})
        logger.info("[Teacher] Epoch %d/%d | val_patient_auc=%.4f", epoch, epochs, auc)

        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), ckpt)

    pd.DataFrame(history).to_csv(out / "teacher_history.csv", index=False)
    logger.info("Teacher best AUC: %.4f", best_auc)
    with open(out / "teacher_results.json", "w") as f:
        json.dump({"best_val_patient_auc": best_auc}, f, indent=2)
    return ckpt


# ── STEP 2: Distill into TinyConv ─────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """Hard BCE + soft KL divergence distillation loss."""

    def __init__(self, *, temperature: float, alpha: float,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.T = temperature
        self.alpha = alpha          # weight on hard loss
        self.pos_weight = pos_weight

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # Hard BCE loss (standard supervised signal)
        hard_loss = F.binary_cross_entropy_with_logits(
            student_logits, targets,
            pos_weight=self.pos_weight,
            reduction="mean",
        )

        # Soft distillation: KL(student || teacher) at temperature T
        # For binary case: treat as 2-class [p, 1-p]
        s_log = F.logsigmoid(student_logits / self.T)          # log p(pos)
        s_log_neg = F.logsigmoid(-student_logits / self.T)     # log p(neg)
        t_soft_pos = torch.sigmoid(teacher_logits / self.T).detach()
        t_soft_neg = 1.0 - t_soft_pos

        # KL = sum_c [ t_c * (log t_c - log s_c) ]
        kl = t_soft_pos * (torch.log(t_soft_pos.clamp(1e-8)) - s_log) + \
             t_soft_neg * (torch.log(t_soft_neg.clamp(1e-8)) - s_log_neg)
        kl_loss = kl.mean() * (self.T ** 2)   # T^2 scaling from Hinton et al.

        total = self.alpha * hard_loss + (1 - self.alpha) * kl_loss
        return total, {"hard": float(hard_loss.detach()), "kl": float(kl_loss.detach())}


def train_student(args, data, teacher_ckpt: Path, device, logger) -> dict:
    out = ensure_dir(args.output_dir)

    epochs = 2 if args.smoke_test else args.student_epochs
    bs     = min(args.batch_size, 4) if args.smoke_test else args.batch_size
    sz     = min(args.img_size, 64) if args.smoke_test else args.img_size

    train_ds = MetadataImageDataset(data.train_frame, image_size=sz, train=True)
    val_ds   = MetadataImageDataset(data.val_frame,   image_size=sz, train=False)
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=args.num_workers)
    vl = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=args.num_workers)

    # ── Load teacher (frozen) ────────────────────────────────────────────────
    teacher = build_mobilenet_df().to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    logger.info("Teacher loaded from %s", teacher_ckpt)

    # ── Pre-compute teacher logits on training set (faster than inline) ─────
    logger.info("Pre-computing teacher logits on %d training images...", len(train_ds))
    teacher_logits_map: dict[str, float] = {}
    teacher.eval()
    with torch.no_grad():
        for batch in tl:
            imgs = batch["image"].to(device)
            logits = teacher(imgs).squeeze(1).cpu().numpy()
            for i, iid in enumerate(batch["image_id"]):
                teacher_logits_map[str(iid)] = float(logits[i])
    logger.info("Teacher logits computed for %d images", len(teacher_logits_map))

    # ── Student ──────────────────────────────────────────────────────────────
    student = TinyConvClassifier(base_channels=args.base_channels).to(device)
    if BYOL_ENCODER.exists():
        enc_state = torch.load(BYOL_ENCODER, map_location=device, weights_only=True)
        student.encoder.load_state_dict(enc_state)
        logger.info("Loaded BYOL encoder into student")

    # Prior bias
    pos_rate = float(data.train_frame["target"].sum()) / max(len(data.train_frame), 1)
    if 0 < pos_rate < 1:
        with torch.no_grad():
            student.head[-1].bias.fill_(float(np.log(pos_rate / (1 - pos_rate))))

    pw = _pos_weight(data.train_frame, device)
    criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha, pos_weight=pw)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.student_lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_auc = float("-inf")
    history = []

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        student.train()
        ep_loss = ep_hard = ep_kl = 0.0
        n_batches = 0
        for batch in tl:
            imgs    = batch["image"].to(device)
            targets = batch["target"].float().to(device)

            # Look up teacher logits for this batch
            t_logits = torch.tensor(
                [teacher_logits_map.get(str(iid), 0.0) for iid in batch["image_id"]],
                dtype=torch.float32, device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            s_logits = student(imgs).squeeze(1)
            loss, info = criterion(s_logits, t_logits, targets)
            loss.backward()
            optimizer.step()

            ep_loss += float(loss); ep_hard += info["hard"]; ep_kl += info["kl"]
            n_batches += 1
        scheduler.step()

        # ── Eval ──────────────────────────────────────────────────────────
        student.eval()
        rows = []
        with torch.no_grad():
            for batch in vl:
                imgs = batch["image"].to(device)
                probs = torch.sigmoid(student(imgs).squeeze(1)).cpu().numpy()
                for i, prob in enumerate(probs):
                    rows.append({
                        "image_id":    batch["image_id"][i],
                        "patient_key": batch["patient_key"][i],
                        "target":      float(batch["target"][i]),
                        "probability": float(prob),
                        "contrast":    batch["contrast"][i],
                        "split":       batch["split"][i],
                    })
        preds = pd.DataFrame(rows)
        auc = _patient_auc_max(preds)
        n = max(n_batches, 1)
        history.append({
            "epoch": epoch,
            "loss": ep_loss / n, "hard_loss": ep_hard / n, "kl_loss": ep_kl / n,
            "val_patient_auc": auc,
        })

        logger.info(
            "[Student] Epoch %d/%d | loss=%.4f hard=%.4f kl=%.4f | val_patient_auc=%.4f",
            epoch, epochs, ep_loss / n, ep_hard / n, ep_kl / n, auc,
        )

        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            preds.to_csv(out / "best_val_preds.csv", index=False)
            torch.save(student.state_dict(), out / "best_student_model.pt")

    pd.DataFrame(history).to_csv(out / "student_history.csv", index=False)
    results = {
        "teacher_auc": None,  # filled in main
        "student_best_auc": best_auc,
        "baseline_df_auc": 0.644,
        "gain": round(best_auc - 0.644, 4),
        "temperature": args.temperature,
        "alpha": args.alpha,
        "student_model": str(out / "best_student_model.pt"),
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    args = build_parser().parse_args()
    logger = configure_logging()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    # Resolve default dirs based on contrast
    if args.teacher_output_dir is None:
        args.teacher_output_dir = REPO_ROOT / f"runs/baselines/mobilenet_v2_{args.contrast}"
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / f"runs/distillation/tinyconv_{args.contrast}_from_mobilenet"

    # For BF, skip teacher training if existing BF MobileNetV2 checkpoint available
    bf_ckpt = REPO_ROOT / "runs/baselines/mobilenet_v2/best_model.pt"
    if args.contrast == "bf" and bf_ckpt.exists() and not args.overwrite:
        logger.info("Using existing MobileNetV2 BF teacher from %s", bf_ckpt)
        args.teacher_output_dir = REPO_ROOT / "runs/baselines/mobilenet_v2"
        args.skip_teacher = True

    data = load_single_contrast_data(
        images_csv=args.images_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        contrast=args.contrast,
        label_source="image",
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
    logger.info("%s data: %d train / %d val", args.contrast.upper(),
                len(data.train_frame), len(data.val_frame))

    # Step 1: Train or load teacher
    teacher_results_path = args.teacher_output_dir / "teacher_results.json"
    # For existing baselines, use metrics.json
    metrics_path = args.teacher_output_dir / "metrics.json"
    if args.skip_teacher and (args.teacher_output_dir / "best_model.pt").exists():
        teacher_ckpt = args.teacher_output_dir / "best_model.pt"
        logger.info("Skipping teacher training, using %s", teacher_ckpt)
        if teacher_results_path.exists():
            teacher_auc = json.load(open(teacher_results_path)).get("best_val_patient_auc", float("nan"))
        elif metrics_path.exists():
            teacher_auc = json.load(open(metrics_path)).get("best_val_patient_auc",
                          json.load(open(metrics_path)).get("patient_auc", float("nan")))
        else:
            teacher_auc = float("nan")
    else:
        logger.info("=== Step 1: Training MobileNetV2 teacher on %s ===", args.contrast.upper())
        teacher_ckpt = train_teacher(args, data, device, logger)
        teacher_auc = json.load(open(teacher_results_path))["best_val_patient_auc"] \
                      if teacher_results_path.exists() else float("nan")
    logger.info("Teacher AUC: %.4f", teacher_auc)

    # Step 2: Distill into TinyConv
    baseline_auc = 0.6920 if args.contrast == "bf" else 0.6444
    logger.info("=== Step 2: Distilling into TinyConv %s student ===", args.contrast.upper())
    results = train_student(args, data, teacher_ckpt, device, logger)
    results["teacher_auc"] = teacher_auc
    results["baseline_auc"] = baseline_auc
    results["gain"] = round(results["student_best_auc"] - baseline_auc, 4)

    print(f"\nKnowledge Distillation Summary ({args.contrast.upper()})")
    print(f"  Teacher (MobileNetV2 {args.contrast.upper()}):  AUC = {teacher_auc:.4f}")
    print(f"  Student (TinyConv {args.contrast.upper()}):     AUC = {results['student_best_auc']:.4f}  "
          f"(baseline: {baseline_auc:.4f})")
    print(f"  Gain over baseline:           {results['gain']:+.4f}")
    print(f"  T={args.temperature}, α={args.alpha}")
    print(f"  Student model:                {results['student_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
