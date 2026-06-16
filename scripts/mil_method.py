#!/usr/bin/env python3
"""Locality-constrained weakly-supervised classifier (MIL) -- method harness + ablation tool.

Same controlled setup as the pulse-check (MobileNetV2, 640px tiles, image-level labels
only, train mar2020 / zero-shot test nov2021, patient = max over bags), with a swappable
aggregator so this doubles as the aggregation ablation.

    python scripts/mil_method.py --agg attention   # ceiling-check
    python scripts/mil_method.py --agg max          # reproduces the pulse-check
    python scripts/mil_method.py --agg mean

Reference on the same shift/patients:
    whole-image MobileNet (image labels):  0.517   |  max-MIL local (image labels): 0.711
    YOLOv8 detector (box labels):          0.914
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cross_study_pretrained_experiment import bootstrap_auc, _wilcoxon_auc
from train_yolov8_baseline import patient_key_from_filename
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything

DATA = REPO / "data" / "yolo_mar_to_nov_bf"
OUT = REPO / "results" / "mil_locality_pulsecheck"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = pd.read_csv(REPO / "splits" / "mar_to_nov_split.csv")
PAT_LABEL = dict(zip(SPLIT["patient_key"], (SPLIT["patient_label"] == "positive").astype(int)))
SEED = 42
FOV_RE = re.compile(r"^(.*)_tile_\d+$")
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def fov_of(stem):
    m = FOV_RE.match(stem)
    return m.group(1) if m else stem


def build_bags(split):
    img_dir, lab_dir = DATA / "images" / split, DATA / "labels" / split
    bags = {}
    for t in sorted(glob.glob(str(img_dir / "*.jpg"))):
        stem = Path(t).stem
        fov = fov_of(stem)
        lab = lab_dir / f"{stem}.txt"
        pos = lab.exists() and lab.stat().st_size > 0
        b = bags.setdefault(fov, {"tiles": [], "pos": 0, "pk": patient_key_from_filename(Path(t).name)})
        b["tiles"].append(t)
        b["pos"] = max(b["pos"], int(pos))
    return [v for v in bags.values() if len(v["tiles"]) == 30]


class BagDS(Dataset):
    def __init__(self, bags, tf):
        self.bags, self.tf = bags, tf

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, i):
        b = self.bags[i]
        imgs = torch.stack([self.tf(Image.open(t).convert("RGB")) for t in b["tiles"]])
        return imgs, torch.tensor(float(b["pos"])), b["pk"]


class MILModel(nn.Module):
    def __init__(self, agg="attention"):
        super().__init__()
        m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dim = 1280
        self.agg = agg
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.dim, 1))
        if agg == "attention":
            self.attn = nn.Sequential(nn.Linear(self.dim, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x):                              # x: [B,N,3,H,W]
        B, N = x.shape[:2]
        f = self.pool(self.backbone(x.reshape(B * N, *x.shape[2:]))).flatten(1).view(B, N, self.dim)
        if self.agg == "attention":
            a = torch.softmax(self.attn(f), dim=1)     # [B,N,1]
            bag = (a * f).sum(1)                        # [B,dim]
            return self.head(bag).squeeze(1)           # [B]
        logits = self.head(f).squeeze(-1)              # [B,N]
        return logits.max(1).values if self.agg == "max" else logits.mean(1)


@torch.no_grad()
def patient_eval(model, bags, device, tf, with_ci=False):
    loader = DataLoader(BagDS(bags, tf), batch_size=8, shuffle=False, num_workers=0)
    probs, pks = [], []
    model.eval()
    for imgs, _, pk in loader:
        p = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
        probs.extend(p.tolist())
        pks.extend(pk)
    df = pd.DataFrame({"pk": pks, "p": probs})
    pat = df.groupby("pk")["p"].max().reset_index()
    pat["t"] = pat["pk"].map(PAT_LABEL)
    pat = pat.dropna()
    if with_ci:
        return pat, bootstrap_auc(pat["t"].values, pat["p"].values)
    return pat, _wilcoxon_auc(pat["t"].values, pat["p"].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", choices=["max", "mean", "attention"], default="attention")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--bags-per-batch", type=int, default=4)
    args = ap.parse_args()

    seed_everything(SEED)
    device = resolve_device("auto")
    IMG = args.img_size
    tf_train = T.Compose([T.Resize((IMG, IMG)), T.RandomHorizontalFlip(), T.ToTensor(), NORM])
    tf_eval = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), NORM])
    print(f"device={device}  agg={args.agg}  img={IMG}  epochs={args.epochs}", flush=True)

    train_bags, val_bags, test_bags = build_bags("train"), build_bags("val"), build_bags("test")
    n_pos = sum(b["pos"] for b in train_bags)
    print(f"train bags {len(train_bags)} ({n_pos} pos) | val {len(val_bags)} | test {len(test_bags)}", flush=True)

    model = MILModel(args.agg).to(device)
    pos, neg = float(n_pos), float(len(train_bags) - n_pos)
    pw = torch.tensor([(neg / max(pos, 1)) ** 0.5], dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(BagDS(train_bags, tf_train), batch_size=args.bags_per_batch,
                        shuffle=True, num_workers=0)

    best_val, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        for imgs, labels, _ in loader:
            logit = model(imgs.to(device))
            loss = crit(logit, labels.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        _, val_auc = patient_eval(model, val_bags, device, tf_eval)
        print(f"epoch {ep}/{args.epochs}  mar_val_patient_auc={val_auc:.4f}", flush=True)
        if np.isfinite(val_auc) and val_auc > best_val:
            best_val, best_state = val_auc, {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    pat, (auc, lo, hi) = patient_eval(model, test_bags, device, tf_eval, with_ci=True)
    pat.to_csv(OUT / f"test_patient_scores_{args.agg}.csv", index=False)
    (OUT / f"summary_{args.agg}.json").write_text(json.dumps(
        {"agg": args.agg, "img": IMG, "epochs": args.epochs, "best_val_auc": round(best_val, 4),
         "test_auc": round(auc, 4), "test_lo": round(lo, 4), "test_hi": round(hi, 4)}, indent=2))

    print(f"\n========== MIL method ({args.agg}) ==========", flush=True)
    print(f"  cross-study mar->nov patient AUC = {auc:.4f} [{lo:.4f}, {hi:.4f}]  (best mar val {best_val:.4f})")
    print(f"  reference:  whole-image 0.517 | max-MIL 0.711 | detector 0.914")


if __name__ == "__main__":
    main()
