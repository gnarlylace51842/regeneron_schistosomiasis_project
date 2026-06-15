#!/usr/bin/env python3
"""Option A pulse-check: does forcing the classifier LOCAL recover cross-study robustness?

Controlled comparison against the whole-image MobileNet baseline:
  - SAME backbone (MobileNetV2, ImageNet)
  - SAME supervision (image/FOV-level labels only -- NO box coordinates)
  - SAME patient aggregation (max)
  - ONLY difference: the model sees local 640px tiles via max-pooling MIL, instead of
    the whole 4032x3024 image resized to 224.

Train mar2020, zero-shot test nov2021 (same 380 patients as the detector).
Baseline (whole-image MobileNet, cross-study): patient AUC 0.517 [0.432, 0.600].
If this lifts toward the detector's 0.914, locality is the fix and the method has a pulse.
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cross_study_pretrained_experiment import build_model, bootstrap_auc, _wilcoxon_auc
from train_yolov8_baseline import patient_key_from_filename
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything

DATA = REPO / "data" / "yolo_mar_to_nov_bf"
OUT = REPO / "results" / "mil_locality_pulsecheck"
OUT.mkdir(parents=True, exist_ok=True)

SPLIT = pd.read_csv(REPO / "splits" / "mar_to_nov_split.csv")
PAT_LABEL = dict(zip(SPLIT["patient_key"], (SPLIT["patient_label"] == "positive").astype(int)))

IMG = 224
SEED = 42
EPOCHS = 12
BAGS_PER_BATCH = 4
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF_TRAIN = T.Compose([T.Resize((IMG, IMG)), T.RandomHorizontalFlip(), T.ToTensor(), NORM])
TF_EVAL = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), NORM])
FOV_RE = re.compile(r"^(.*)_tile_\d+$")


def fov_of(stem: str) -> str:
    m = FOV_RE.match(stem)
    return m.group(1) if m else stem


def build_bags(split: str) -> list[dict]:
    """FOV bags (30 tiles) with image-level label = does the FOV contain any egg."""
    img_dir, lab_dir = DATA / "images" / split, DATA / "labels" / split
    bags: dict[str, dict] = {}
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
        return imgs, torch.tensor(float(b["pos"]))


class TileDS(Dataset):
    def __init__(self, split, tf):
        self.tiles = sorted(glob.glob(str(DATA / "images" / split / "*.jpg")))
        self.tf = tf

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        t = self.tiles[i]
        return self.tf(Image.open(t).convert("RGB")), patient_key_from_filename(Path(t).name)


@torch.no_grad()
def infer_patient_auc(model, split, device, with_ci=False):
    """Per-tile inference -> max prob per patient -> patient-level AUC."""
    loader = DataLoader(TileDS(split, TF_EVAL), batch_size=64, shuffle=False, num_workers=0)
    probs, pks = [], []
    model.eval()
    for x, pk in loader:
        p = torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy()
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
    seed_everything(SEED)
    device = resolve_device("auto")
    print(f"device={device}", flush=True)

    train_bags = build_bags("train")
    n_pos = sum(b["pos"] for b in train_bags)
    print(f"train bags (FOVs): {len(train_bags)}  positive: {n_pos} "
          f"({100*n_pos/max(len(train_bags),1):.1f}%)", flush=True)

    model = build_model("mobilenet_v2").to(device)
    pos = float(n_pos); neg = float(len(train_bags) - n_pos)
    pw = torch.tensor([(neg / max(pos, 1)) ** 0.5], dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loader = DataLoader(BagDS(train_bags, TF_TRAIN), batch_size=BAGS_PER_BATCH,
                        shuffle=True, num_workers=0)

    best_val, best_state = -1.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        for imgs, labels in loader:
            B, N = imgs.shape[:2]
            x = imgs.view(B * N, 3, IMG, IMG).to(device)
            logits = model(x).view(B, N)            # per-tile logits
            bag_logit = logits.max(dim=1).values     # max-MIL pooling
            loss = crit(bag_logit, labels.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        _, val_auc = infer_patient_auc(model, "val", device)
        print(f"epoch {ep}/{EPOCHS}  mar_val_patient_auc={val_auc:.4f}", flush=True)
        if np.isfinite(val_auc) and val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    pat, (auc, lo, hi) = infer_patient_auc(model, "test", device, with_ci=True)
    pat.to_csv(OUT / "test_patient_scores.csv", index=False)

    n_p = int(pat["t"].sum()); n_n = int(len(pat) - n_p)
    print("\n================ OPTION A PULSE-CHECK ================", flush=True)
    print(f"  Local MIL classifier (image labels only), cross-study mar->nov:")
    print(f"    patient AUC = {auc:.4f}  [{lo:.4f}, {hi:.4f}]   (n={len(pat)}, pos={n_p}, neg={n_n})")
    print(f"    best mar val AUC during training: {best_val:.4f}")
    print(f"  Baselines on the SAME shift/patients:")
    print(f"    whole-image MobileNet (image labels):   0.517 [0.432, 0.600]")
    print(f"    YOLOv8 detector (box labels):           0.914 [0.859, 0.961]")
    verdict = "PULSE — locality lifts it off chance" if auc > 0.62 else "FLAT — locality alone is not enough"
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
