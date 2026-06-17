#!/usr/bin/env python3
"""Cross-contrast fusion pulse-check (ideas #1 + #2): does learned early BF+DF fusion
beat single-contrast?  Cheap MIL proxy before committing to a 6-channel detector.

Runs a 3-way ablation on IDENTICAL paired FOVs (train mar2020 / test nov2021):
  - bf     : 3-channel brightfield MIL
  - df     : 3-channel darkfield MIL (polarity-inverted)
  - fusion : 6-channel BF + inverted-DF, MobileNet first conv INFLATED from pretrained
             RGB weights (idea #1), DF inverted for polarity reconciliation (idea #2)

Reference (BF-only, prior harness): mar->nov patient AUC ~0.71. If fusion clearly beats
the best single contrast here, the 6-channel detector build is justified.
"""

from __future__ import annotations

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
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from cross_study_pretrained_experiment import bootstrap_auc, _wilcoxon_auc
from train_yolov8_baseline import patient_key_from_filename
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything

BF = REPO / "data" / "yolo_mar_to_nov_bf"
DF = REPO / "data" / "yolo_mar_to_nov_df"
OUT = REPO / "results" / "mil_fusion_pulsecheck"; OUT.mkdir(parents=True, exist_ok=True)
SPLIT = pd.read_csv(REPO / "splits" / "mar_to_nov_split.csv")
PAT = dict(zip(SPLIT["patient_key"], (SPLIT["patient_label"] == "positive").astype(int)))
IMG, SEED, EPOCHS = 224, 42, 12
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
RESIZE = T.Resize((IMG, IMG))
TILE_RE = re.compile(r"^(.*)_tile_(\d+)$")

_img = pd.read_csv(REPO / "metadata" / "images.csv")
BF2DF = {}
for pk, g in _img.groupby("pair_key"):
    b, d = g[g.contrast == "brightfield"], g[g.contrast == "darkfield"]
    if len(b) and len(d):
        BF2DF[Path(b.image_name.iloc[0]).stem] = Path(d.image_name.iloc[0]).stem


def build_bags(split):
    lab = BF / "labels" / split
    bags = {}
    for t in sorted(glob.glob(str(BF / "images" / split / "*.jpg"))):
        stem = Path(t).stem
        m = TILE_RE.match(stem)
        bf_img, nn = m.group(1), m.group(2)
        if bf_img not in BF2DF:
            continue
        df_t = DF / "images" / split / f"{BF2DF[bf_img]}_tile_{nn}.jpg"
        if not df_t.exists():
            continue
        lf = lab / f"{stem}.txt"
        pos = lf.exists() and lf.stat().st_size > 0
        b = bags.setdefault(bf_img, {"bf": [], "df": [], "pos": 0,
                                     "pk": patient_key_from_filename(Path(t).name)})
        b["bf"].append(t); b["df"].append(str(df_t)); b["pos"] = max(b["pos"], int(pos))
    return [v for v in bags.values() if len(v["bf"]) == 30]


def load_bf(p):
    return NORM(T.functional.to_tensor(RESIZE(Image.open(p).convert("RGB"))))

def load_df(p):  # polarity reconciliation: invert so DF is dark-on-light like BF
    return NORM(T.functional.to_tensor(RESIZE(ImageOps.invert(Image.open(p).convert("RGB")))))


class BagDS(Dataset):
    def __init__(self, bags, mode):
        self.bags, self.mode = bags, mode

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, i):
        b = self.bags[i]
        tiles = []
        for j in range(30):
            if self.mode == "bf":
                tiles.append(load_bf(b["bf"][j]))
            elif self.mode == "df":
                tiles.append(load_df(b["df"][j]))
            else:  # fusion 6ch
                tiles.append(torch.cat([load_bf(b["bf"][j]), load_df(b["df"][j])], 0))
        return torch.stack(tiles), torch.tensor(float(b["pos"])), b["pk"]


class MIL(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
        if in_ch == 6:  # idea #1: inflate first conv from pretrained RGB weights
            conv = m.features[0][0]
            new = nn.Conv2d(6, conv.out_channels, 3, 2, 1, bias=False)
            with torch.no_grad():
                new.weight[:, :3] = conv.weight          # BF = pretrained RGB
                new.weight[:, 3:] = conv.weight * 0.5     # DF = inflated (halved for scale)
            m.features[0][0] = new
        self.backbone, self.pool, self.dim = m.features, nn.AdaptiveAvgPool2d(1), 1280
        self.attn = nn.Sequential(nn.Linear(self.dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.dim, 1))

    def forward(self, x):  # [B,N,C,H,W] -> [B] bag logits (attention-MIL)
        B, N = x.shape[:2]
        f = self.pool(self.backbone(x.reshape(B * N, *x.shape[2:]))).flatten(1).view(B, N, self.dim)
        a = torch.softmax(self.attn(f), dim=1)
        return self.head((a * f).sum(1)).squeeze(1)


@torch.no_grad()
def patient_auc(model, bags, mode, device, ci=False):
    loader = DataLoader(BagDS(bags, mode), batch_size=6, shuffle=False, num_workers=0)
    probs, pks = [], []
    model.eval()
    for imgs, _, pk in loader:
        probs.extend(torch.sigmoid(model(imgs.to(device))).cpu().numpy().tolist())
        pks.extend(pk)
    pat = pd.DataFrame({"pk": pks, "p": probs}).groupby("pk")["p"].max().reset_index()
    pat["t"] = pat["pk"].map(PAT); pat = pat.dropna()
    return (pat, bootstrap_auc(pat["t"].values, pat["p"].values)) if ci else (pat, _wilcoxon_auc(pat["t"].values, pat["p"].values))


def run_mode(mode, train_bags, val_bags, test_bags, device):
    seed_everything(SEED)
    in_ch = 6 if mode == "fusion" else 3
    model = MIL(in_ch).to(device)
    pos = sum(b["pos"] for b in train_bags); neg = len(train_bags) - pos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(neg / max(pos, 1)) ** 0.5]).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loader = DataLoader(BagDS(train_bags, mode), batch_size=4, shuffle=True, num_workers=0)
    best, best_state = -1.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        for imgs, lab, _ in loader:
            loss = crit(model(imgs.to(device)), lab.to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
        _, va = patient_auc(model, val_bags, mode, device)
        print(f"  [{mode}] epoch {ep}/{EPOCHS} mar_val={va:.4f}", flush=True)
        if np.isfinite(va) and va > best:
            best, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    _, (auc, lo, hi) = patient_auc(model, test_bags, mode, device, ci=True)
    return {"mode": mode, "best_val": round(best, 4), "test_auc": round(auc, 4),
            "lo": round(lo, 4), "hi": round(hi, 4)}


def main():
    device = resolve_device("auto")
    tr, va, te = build_bags("train"), build_bags("val"), build_bags("test")
    print(f"device={device}  paired bags  train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    res = [run_mode(m, tr, va, te, device) for m in ["bf", "df", "fusion"]]
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print("\n================ FUSION PULSE-CHECK (mar->nov patient AUC) ================", flush=True)
    for r in res:
        print(f"  {r['mode']:7s}: {r['test_auc']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]  (mar val {r['best_val']:.4f})")
    best_single = max(r["test_auc"] for r in res if r["mode"] in ("bf", "df"))
    fus = [r for r in res if r["mode"] == "fusion"][0]["test_auc"]
    print(f"  fusion - best_single = {fus - best_single:+.4f}  -> "
          f"{'PULSE: fusion helps' if fus - best_single > 0.02 else 'FLAT: fusion no better'}")


if __name__ == "__main__":
    main()
