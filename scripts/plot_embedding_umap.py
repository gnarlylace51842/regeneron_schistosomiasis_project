"""
UMAP visualization of BYOL encoder representations.

Two plots side-by-side:
  Left:  BF (blue) vs DF (orange) — cross-modal alignment
  Right: Negative (grey) vs Positive (red) — class separation

If cross-contrast BYOL pre-training worked, BF and DF embeddings should be
interleaved, showing the encoder learned a shared representation space.
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap  # type: ignore
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "src")
from schisto_mobile_ai.models.byol_ssl import TinyConvEncoder, CrossContrastBYOL
from schisto_mobile_ai.data.classification import SimpleImageTransform
from schisto_mobile_ai.data import load_dual_contrast_data

RESULTS  = pathlib.Path("results")
RAW_DIR  = pathlib.Path("data/raw")
BASE_CHANNELS = 32
IMAGE_SIZE    = 224
DEVICE = torch.device("cpu")

ENCODER_WEIGHTS = pathlib.Path(
    "runs/ssl/pretrain_byol/20260408_105425_pretrain_byol_byol_pretrain_100ep"
    "/encoder_weights.pt"
)
BYOL_WEIGHTS = pathlib.Path(
    "runs/ssl/pretrain_byol/20260408_105425_pretrain_byol_byol_pretrain_100ep"
    "/byol_model_weights.pt"
)


# ── Load encoder ─────────────────────────────────────────────────────────────

encoder = TinyConvEncoder(base_channels=BASE_CHANNELS)
if ENCODER_WEIGHTS.exists():
    state = torch.load(ENCODER_WEIGHTS, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state)
    print(f"Loaded encoder from {ENCODER_WEIGHTS}")
else:
    byol = CrossContrastBYOL(base_channels=BASE_CHANNELS)
    state = torch.load(BYOL_WEIGHTS, map_location="cpu", weights_only=True)
    byol.load_state_dict(state)
    encoder = byol.online_encoder
    print(f"Extracted encoder from BYOL model")
encoder.eval().to(DEVICE)


# ── Load val split ────────────────────────────────────────────────────────────

bundle = load_dual_contrast_data(
    pairs_csv="metadata/pairs.csv",
    patients_csv="metadata/patients.csv",
    split_csv="splits/random_patient_split.csv",
    raw_dir=str(RAW_DIR),
    label_source="image",
)
val_frame = bundle.val_frame
print(f"Val frame: {len(val_frame)} pairs, columns: {list(val_frame.columns[:6])}")


# ── Dataset that returns both BF and DF images ────────────────────────────────

class PairEmbedDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        # Use the same column names as PairedContrastDataset
        bf_path = pathlib.Path(row["brightfield_path"])
        df_path = pathlib.Path(row["darkfield_path"])
        label   = int(row.get("label", row.get("target", 0)) in {1, "positive", "1", True})

        def load(p):
            img = Image.open(p).convert("RGB")
            return self.transform(img)

        return {
            "bf": load(bf_path),
            "df": load(df_path),
            "label": label,
            "pair_key": row.get("pair_key", str(idx)),
        }


transform = SimpleImageTransform(image_size=IMAGE_SIZE, train=False)

# Find the right column names
bf_col = "brightfield_path" if "brightfield_path" in val_frame.columns else None
if bf_col is None:
    # construct paths from relative path columns
    if "brightfield_relative_path" in val_frame.columns:
        val_frame = val_frame.copy()
        val_frame["brightfield_path"] = val_frame["brightfield_relative_path"].apply(
            lambda p: str(RAW_DIR / p)
        )
        val_frame["darkfield_path"] = val_frame["darkfield_relative_path"].apply(
            lambda p: str(RAW_DIR / p)
        )
    else:
        raise ValueError(f"Cannot find image path columns. Cols: {list(val_frame.columns)}")

# Normalize label column
if "label" not in val_frame.columns and "patient_level_label" in val_frame.columns:
    val_frame = val_frame.copy()
    val_frame["label"] = (val_frame["patient_level_label"] == "positive").astype(int)
elif "label" in val_frame.columns:
    val_frame = val_frame.copy()
    val_frame["label"] = val_frame["label"].apply(
        lambda x: 1 if str(x).lower() in {"positive", "1", "true", "yes"} else 0
    )

ds = PairEmbedDataset(val_frame, transform)
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

print(f"Extracting embeddings for {len(ds)} pairs...")
bf_embeds, df_embeds, all_labels = [], [], []

with torch.no_grad():
    for batch in loader:
        bf_e = encoder(batch["bf"].to(DEVICE)).cpu().numpy()
        df_e = encoder(batch["df"].to(DEVICE)).cpu().numpy()
        bf_embeds.append(bf_e)
        df_embeds.append(df_e)
        all_labels.extend(batch["label"].tolist())

bf_embeds  = np.concatenate(bf_embeds, axis=0)
df_embeds  = np.concatenate(df_embeds, axis=0)
all_labels = np.array(all_labels)

all_embeds = np.concatenate([bf_embeds, df_embeds], axis=0)
modalities = np.array(["BF"] * len(bf_embeds) + ["DF"] * len(df_embeds))
labels_all = np.concatenate([all_labels, all_labels])

print(f"Total embeddings: {all_embeds.shape}  positives: {labels_all.sum()}")


# ── UMAP ─────────────────────────────────────────────────────────────────────

print("Fitting UMAP (this takes ~30s)...")
reducer = umap.UMAP(n_neighbors=20, min_dist=0.2, random_state=42, n_jobs=1)
emb2d = reducer.fit_transform(all_embeds)
print("UMAP done.")


# ── Figures ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("BYOL Cross-Contrast Encoder Representations (Val Set)",
             fontsize=13, fontweight="bold")

# Left: modality
ax = axes[0]
bf_m = modalities == "BF"
df_m = modalities == "DF"
ax.scatter(emb2d[df_m, 0], emb2d[df_m, 1],
           c="#e07b39", s=14, alpha=0.55, linewidths=0, label=f"DF  (n={df_m.sum()})")
ax.scatter(emb2d[bf_m, 0], emb2d[bf_m, 1],
           c="#4a90d9", s=14, alpha=0.55, linewidths=0, label=f"BF  (n={bf_m.sum()})")
ax.set_title("Modality: BF vs DF\n(interleaved → cross-modal alignment)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9, markerscale=2)
ax.set_xlabel("UMAP 1", fontsize=9)
ax.set_ylabel("UMAP 2", fontsize=9)
ax.grid(True, alpha=0.2)

# Right: label
ax = axes[1]
neg_m = labels_all == 0
pos_m = labels_all == 1
ax.scatter(emb2d[neg_m, 0], emb2d[neg_m, 1],
           c="#cccccc", s=14, alpha=0.5, linewidths=0, label=f"Negative (n={neg_m.sum()})")
ax.scatter(emb2d[pos_m, 0], emb2d[pos_m, 1],
           c="#c0392b", s=18, alpha=0.8, linewidths=0, label=f"Positive (n={pos_m.sum()})")
ax.set_title("Label: Negative vs Positive\n(separated → class discriminability)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9, markerscale=2)
ax.set_xlabel("UMAP 1", fontsize=9)
ax.set_ylabel("UMAP 2", fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
out = RESULTS / "byol_embedding_umap.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved UMAP figure to {out}")
