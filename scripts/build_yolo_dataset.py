#!/usr/bin/env python3
"""Build a YOLOv8-format dataset from the schistosomiasis tiles.

Converts 4032×3024 images into 30 tiles of 640×640 each, with YOLO-format
labels. Eggs are annotated as center points; we construct fixed-size 150×150
bounding boxes around each center.

Usage:
    python scripts/build_yolo_dataset.py --split mar_to_nov --contrast brightfield
    python scripts/build_yolo_dataset.py --split mar_to_nov --contrast darkfield
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR   = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.utils.io import ensure_dir

IMAGES_CSV = REPO_ROOT / "metadata" / "images.csv"
RAW_DIR    = REPO_ROOT / "data"     / "raw"
ANN_MAR    = RAW_DIR / "mar2020" / "mar2020_docs" / "mar2020_annotations_08262024.csv"
ANN_NOV    = RAW_DIR / "nov2021" / "nov2021_docs" / "nov2021_annotations_08262024.csv"

TILE_SIZE = 640
TILE_COLS = 6
TILE_ROWS = 5
IMG_W, IMG_H = 4032, 3024
BBOX_SIZE = 150  # px, fixed bounding box size around each egg center

COL_STARTS = np.linspace(0, IMG_W - TILE_SIZE, TILE_COLS, dtype=int).tolist()
ROW_STARTS = np.linspace(0, IMG_H - TILE_SIZE, TILE_ROWS, dtype=int).tolist()


def _tile_positions() -> list[tuple[int, int]]:
    return [(c, r) for r in ROW_STARTS for c in COL_STARTS]


def _load_annotations() -> pd.DataFrame:
    parts = [pd.read_csv(ANN_MAR), pd.read_csv(ANN_NOV)]
    ann = pd.concat(parts, ignore_index=True)
    ann = ann[ann["objectType"] == "S.haematobium"].reset_index(drop=True)
    return ann


def _yolo_label_for_tile(
    ann_xy: np.ndarray,
    tile_x: int,
    tile_y: int,
) -> list[str]:
    """Return YOLO-format label lines for one tile.

    Each line: '<class> <x_center_norm> <y_center_norm> <width_norm> <height_norm>'.
    Class 0 = S.haematobium egg.
    Bounding boxes are clipped to tile bounds.
    """
    if len(ann_xy) == 0:
        return []
    in_tile = (
        (ann_xy[:, 0] >= tile_x) & (ann_xy[:, 0] < tile_x + TILE_SIZE) &
        (ann_xy[:, 1] >= tile_y) & (ann_xy[:, 1] < tile_y + TILE_SIZE)
    )
    eggs = ann_xy[in_tile]
    if len(eggs) == 0:
        return []
    lines = []
    half = BBOX_SIZE / 2
    for x, y in eggs:
        # Box in tile coords
        x_local = float(x) - tile_x
        y_local = float(y) - tile_y
        x_min = max(0.0, x_local - half)
        y_min = max(0.0, y_local - half)
        x_max = min(float(TILE_SIZE), x_local + half)
        y_max = min(float(TILE_SIZE), y_local + half)
        w = x_max - x_min
        h = y_max - y_min
        if w <= 0 or h <= 0:
            continue
        cx = (x_min + x_max) / 2.0 / TILE_SIZE
        cy = (y_min + y_max) / 2.0 / TILE_SIZE
        wn = w / TILE_SIZE
        hn = h / TILE_SIZE
        lines.append(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
    return lines


def build(split_csv: Path, contrast: str, out_root: Path) -> None:
    spl = pd.read_csv(split_csv)
    meta = pd.read_csv(IMAGES_CSV)
    ann = _load_annotations()

    contrast_label = {"brightfield": "BF", "darkfield": "DF"}[contrast]
    print(f"Building YOLO dataset: split={split_csv.name} contrast={contrast} ({contrast_label})")

    imgs_dir = ensure_dir(out_root / "images")
    labs_dir = ensure_dir(out_root / "labels")
    for split in ["train", "val", "test"]:
        ensure_dir(imgs_dir / split)
        ensure_dir(labs_dir / split)

    positions = _tile_positions()
    counts = {"train": 0, "val": 0, "test": 0}
    pos_counts = {"train": 0, "val": 0, "test": 0}

    for split in ["train", "val", "test"]:
        keys = set(spl[spl["split"] == split]["patient_key"])
        imgs = meta[
            meta["patient_key"].isin(keys) &
            (meta["contrast"] == contrast)
        ]
        print(f"  {split}: {len(imgs)} source images")
        for _, row in imgs.iterrows():
            img_name    = row["image_name"]
            rel_path    = row["relative_path"]
            img_path    = RAW_DIR / rel_path
            img_ann     = ann[ann["imageName"] == img_name]
            ann_xy      = img_ann[["xCoord", "yCoord"]].values.astype(float)

            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    for ti, (tx, ty) in enumerate(positions):
                        tile = im.crop((tx, ty, tx + TILE_SIZE, ty + TILE_SIZE))
                        stem = f"{Path(img_name).stem}_tile_{ti:02d}"
                        tile_path = imgs_dir / split / f"{stem}.jpg"
                        label_path = labs_dir / split / f"{stem}.txt"
                        tile.save(tile_path, "JPEG", quality=90)
                        lines = _yolo_label_for_tile(ann_xy, tx, ty)
                        label_path.write_text("\n".join(lines))
                        counts[split] += 1
                        if lines:
                            pos_counts[split] += 1
            except Exception as e:
                print(f"    WARN: skipping {img_name}: {e}")

    print("\nDone. Counts:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {counts[split]} tiles total, {pos_counts[split]} with eggs "
              f"({100*pos_counts[split]/max(counts[split],1):.1f}%)")

    yaml_path = out_root / "data.yaml"
    yaml_content = f"""path: {out_root.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: schistosoma_egg
"""
    yaml_path.write_text(yaml_content)
    print(f"\nWrote data.yaml: {yaml_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="mar_to_nov",
                    help="Split file stem under splits/ (default: mar_to_nov)")
    ap.add_argument("--contrast", choices=["brightfield", "darkfield"], default="brightfield")
    ap.add_argument("--out", default=None,
                    help="Output dir (default: data/yolo_<split>_<bf|df>)")
    args = ap.parse_args()

    split_csv = REPO_ROOT / "splits" / f"{args.split}_split.csv"
    contrast_short = "bf" if args.contrast == "brightfield" else "df"
    out_root = Path(args.out) if args.out else REPO_ROOT / "data" / f"yolo_{args.split}_{contrast_short}"
    out_root = ensure_dir(out_root)
    build(split_csv, args.contrast, out_root)


if __name__ == "__main__":
    main()
