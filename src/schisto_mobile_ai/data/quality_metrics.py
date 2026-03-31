"""Real quality-metrics and QC-figure generation for the schistosomiasis dataset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps

from schisto_mobile_ai.data.manifest import validate_required_columns
from schisto_mobile_ai.paths import REPO_ROOT
from schisto_mobile_ai.utils.io import ensure_dir, write_json


VALID_CONTRASTS = ("brightfield", "darkfield")
QUALITY_COLUMNS = [
    "image_id",
    "image_name",
    "study_id",
    "patient_id",
    "patient_key",
    "pair_key",
    "frame_num",
    "contrast",
    "relative_path",
    "file_exists",
    "load_success",
    "error_message",
    "blur_score",
    "brightness_mean",
    "brightness_std",
    "intensity_min",
    "intensity_max",
    "contrast_std",
    "edge_density",
    "width",
    "height",
]

FIGURE_FILENAMES = {
    "blur_score_hist": "blur_score_by_contrast.png",
    "brightness_mean_hist": "brightness_mean_by_contrast.png",
    "contrast_std_hist": "contrast_std_by_contrast.png",
    "edge_density_hist": "edge_density_by_contrast.png",
    "sample_pair_grid": "bf_df_sample_pair_grid.png",
    "sharp_vs_blurry_panel": "sharp_vs_blurry_examples.png",
}


@dataclass
class QualityRunResult:
    """Container for quality-metric outputs and the run summary."""

    quality_frame: pd.DataFrame
    summary: dict[str, Any]
    csv_path: Path
    summary_path: Path
    figures_dir: Path
def _limit_rows(frame: pd.DataFrame, *, subset_size: int | None, smoke_test: bool) -> pd.DataFrame:
    effective_subset = subset_size
    if smoke_test and effective_subset is None:
        effective_subset = 64

    if effective_subset is None:
        return frame.copy()
    if effective_subset <= 0:
        raise ValueError("--subset-size must be a positive integer when provided.")
    return frame.head(effective_subset).copy()


def _guard_outputs(
    *,
    output_csv: Path,
    summary_json: Path,
    figures_dir: Path,
    overwrite: bool,
) -> None:
    blocking_paths = []
    for path in [output_csv, summary_json]:
        if path.exists():
            blocking_paths.append(path)
    if figures_dir.exists() and any(figures_dir.iterdir()):
        blocking_paths.append(figures_dir)

    if blocking_paths and not overwrite:
        blocking_text = ", ".join(str(path) for path in blocking_paths)
        raise FileExistsError(
            "Quality-metric outputs already exist. Pass --overwrite to replace them: "
            f"{blocking_text}"
        )


def _resolve_image_path(raw_dir: Path, relative_path: str) -> Path:
    return raw_dir / Path(relative_path)


def _open_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).copy()


def _resize_for_metrics(image: Image.Image, *, max_side: int) -> Image.Image:
    if max_side <= 0:
        raise ValueError("--max-side must be a positive integer.")

    width, height = image.size
    largest_side = max(width, height)
    if largest_side <= max_side:
        return image

    scale = max_side / float(largest_side)
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(resized_size, Image.Resampling.BILINEAR)


def _laplacian_variance(gray_array: np.ndarray) -> float:
    padded = np.pad(gray_array, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - (4.0 * center)
    )
    return float(np.var(laplacian))


def _edge_density(gray_array: np.ndarray) -> float:
    grad_y, grad_x = np.gradient(gray_array)
    magnitude = np.hypot(grad_x, grad_y)
    threshold = float(magnitude.mean() + magnitude.std())
    return float(np.mean(magnitude > threshold))


def _compute_metrics_from_image(image: Image.Image, *, max_side: int) -> dict[str, float | int]:
    width, height = image.size
    gray_image = image.convert("L")
    metric_image = _resize_for_metrics(gray_image, max_side=max_side)
    gray_array = np.asarray(metric_image, dtype=np.float32)
    blurred_array = np.asarray(metric_image.filter(ImageFilter.BoxBlur(radius=4)), dtype=np.float32)

    brightness_mean = float(gray_array.mean())
    brightness_std = float(gray_array.std())
    local_contrast = gray_array - blurred_array

    return {
        "blur_score": _laplacian_variance(gray_array),
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "intensity_min": float(gray_array.min()),
        "intensity_max": float(gray_array.max()),
        "contrast_std": float(local_contrast.std()),
        "edge_density": _edge_density(gray_array),
        "width": int(width),
        "height": int(height),
    }


def _build_quality_row(row: pd.Series, raw_dir: Path, *, max_side: int) -> dict[str, Any]:
    quality_row: dict[str, Any] = {
        "image_id": row["image_id"],
        "image_name": row.get("image_name", ""),
        "study_id": row["study_id"],
        "patient_id": row["patient_id"],
        "patient_key": row["patient_key"],
        "pair_key": row["pair_key"],
        "frame_num": row.get("frame_num", ""),
        "contrast": row["contrast"],
        "relative_path": row["relative_path"],
        "file_exists": False,
        "load_success": False,
        "error_message": "",
        "blur_score": np.nan,
        "brightness_mean": np.nan,
        "brightness_std": np.nan,
        "intensity_min": np.nan,
        "intensity_max": np.nan,
        "contrast_std": np.nan,
        "edge_density": np.nan,
        "width": np.nan,
        "height": np.nan,
    }

    image_path = _resolve_image_path(raw_dir, str(row["relative_path"]))
    quality_row["file_exists"] = image_path.exists()
    if not image_path.exists():
        quality_row["error_message"] = f"Missing image file: {image_path}"
        return quality_row

    try:
        image = _open_image(image_path)
    except Exception as exc:
        quality_row["error_message"] = f"{type(exc).__name__}: {exc}"
        return quality_row

    metrics = _compute_metrics_from_image(image, max_side=max_side)
    quality_row.update(metrics)
    quality_row["load_success"] = True
    return quality_row


def load_quality_inputs(
    images_csv: str | Path,
    pairs_csv: str | Path,
    *,
    subset_size: int | None = None,
    smoke_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the metadata tables needed for QC."""
    images_path = Path(images_csv)
    pairs_path = Path(pairs_csv)
    if not images_path.exists():
        raise FileNotFoundError(f"images.csv does not exist: {images_path}")
    if not pairs_path.exists():
        raise FileNotFoundError(f"pairs.csv does not exist: {pairs_path}")

    images = pd.read_csv(images_path, dtype=str)
    pairs = pd.read_csv(pairs_path, dtype=str)
    validate_required_columns(
        images,
        ["image_id", "study_id", "patient_id", "patient_key", "pair_key", "contrast", "relative_path"],
        table_name="images.csv",
    )
    validate_required_columns(
        pairs,
        ["pair_key", "pair_status", "brightfield_relative_path", "darkfield_relative_path"],
        table_name="pairs.csv",
    )

    images = images[images["contrast"].isin(VALID_CONTRASTS)].copy()
    images = images.sort_values(["study_id", "patient_id", "frame_num", "contrast", "relative_path"]).reset_index(drop=True)
    images = _limit_rows(images, subset_size=subset_size, smoke_test=smoke_test)

    allowed_pair_keys = set(images["pair_key"])
    pairs = pairs[pairs["pair_key"].isin(allowed_pair_keys)].copy()
    pairs = pairs.sort_values(["study_id", "patient_id", "frame_num"]).reset_index(drop=True)
    return images, pairs


def compute_image_quality_frame(
    images: pd.DataFrame,
    *,
    raw_dir: str | Path,
    max_side: int = 768,
) -> pd.DataFrame:
    """Compute real per-image quality metrics from the image files."""
    root = Path(raw_dir)
    records = [
        _build_quality_row(row, root, max_side=max_side)
        for _, row in images.iterrows()
    ]
    quality_frame = pd.DataFrame(records)
    return quality_frame[QUALITY_COLUMNS].copy()


def _metric_summary(series: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def _sample_evenly(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if frame.empty or count <= 0:
        return frame.iloc[0:0].copy()
    if len(frame) <= count:
        return frame.copy()
    indices = np.linspace(0, len(frame) - 1, count, dtype=int)
    return frame.iloc[indices].copy()


def _display_array(path: Path, *, max_side: int = 400) -> np.ndarray:
    image = _open_image(path).convert("RGB")
    image = _resize_for_metrics(image, max_side=max_side)
    return np.asarray(image)


def _plot_histogram_by_contrast(
    quality_frame: pd.DataFrame,
    *,
    metric_name: str,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    valid = quality_frame[quality_frame["load_success"]].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"brightfield": "#d17b0f", "darkfield": "#1f5a7a"}
    bins = 40

    for contrast in VALID_CONTRASTS:
        contrast_values = pd.to_numeric(
            valid.loc[valid["contrast"] == contrast, metric_name],
            errors="coerce",
        ).dropna()
        if contrast_values.empty:
            continue
        ax.hist(
            contrast_values,
            bins=bins,
            alpha=0.6,
            label=f"{contrast} (n={len(contrast_values)})",
            color=colors[contrast],
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Image count")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_pair_grid(
    pairs: pd.DataFrame,
    quality_frame: pd.DataFrame,
    *,
    raw_dir: Path,
    output_path: Path,
    max_examples: int,
) -> None:
    complete_pairs = pairs[pairs["pair_status"] == "complete"].copy()
    if complete_pairs.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No complete BF/DF pairs available.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    sampled_pairs = _sample_evenly(complete_pairs, max_examples)
    blur_lookup = quality_frame.set_index("image_id")["blur_score"].to_dict()
    row_count = len(sampled_pairs)
    fig, axes = plt.subplots(row_count, 2, figsize=(10, max(3, row_count * 3)))
    if row_count == 1:
        axes = np.array([axes])

    for row_index, (_, pair_row) in enumerate(sampled_pairs.iterrows()):
        pair_key = pair_row["pair_key"]
        for column_index, side in enumerate(["brightfield", "darkfield"]):
            axis = axes[row_index, column_index]
            image_path = raw_dir / Path(pair_row[f"{side}_relative_path"])
            image_id = pair_row[f"{side}_image_id"]
            if image_path.exists():
                axis.imshow(_display_array(image_path, max_side=320))
            else:
                axis.text(0.5, 0.5, "Missing file", ha="center", va="center")
            blur_score = blur_lookup.get(image_id, np.nan)
            axis.set_title(
                f"{side}\n{pair_key}\nblur={blur_score:.2f}" if pd.notna(blur_score) else f"{side}\n{pair_key}"
            )
            axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_sharp_vs_blurry_panel(
    quality_frame: pd.DataFrame,
    *,
    raw_dir: Path,
    output_path: Path,
    sample_count: int,
) -> None:
    valid = quality_frame[quality_frame["load_success"]].copy()
    valid["blur_score"] = pd.to_numeric(valid["blur_score"], errors="coerce")
    valid = valid.dropna(subset=["blur_score"]).sort_values("blur_score").reset_index(drop=True)

    if valid.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No valid images available for blur examples.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    blurry = valid.head(sample_count)
    sharp = valid.tail(sample_count).sort_values("blur_score", ascending=False)
    column_count = max(len(blurry), len(sharp))
    fig, axes = plt.subplots(2, column_count, figsize=(3 * column_count, 6))
    if column_count == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for row_index, (label, frame) in enumerate([("Blurry", blurry), ("Sharp", sharp)]):
        for column_index in range(column_count):
            axis = axes[row_index, column_index]
            axis.axis("off")
            if column_index >= len(frame):
                continue
            record = frame.iloc[column_index]
            image_path = raw_dir / Path(record["relative_path"])
            if image_path.exists():
                axis.imshow(_display_array(image_path, max_side=280))
            else:
                axis.text(0.5, 0.5, "Missing file", ha="center", va="center")
            axis.set_title(
                f"{label}\n{record['image_name']}\n{record['contrast']} | blur={record['blur_score']:.2f}",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def generate_qc_figures(
    quality_frame: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    raw_dir: str | Path,
    figures_dir: str | Path,
    pair_samples: int = 6,
    sharp_samples: int = 4,
) -> list[Path]:
    """Generate the histogram and example-panel QC figures."""
    destination = ensure_dir(figures_dir)
    raw_root = Path(raw_dir)

    figure_paths = [
        destination / FIGURE_FILENAMES["blur_score_hist"],
        destination / FIGURE_FILENAMES["brightness_mean_hist"],
        destination / FIGURE_FILENAMES["contrast_std_hist"],
        destination / FIGURE_FILENAMES["edge_density_hist"],
        destination / FIGURE_FILENAMES["sample_pair_grid"],
        destination / FIGURE_FILENAMES["sharp_vs_blurry_panel"],
    ]

    _plot_histogram_by_contrast(
        quality_frame,
        metric_name="blur_score",
        title="Blur Score by Contrast",
        xlabel="Variance of Laplacian",
        output_path=figure_paths[0],
    )
    _plot_histogram_by_contrast(
        quality_frame,
        metric_name="brightness_mean",
        title="Brightness Mean by Contrast",
        xlabel="Mean grayscale intensity",
        output_path=figure_paths[1],
    )
    _plot_histogram_by_contrast(
        quality_frame,
        metric_name="contrast_std",
        title="Local Contrast Std by Contrast",
        xlabel="Local contrast standard deviation",
        output_path=figure_paths[2],
    )
    _plot_histogram_by_contrast(
        quality_frame,
        metric_name="edge_density",
        title="Edge Density by Contrast",
        xlabel="Proportion of edge pixels",
        output_path=figure_paths[3],
    )
    _plot_pair_grid(
        pairs,
        quality_frame,
        raw_dir=raw_root,
        output_path=figure_paths[4],
        max_examples=pair_samples,
    )
    _plot_sharp_vs_blurry_panel(
        quality_frame,
        raw_dir=raw_root,
        output_path=figure_paths[5],
        sample_count=sharp_samples,
    )
    return figure_paths


def build_quality_summary(
    quality_frame: pd.DataFrame,
    *,
    images_csv: Path,
    pairs_csv: Path,
    raw_dir: Path,
    output_csv: Path,
    summary_json: Path,
    figures_dir: Path,
    figure_paths: list[Path],
    max_side: int,
    subset_size: int | None,
    smoke_test: bool,
) -> dict[str, Any]:
    """Build a compact JSON summary describing the quality-metric run."""
    valid = quality_frame[quality_frame["load_success"]].copy()
    missing = quality_frame[~quality_frame["file_exists"]].copy()
    failed = quality_frame[quality_frame["file_exists"] & ~quality_frame["load_success"]].copy()

    summary = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "repo_root": str(REPO_ROOT),
        "inputs": {
            "images_csv": str(images_csv),
            "pairs_csv": str(pairs_csv),
            "raw_dir": str(raw_dir),
            "subset_size": subset_size,
            "smoke_test": bool(smoke_test),
            "max_side": int(max_side),
        },
        "outputs": {
            "image_quality_csv": str(output_csv),
            "image_quality_summary_json": str(summary_json),
            "figures_dir": str(figures_dir),
            "figure_files": [str(path) for path in figure_paths],
        },
        "summary": {
            "n_images_requested": int(len(quality_frame)),
            "n_images_processed": int(valid["image_id"].nunique()),
            "n_missing_files": int(len(missing)),
            "n_load_failures": int(len(failed)),
            "by_contrast": {
                contrast: int((quality_frame["contrast"] == contrast).sum())
                for contrast in VALID_CONTRASTS
            },
            "metrics": {
                "blur_score": _metric_summary(valid["blur_score"]),
                "brightness_mean": _metric_summary(valid["brightness_mean"]),
                "contrast_std": _metric_summary(valid["contrast_std"]),
                "edge_density": _metric_summary(valid["edge_density"]),
            },
        },
        "missing_file_examples": missing[
            ["image_id", "relative_path", "error_message"]
        ].head(20).to_dict(orient="records"),
        "load_failure_examples": failed[
            ["image_id", "relative_path", "error_message"]
        ].head(20).to_dict(orient="records"),
    }
    return summary


def format_quality_summary(summary: dict[str, Any]) -> str:
    """Format a human-readable terminal summary."""
    stats = summary["summary"]
    metric_blur = stats["metrics"]["blur_score"]
    lines = [
        "Quality Metrics Summary",
        f"  Images requested: {stats['n_images_requested']}",
        f"  Images processed: {stats['n_images_processed']}",
        f"  Missing files: {stats['n_missing_files']}",
        f"  Load failures: {stats['n_load_failures']}",
        f"  Brightfield images: {stats['by_contrast']['brightfield']}",
        f"  Darkfield images: {stats['by_contrast']['darkfield']}",
        f"  Blur score mean: {metric_blur['mean']:.2f}" if metric_blur["mean"] is not None else "  Blur score mean: n/a",
        f"  Output CSV: {summary['outputs']['image_quality_csv']}",
        f"  Summary JSON: {summary['outputs']['image_quality_summary_json']}",
        f"  Figures dir: {summary['outputs']['figures_dir']}",
    ]
    return "\n".join(lines)


def run_quality_metrics(
    *,
    images_csv: str | Path,
    pairs_csv: str | Path,
    raw_dir: str | Path,
    output_csv: str | Path,
    summary_json: str | Path,
    figures_dir: str | Path,
    subset_size: int | None = None,
    smoke_test: bool = False,
    max_side: int = 768,
    pair_samples: int = 6,
    sharp_samples: int = 4,
    overwrite: bool = False,
) -> QualityRunResult:
    """Run the full QC pipeline and persist CSV, JSON, and figures."""
    images_path = Path(images_csv)
    pairs_path = Path(pairs_csv)
    raw_root = Path(raw_dir)
    output_csv_path = Path(output_csv)
    summary_json_path = Path(summary_json)
    figures_dir_path = Path(figures_dir)

    _guard_outputs(
        output_csv=output_csv_path,
        summary_json=summary_json_path,
        figures_dir=figures_dir_path,
        overwrite=overwrite,
    )
    ensure_dir(output_csv_path.parent)
    ensure_dir(summary_json_path.parent)
    ensure_dir(figures_dir_path)

    images, pairs = load_quality_inputs(
        images_path,
        pairs_path,
        subset_size=subset_size,
        smoke_test=smoke_test,
    )
    quality_frame = compute_image_quality_frame(images, raw_dir=raw_root, max_side=max_side)
    quality_frame.to_csv(output_csv_path, index=False)

    figure_paths = generate_qc_figures(
        quality_frame,
        pairs,
        raw_dir=raw_root,
        figures_dir=figures_dir_path,
        pair_samples=pair_samples,
        sharp_samples=sharp_samples,
    )
    summary = build_quality_summary(
        quality_frame,
        images_csv=images_path,
        pairs_csv=pairs_path,
        raw_dir=raw_root,
        output_csv=output_csv_path,
        summary_json=summary_json_path,
        figures_dir=figures_dir_path,
        figure_paths=figure_paths,
        max_side=max_side,
        subset_size=subset_size,
        smoke_test=smoke_test,
    )
    write_json(summary_json_path, summary)

    return QualityRunResult(
        quality_frame=quality_frame,
        summary=summary,
        csv_path=output_csv_path,
        summary_path=summary_json_path,
        figures_dir=figures_dir_path,
    )
