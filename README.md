# Resource-Aware Conditional Dual-Contrast AI for Schistosomiasis Diagnosis

This repository is a clean starting point for a high-school-level but publication-serious machine learning project on mobile microscopy images.

The goal is to keep the code:

- easy to read
- reproducible
- light enough to run on Apple silicon without a GPU
- structured so that dataset-specific choices are explicit instead of hidden

The current scaffold gives you:

- a `src/` package for reusable Python code
- a `scripts/` folder for command-line entry points
- a simple YAML config pattern
- dedicated `runs/` and `results/` folders
- smoke-test and small-subset flags in every script stub

## Project Principles

- Do not hardcode dataset assumptions before the real dataset is audited.
- Keep core logic in Python modules, not notebooks.
- Make every experiment reproducible by saving config and arguments.
- Prefer lightweight, common libraries.
- Support `cpu` and `mps` execution on macOS Apple silicon.

## Recommended Setup

1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies.
3. Run a smoke test to confirm the scaffold works.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/audit_dataset.py --smoke-test --run-name scaffold_check
```

## Repository Layout

```text
.
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── results/
│   ├── audits/
│   ├── benchmarks/
│   ├── evaluations/
│   ├── metadata/
│   ├── quality_metrics/
│   └── splits/
├── runs/
│   ├── experiments/
│   └── smoke/
├── scripts/
│   ├── audit_dataset.py
│   ├── benchmark_inference.py
│   ├── build_metadata.py
│   ├── compute_quality_metrics.py
│   ├── eval_patient_level.py
│   ├── make_splits.py
│   ├── train_dual_contrast.py
│   └── train_single_contrast.py
└── src/
    └── schisto_mobile_ai/
        ├── data/
        ├── eval/
        ├── models/
        ├── profiling/
        └── utils/
```

## Configuration Pattern

The default config lives in [`configs/default.yaml`](/Users/dylanashraf/Documents/Programming/diagnostic_system_regeneron/regeneron_schistosomiasis_project/configs/default.yaml).

Why YAML?

- it is easy to read
- it supports comments
- it is simple for students to edit safely

Use the config like this:

```bash
python scripts/train_single_contrast.py --config configs/default.yaml --subset-size 32
```

Important: the config intentionally includes `null` placeholders and TODO comments for dataset-specific fields such as metadata schema, grouping, and contrast definitions.

## Script Conventions

Every script in `scripts/`:

- uses `argparse`
- accepts `--smoke-test`
- accepts `--subset-size`
- accepts `--config`
- creates a clearly named output folder
- saves the parsed arguments and resolved config

Example:

```bash
python scripts/build_metadata.py \
  --config configs/default.yaml \
  --subset-size 50 \
  --run-name first_pass
```

## What Is Implemented Right Now

This scaffold is intentionally conservative.

Implemented now:

- shared config loading
- output-folder creation
- reproducibility helpers
- basic model and evaluation placeholders
- reusable script stub behavior

Still TODO for the real project:

- define dataset schema after auditing the files
- decide the exact single-contrast and dual-contrast sample format
- implement metadata parsing for the real dataset
- implement training loops, augmentations, and evaluation pipelines
- benchmark real model inference on representative inputs

## Suggested First Steps

1. Put raw files in `data/raw/`.
2. Run `python scripts/audit_dataset.py --smoke-test`.
3. Decide and document the metadata schema.
4. Implement `build_metadata.py` for the real dataset.
5. Create patient-safe splits before any model training.

## Reproducibility Notes

- Script outputs are written into task-specific folders under `runs/` or `results/`.
- Each stub saves `run_args.json`, `resolved_config.json`, and a short status file.
- Seeds and device selection are centralized in the `utils/` package.

## Device Support

The helpers in `src/schisto_mobile_ai/utils/reproducibility.py` support:

- `cpu`
- `mps`
- `auto` (prefers `mps` when available)

No CUDA-specific logic is required for this repo.

