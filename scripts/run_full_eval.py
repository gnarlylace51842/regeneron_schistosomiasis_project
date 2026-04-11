#!/usr/bin/env python3
"""Run all evaluation experiments after training completes.

Orchestrates:
  1. Conditional inference on augmented-trained BYOL models (no pseudo-supervision)
  2. Conditional inference on pseudo-supervised models (built on augmented models)
  3. Both with TTA=8 for best results
  4. Regenerates model comparison table with all results

Run after the sequential training pipeline completes:
    python scripts/run_full_eval.py

Or specify individual models to evaluate:
    python scripts/run_full_eval.py --bf-model path/to/bf.pt --df-model path/to/df.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
BYOL_WEIGHTS = (REPO_ROOT / "runs" / "ssl" / "pretrain_byol" /
                "20260408_105425_pretrain_byol_byol_pretrain_100ep" / "byol_model_weights.pt")


def _find_best_model(finetune_dir: Path, pattern: str) -> Path | None:
    """Find best_model.pt for the most recent run matching pattern."""
    matches = sorted(
        [d for d in finetune_dir.iterdir()
         if d.is_dir() and pattern in d.name and "smoke" not in d.name]
    )
    for run_dir in reversed(matches):
        pt = run_dir / "best_model.pt"
        if pt.exists():
            return pt
    return None


def _run_eval(
    bf_model: Path,
    df_model: Path,
    run_name: str,
    *,
    tta_views: int = 8,
    eval_split: str = "val",
    gate_mode: str = "both",
    overwrite: bool = True,
) -> dict:
    """Run eval_conditional_inference.py and return operating points."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_conditional_inference.py"),
        "--bf-model", str(bf_model),
        "--df-model", str(df_model),
        "--byol-weights", str(BYOL_WEIGHTS),
        "--gate-mode", gate_mode,
        "--eval-split", eval_split,
        "--tta-views", str(tta_views),
        "--run-name", run_name,
        "--output-dir", str(REPO_ROOT / "results" / "conditional_inference"),
    ]
    if overwrite:
        cmd.append("--overwrite")

    print(f"\n{'=' * 70}")
    print(f"  Running eval: {run_name} (TTA={tta_views}, split={eval_split})")
    print(f"  BF:  {bf_model.name}")
    print(f"  DF:  {df_model.name}")
    print(f"{'=' * 70}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"WARNING: eval failed for {run_name} (exit code {result.returncode})")
        return {}

    ops_path = REPO_ROOT / "results" / "conditional_inference" / run_name / "operating_points.json"
    if ops_path.exists():
        with open(ops_path) as f:
            return json.load(f)
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf-model", type=Path, default=None,
                        help="Override BF model path (default: auto-detect augmented model)")
    parser.add_argument("--df-model", type=Path, default=None,
                        help="Override DF model path (default: auto-detect augmented model)")
    parser.add_argument("--tta-views", type=int, default=8)
    parser.add_argument("--skip-pseudo", action="store_true",
                        help="Skip pseudo-supervised model evaluation")
    args = parser.parse_args()

    finetune_dir = REPO_ROOT / "runs" / "ssl" / "finetune"
    pseudo_dir = REPO_ROOT / "runs" / "pseudo_supervised"

    # ── Locate augmented BYOL models ──
    if args.bf_model:
        aug_bf = args.bf_model
    else:
        aug_bf = _find_best_model(finetune_dir, "byol_aug_bf_1p00")
        if aug_bf is None:
            aug_bf = _find_best_model(finetune_dir, "byol_ft_bf_1p00")
            print(f"  augmented BF not found, falling back to: {aug_bf}")

    if args.df_model:
        aug_df = args.df_model
    else:
        aug_df = _find_best_model(finetune_dir, "byol_aug_df_1p00")
        if aug_df is None:
            aug_df = _find_best_model(finetune_dir, "byol_ft_df_1p00")
            print(f"  augmented DF not found, falling back to: {aug_df}")

    if aug_bf is None or aug_df is None:
        print("ERROR: Could not find BF or DF model. Run fine-tuning first.", file=sys.stderr)
        return 1

    print(f"\nAugmented BF model: {aug_bf}")
    print(f"Augmented DF model: {aug_df}")

    results = {}

    # ── Eval 1: Augmented BYOL models (no pseudo) ──
    ops_aug_val = _run_eval(
        aug_bf, aug_df,
        run_name="byol_aug_val",
        tta_views=args.tta_views,
        eval_split="val",
    )
    results["aug_val"] = ops_aug_val

    ops_aug_test = _run_eval(
        aug_bf, aug_df,
        run_name="byol_aug_test",
        tta_views=args.tta_views,
        eval_split="test",
    )
    results["aug_test"] = ops_aug_test

    # ── Eval 2: Pseudo-supervised models ──
    if not args.skip_pseudo:
        pseudo_bf = pseudo_dir / "bf_model_round1.pt"
        pseudo_df = pseudo_dir / "df_model_round1.pt"

        if pseudo_bf.exists() and pseudo_df.exists():
            print(f"\nPseudo-supervised BF model: {pseudo_bf}")
            print(f"Pseudo-supervised DF model: {pseudo_df}")

            ops_pseudo_val = _run_eval(
                pseudo_bf, pseudo_df,
                run_name="byol_pseudo_val",
                tta_views=args.tta_views,
                eval_split="val",
            )
            results["pseudo_val"] = ops_pseudo_val

            ops_pseudo_test = _run_eval(
                pseudo_bf, pseudo_df,
                run_name="byol_pseudo_test",
                tta_views=args.tta_views,
                eval_split="test",
            )
            results["pseudo_test"] = ops_pseudo_test
        else:
            print(f"\nPseudo-supervised models not found at {pseudo_dir} — skipping.")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for tag, ops in results.items():
        if not ops:
            print(f"{tag}: FAILED")
            continue
        baselines = ops.get("baselines", {})
        peak = ops.get("confidence_gate_peak", {})
        bf_auc = baselines.get("bf_only_patient_auc", float("nan"))
        fused_auc = baselines.get("always_fused_naive_patient_auc", float("nan"))
        cond_auc = peak.get("patient_auc", float("nan"))
        df_frac = peak.get("df_fraction", float("nan"))
        print(f"\n{tag.upper()}")
        print(f"  BF-only AUC:     {bf_auc:.4f}")
        print(f"  Always-fused:    {fused_auc:.4f}")
        print(f"  Conditional peak: {cond_auc:.4f} at {df_frac:.0%} DF")
        sens80 = peak.get("sensitivity_at_spec0p8")
        if sens80:
            print(f"  Sens@80%spec:    {float(sens80):.4f}")

    # ── Regenerate comparison table ──
    print("\n" + "=" * 70)
    print("Regenerating model comparison table...")
    comp_result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "plot_model_comparison.py")],
        capture_output=False, text=True,
    )
    if comp_result.returncode != 0:
        print("WARNING: comparison table generation failed.")

    out = REPO_ROOT / "results" / "eval_summary.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved eval summary to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
