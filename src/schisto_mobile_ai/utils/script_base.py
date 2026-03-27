"""Shared command-line behavior for lightweight starter scripts."""

from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from schisto_mobile_ai.config import load_config, save_config_snapshot
from schisto_mobile_ai.paths import REPO_ROOT, ensure_dir
from schisto_mobile_ai.utils.io import write_json, write_text
from schisto_mobile_ai.utils.logging import configure_logging
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything


def build_common_parser(description: str) -> argparse.ArgumentParser:
    """Build a parser with shared flags used across all scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "default.yaml",
        help="Path to a YAML or JSON config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. If omitted, one is created automatically.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="baseline",
        help="Short label added to the output folder name.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the smallest possible path for environment verification.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Use only the first N items when dataset logic is implemented.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Worker count for future data loaders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output.",
    )
    return parser


def _slugify(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in text.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "run"


def _default_output_root(default_output_subdir: str, smoke_test: bool) -> Path:
    if smoke_test and default_output_subdir.startswith("runs/experiments/"):
        return REPO_ROOT / default_output_subdir.replace("runs/experiments/", "runs/smoke/", 1)
    return REPO_ROOT / default_output_subdir


def resolve_output_dir(
    *,
    args: argparse.Namespace,
    task_name: str,
    default_output_subdir: str,
) -> Path:
    """Create and return a descriptive output directory."""
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts = [timestamp, _slugify(task_name), _slugify(args.run_name)]
        if args.subset_size is not None:
            name_parts.append(f"subset{args.subset_size}")
        if args.smoke_test and "runs/smoke/" not in str(_default_output_root(default_output_subdir, True)):
            name_parts.append("smoke")
        output_dir = _default_output_root(default_output_subdir, args.smoke_test) / "_".join(
            name_parts
        )

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    if any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}. "
            "Pass --overwrite or use a different --run-name/--output-dir."
        )

    return output_dir


def write_stub_outputs(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    description: str,
    extra_todos: list[str] | None = None,
) -> None:
    """Persist the minimum artifacts needed for a reproducible stub run."""
    command_text = " ".join(shlex.quote(part) for part in sys.argv)
    write_json(output_dir / "run_args.json", vars(args))
    save_config_snapshot(config, output_dir / "resolved_config.json")
    write_text(output_dir / "command.txt", f"{command_text}\n")

    lines = [
        f"Task: {description}",
        f"Smoke test: {args.smoke_test}",
        f"Subset size: {args.subset_size}",
        "",
        "This command is currently a starter stub.",
        "It created the expected output folder and saved reproducibility metadata.",
        "",
        "Next steps:",
        "TODO: replace this stub with task-specific implementation.",
        "TODO: adapt the logic only after the real dataset schema is documented.",
    ]

    for item in extra_todos or []:
        lines.append(item)

    write_text(output_dir / "STATUS.txt", "\n".join(lines) + "\n")


def run_placeholder_task(
    *,
    task_name: str,
    description: str,
    default_output_subdir: str,
    extra_todos: list[str] | None = None,
) -> int:
    """Shared entry point for script stubs that already behave like real CLIs."""
    parser = build_common_parser(description)
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)
    config = load_config(args.config)
    output_dir = resolve_output_dir(
        args=args,
        task_name=task_name,
        default_output_subdir=default_output_subdir,
    )

    seed_everything(args.seed)
    device = resolve_device(args.device)
    config.setdefault("runtime", {})
    config["runtime"]["resolved_device"] = device

    write_stub_outputs(
        output_dir=output_dir,
        args=args,
        config=config,
        description=description,
        extra_todos=extra_todos,
    )

    logger.info("Created stub output for %s at %s", task_name, output_dir)
    logger.info("Resolved device: %s", device)
    return 0

