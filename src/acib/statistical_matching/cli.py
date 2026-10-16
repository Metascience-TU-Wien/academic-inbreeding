# Copyright © 2026 TU Wien

"""Command-line interface for statistical matching."""

from __future__ import annotations

import argparse
from pathlib import Path

from acib.pipeline.config import PROJECT_ROOT, load_statistical_matching_config
from acib.pipeline.dependencies import ensure_stage_dependencies
from acib.statistical_matching.constants import COVARIATE_BALANCE_FIGURE_FILENAME
from acib.statistical_matching.workflow import run_statistical_matching


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: Parser for statistical-matching options.
    """
    parser = argparse.ArgumentParser(description="Run ATT matching and weighting analyses.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/pipeline.toml")
    parser.add_argument("--validate-paths", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run configured matching and print its result table.

    :param argv: Optional command-line arguments.
    """
    args = build_parser().parse_args(argv)
    ensure_stage_dependencies("statistical_matching", args.config)
    config = load_statistical_matching_config(args.config)
    if args.validate_paths:
        config.check_input_exists()
    result = run_statistical_matching(config)
    print(result.console_table)
    plot_path = config.outputs.plots_dir / COVARIATE_BALANCE_FIGURE_FILENAME
    if plot_path.exists():
        print(f"Wrote covariate balance plot: {plot_path}")
