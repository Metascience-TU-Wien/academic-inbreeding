# Copyright © 2026 TU Wien

"""Command-line interface for independently running regression analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from acib.pipeline.config import PROJECT_ROOT
from acib.regression.workflow import run_regression_suite_from_config


def build_parser() -> argparse.ArgumentParser:
    """Build the regression CLI parser.

    :return: Parser for the pipeline config and regression-specific options.
    """
    parser = argparse.ArgumentParser(description="Run configured regression models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "pipeline.toml",
        help="Path to pipeline TOML configuration.",
    )
    parser.add_argument(
        "--validate-paths",
        action="store_true",
        help="Validate regression inputs before running.",
    )
    parser.add_argument(
        "--print-model-summaries",
        action="store_true",
        help="Print fitted model summaries before writing outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run regression from shared pipeline configuration.

    :param argv: Optional command-line arguments for programmatic invocation.
    """
    args = build_parser().parse_args(argv)
    run_regression_suite_from_config(
        config_path=args.config,
        validate_paths=args.validate_paths,
        print_model_summaries=args.print_model_summaries,
    )
