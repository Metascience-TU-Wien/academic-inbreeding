# Copyright © 2026 TU Wien

"""Command-line interface for rebuilding regression input artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from acib.pipeline.config import PROJECT_ROOT
from acib.regression.build import build_regression_inputs_from_config
from acib.regression.config import load_regression_config


def build_parser() -> argparse.ArgumentParser:
    """Build the regression-input builder CLI parser.

    :return: Parser for shared pipeline configuration and path validation.
    """
    parser = argparse.ArgumentParser(description="Build configured regression input artifacts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "pipeline.toml",
        help="Path to pipeline TOML configuration.",
    )
    parser.add_argument(
        "--validate-paths",
        action="store_true",
        help="Validate external and upstream regression inputs before building.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Build regression input artifacts without fitting regression models.

    :param argv: Optional command-line arguments for programmatic invocation.
    """
    args = build_parser().parse_args(argv)
    config = load_regression_config(args.config)
    if args.validate_paths:
        config.check_paths_exist()
    build_regression_inputs_from_config(config)
