# Copyright © 2026 TU Wien

"""Command-line interface for the standalone mobility stage."""

import argparse
import logging
from pathlib import Path

from acib.mobility.workflow import run_mobility
from acib.pipeline.config import PROJECT_ROOT, load_mobility_config


def build_parser() -> argparse.ArgumentParser:
    """Build the source-compatible mobility command-line parser.

    :return: Parser for the mobility config and path-validation options.
    """
    parser = argparse.ArgumentParser(description="Run academic mobility analysis.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "pipeline.toml",
        help="Path to pipeline TOML configuration.",
    )
    parser.add_argument(
        "--validate-paths",
        action="store_true",
        help="Validate mobility input paths before starting analysis.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run mobility from the configured pipeline TOML.

    :param argv: Optional command-line arguments, primarily for invocation tests.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    config = load_mobility_config(args.config)
    if args.validate_paths:
        config.check_input_paths()
    run_mobility(config)
