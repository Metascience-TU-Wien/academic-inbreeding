# Copyright © 2026 TU Wien

"""Command-line interface for the name-matching pipeline stage."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from acib.name_matching.config import PROJECT_ROOT, load_name_matching_config
from acib.name_matching.workflow import run_name_matching


def main() -> None:
    """Run name matching using a pipeline TOML configuration."""
    parser = argparse.ArgumentParser(
        description="Match configured researcher names to Scopus authors."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "pipeline.toml",
        help="Path to pipeline TOML configuration.",
    )
    parser.add_argument(
        "--validate-paths",
        action="store_true",
        help="Validate configured input paths before loading data.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_name_matching_config(args.config)
    if args.validate_paths:
        config.check_paths_exist()
    run_name_matching(config)
