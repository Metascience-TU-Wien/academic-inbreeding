#!/usr/bin/env python3
# Copyright © 2026 TU Wien

"""Compatibility script entry point for building regression input artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from acib.pipeline.config import PROJECT_ROOT
from acib.regression.build import (
    build_regression_inputs_from_config,
    prepare_table_for_output as _prepare_table_for_output,
)
from acib.regression.config import RegressionConfig, load_regression_config


def build_regression_input_from_config(config: RegressionConfig) -> pd.DataFrame:
    """Build configured regression outputs and return the author-level table.

    :param config: Resolved regression stage configuration.
    :return: Researcher-level regression input.
    """
    return build_regression_inputs_from_config(config).regression_input


def prepare_table_for_output(data: pd.DataFrame) -> pd.DataFrame:
    """Serialize nested values using the regression artifact writer's rules.

    :param data: Table to serialize.
    :return: A copy with nested containers encoded as JSON text.
    """
    return _prepare_table_for_output(data)


def build_regression_input_cli(*, config_path: Path, validate_paths: bool = False) -> pd.DataFrame:
    """Build regression inputs from shared pipeline configuration.

    :param config_path: Shared pipeline TOML path.
    :param validate_paths: Validate all required build inputs before running.
    :return: Researcher-level regression input.
    """
    config = load_regression_config(config_path)
    if validate_paths:
        config.check_paths_exist()
    return build_regression_input_from_config(config)


def main() -> None:
    """Run the standalone regression input builder."""
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
    args = parser.parse_args()
    build_regression_input_cli(config_path=args.config, validate_paths=args.validate_paths)


if __name__ == "__main__":
    main()
