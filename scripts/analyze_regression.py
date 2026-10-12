# Copyright © 2026 TU Wien

"""Script entry point for regression analysis."""

from __future__ import annotations

from pathlib import Path

from acib.regression.cli import main
from acib.regression.config import RegressionConfig, load_regression_config
from acib.regression.workflow import run_regression_suite


def run_regression(
    config: RegressionConfig,
    *,
    print_model_summaries: bool = False,
) -> None:
    """Run regression with an already-resolved stage configuration.

    :param config: Regression paths, options, and output contracts.
    :param print_model_summaries: Print fitted summaries before writing outputs.
    """
    run_regression_suite(config, print_model_summaries=print_model_summaries)


def run_regression_from_config(
    *,
    config_path: Path,
    validate_paths: bool = False,
    print_model_summaries: bool = False,
) -> None:
    """Load and run regression from the shared pipeline TOML.

    :param config_path: Shared pipeline configuration path.
    :param validate_paths: Validate mode-specific inputs before execution.
    :param print_model_summaries: Print fitted summaries before output writing.
    """
    config = load_regression_config(config_path)
    if validate_paths:
        if config.rebuild_input:
            config.check_paths_exist()
        else:
            config.check_regression_input_exists()
    run_regression(config, print_model_summaries=print_model_summaries)


if __name__ == "__main__":
    main()
