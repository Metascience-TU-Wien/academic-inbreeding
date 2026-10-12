# Copyright © 2026 TU Wien

"""Independent statistical-matching workflow."""

from __future__ import annotations

from pathlib import Path

from acib.pipeline.config import load_statistical_matching_config
from acib.regression.input import read_table
from acib.statistical_matching.config import StatisticalMatchingConfig
from acib.statistical_matching.constants import BALANCE_THRESHOLD, DEFAULT_METHODS
from acib.statistical_matching.plots import write_matching_plots
from acib.statistical_matching.schemas import validate_matching_input
from acib.statistical_matching.suite import StatisticalMatchingResult, fit_suite


def run_statistical_matching(
    config: StatisticalMatchingConfig, *, write_outputs: bool = True
) -> StatisticalMatchingResult:
    """Validate, fit, and optionally persist matching results.

    :param config: Resolved matching configuration.
    :param write_outputs: Whether to write tables and plots.
    :return: Estimates and diagnostics from the matching suite.
    """
    config.check_input_exists()
    data = read_table(config.input_path)
    required = {config.treatment_variable, *config.covariates, *config.outcomes}
    data = validate_matching_input(data, required)
    result = fit_suite(data, config)
    if write_outputs:
        from acib.statistical_matching.output import write_outputs as write_artifacts

        write_artifacts(result, config.outputs, treatment_variable=config.treatment_variable)
        write_matching_plots(
            result.covariate_balance,
            result.balance_summary,
            result.propensity_scores,
            result.propensity_scores_by_method,
            result.outcome_effects,
            tuple(o for o in config.outcomes if config.outcome_types.get(o) == "absolute"),
            config.outcome_aliases,
            DEFAULT_METHODS,
            config.treatment_variable,
            BALANCE_THRESHOLD,
            config.outputs.plots_dir,
        )
    return result


def run_statistical_matching_suite(
    config: StatisticalMatchingConfig, *, write_outputs: bool = True
) -> StatisticalMatchingResult:
    """Run the configured matching suite.

    :param config: Resolved matching configuration.
    :param write_outputs: Whether to write tables and plots.
    :return: Estimates and diagnostics from the suite.
    """
    return run_statistical_matching(config, write_outputs=write_outputs)


def run_statistical_matching_suite_from_config(
    *,
    config_path: str | Path,
    validate_paths: bool = False,
    write_outputs: bool = True,
) -> StatisticalMatchingResult:
    """Load canonical configuration and run matching independently.

    :param config_path: Pipeline configuration file.
    :param validate_paths: Whether to check the input artifact before running.
    :param write_outputs: Whether to write tables and plots.
    :return: Estimates and diagnostics from the suite.
    """
    config = load_statistical_matching_config(config_path)
    if validate_paths:
        config.check_input_exists()
    return run_statistical_matching(config, write_outputs=write_outputs)
