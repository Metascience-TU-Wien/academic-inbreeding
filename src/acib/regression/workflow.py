# Copyright © 2026 TU Wien

"""Execute regression input construction and configured model fitting."""

from __future__ import annotations

from pathlib import Path

import structlog

from acib.regression.analysis_config import regression_suite_config_from_pipeline_config
from acib.regression.build import build_regression_inputs_from_config
from acib.regression.config import RegressionConfig, load_regression_config
from acib.regression.input import read_table
from acib.regression.schemas import (
    ProfessorArticleOutputSchema,
    RegressionInputOutputSchema,
)
from acib.regression.suite import RegressionSuite, RegressionSuiteResult

_logger = structlog.getLogger(__name__)


def run_regression_suite(
    config: RegressionConfig,
    *,
    write_outputs: bool = True,
    print_model_summaries: bool = False,
) -> RegressionSuiteResult:
    """Build or load regression inputs and fit the configured model suite.

    :param config: Resolved regression inputs, outputs, and scientific options.
    :param write_outputs: Whether to write configured model and report artifacts.
    :param print_model_summaries: Print all fitted summaries before writing.
    :return: Fitted regression suite and model diagnostics.
    :raises FileNotFoundError: If a required input artifact is missing.
    """
    _logger.info(
        "Starting regression workflow",
        output_paths=[str(path) for path in config.outputs.all_paths()],
        rebuild_input=config.rebuild_input,
    )
    if config.rebuild_input:
        built_inputs = build_regression_inputs_from_config(config)
        regression_input = built_inputs.regression_input
        professor_articles = built_inputs.professor_articles
    else:
        config.check_regression_input_exists()
        regression_input = RegressionInputOutputSchema.validate(
            read_table(config.outputs.regression_input), lazy=True
        )
        professor_articles = None
        if config.outputs.professor_article_input.is_file():
            professor_articles = ProfessorArticleOutputSchema.validate(
                read_table(config.outputs.professor_article_input), lazy=True
            )

    missing_outcomes = set(config.dependent_variables) - set(regression_input.columns)
    if missing_outcomes:
        raise KeyError(
            "Regression input is missing configured outcome(s): "
            + ", ".join(sorted(missing_outcomes))
        )

    suite_config = regression_suite_config_from_pipeline_config(config)
    _logger.info(
        "Fitting regression models",
        regression_rows=len(regression_input),
        dependent_variables=suite_config.dependent_variables,
        model_count=len(suite_config.model_specs),
    )
    suite = RegressionSuite(data=regression_input, config=suite_config)
    result = suite.fit()
    if print_model_summaries or config.print_model_summaries:
        result.print_model_summaries()
    if write_outputs:
        _logger.info(
            "Writing regression outputs",
            output_paths=[str(path) for path in config.outputs.all_paths()],
        )
        result.write(
            config.outputs,
            data=suite.data,
            professor_articles=professor_articles,
        )
    _logger.info(
        "Finished regression workflow",
        fitted_model_count=len(result.models),
        output_paths=[str(path) for path in config.outputs.all_paths()],
    )
    return result


def run_regression_suite_from_config(
    *,
    config_path: str | Path,
    validate_paths: bool = False,
    write_outputs: bool = True,
    print_model_summaries: bool = False,
) -> RegressionSuiteResult:
    """Ensure required upstream artifacts, then run the configured regression suite.

    :param config_path: Pipeline TOML file.
    :param validate_paths: Validate mode-specific required inputs before running.
    :param write_outputs: Whether to write the configured output artifacts.
    :param print_model_summaries: Print fitted summaries before output writing.
    :return: Fitted regression suite result.
    :raises KeyError: If the regression stage is not configured.
    """
    from acib.pipeline.dependencies import ensure_stage_dependencies

    ensure_stage_dependencies("regression", config_path)
    config = load_regression_config(config_path)
    if validate_paths:
        if config.rebuild_input:
            config.check_paths_exist()
        else:
            config.check_regression_input_exists()
    return run_regression_suite(
        config=config,
        write_outputs=write_outputs,
        print_model_summaries=print_model_summaries or config.print_model_summaries,
    )
