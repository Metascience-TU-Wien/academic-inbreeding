# Copyright © 2026 TU Wien

"""Domain configuration for regression analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from acib.regression.constants import (
    DEFAULT_REGRESSION_DEPENDENT_VARIABLES,
    OUTCOME_DISPLAY_LABELS,
    REGRESSION_TITLE_EXCLUSIONS,
)
from acib.regression.specs import RegressionModelSpec, regression_model_specs

if TYPE_CHECKING:
    from acib.pipeline.config import RegressionConfig


@dataclass(frozen=True, slots=True)
class RegressionSuiteConfig:
    """Configure the analysis-domain regression suite.

    :param dependent_variables: Outcomes fitted by the suite, in fitting order.
    :param model_specs: Ordered model specifications.
    """

    dependent_variables: tuple[str, ...]
    model_specs: tuple[RegressionModelSpec, ...]
    model_types: Mapping[str, str] | None = None
    outcome_aliases: Mapping[str, str] = field(default_factory=lambda: dict(OUTCOME_DISPLAY_LABELS))
    relative_performance_outcomes: tuple[str, ...] = (
        "hazen_perc_med",
        "pp_top_10",
        "pp_top_5",
        "pp_top_1",
    )
    absolute_performance_outcomes: tuple[str, ...] = ("n_public", "P_top10", "P_top5", "P_top1")
    zinb_inflation_formulas: Mapping[str, str] | None = None
    zinb_regularization: float = 3.0
    zinb_fit_method: str = "bfgs"
    zinb_max_iterations: int = 1000
    zinb_dispersion_power: int = 2
    mle_failure_policy: str = "error"
    mle_gradient_tolerance: float = 1e-4
    covariance_type: str | None = None
    drop_missing: bool = True
    exclude_titles: tuple[str, ...] = REGRESSION_TITLE_EXCLUSIONS


def regression_suite_config_from_pipeline_config(
    config: RegressionConfig,
) -> RegressionSuiteConfig:
    """Convert the stage configuration into analysis-domain settings.

    :param config: Stage regression configuration.
    :return: Configuration consumed by the model-fitting suite.
    """
    model_specs = (
        tuple(RegressionModelSpec(spec.model_id, tuple(spec.terms)) for spec in config.model_specs)
        if config.model_specs is not None
        else tuple(regression_model_specs())
    )
    model_types = dict(config.dependent_variables) or dict.fromkeys(
        DEFAULT_REGRESSION_DEPENDENT_VARIABLES, "ols"
    )
    return RegressionSuiteConfig(
        dependent_variables=tuple(model_types),
        model_specs=model_specs,
        model_types=model_types,
        outcome_aliases=dict(config.outcome_aliases) or dict(OUTCOME_DISPLAY_LABELS),
        relative_performance_outcomes=tuple(config.relative_performance_outcomes),
        absolute_performance_outcomes=tuple(config.absolute_performance_outcomes),
        zinb_inflation_formulas=dict(config.zinb_inflation_formulas),
        zinb_regularization=config.zinb_regularization,
        zinb_fit_method=config.zinb_fit_method,
        zinb_max_iterations=config.zinb_max_iterations,
        zinb_dispersion_power=config.zinb_dispersion_power,
        mle_failure_policy=config.mle_failure_policy,
        mle_gradient_tolerance=config.mle_gradient_tolerance,
        covariance_type=config.covariance_type,
        drop_missing=config.drop_missing,
        exclude_titles=tuple(config.exclude_titles),
    )
