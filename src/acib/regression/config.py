# Copyright © 2026 TU Wien

"""Stage-owned configuration for the regression pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from acib.regression.constants import (
    AIC_BIC_COMPARISON_FILENAME,
    AIC_BIC_RANKED_FILENAME,
    ASJC_FIELD_SHARE_DIAGNOSTICS_FILENAME,
    AUTHOR_REGRESSION_INPUT_FILENAME,
    BEST_MODELS_BY_AIC_FILENAME,
    BEST_MODELS_BY_BIC_FILENAME,
    DEFAULT_REGRESSION_OUTCOME_METADATA,
    DEPENDENT_VARIABLE_DESCRIPTIVES_FILENAME,
    INBREEDING_DISCIPLINE_SPEC_COMPARISON_FILENAME,
    LATEX_DIRECTORY_NAME,
    MODEL_COEFFICIENTS_FILENAME,
    MODEL_FIT_STATISTICS_FILENAME,
    MODEL_FORMULAS_FILENAME,
    MODEL_SUMMARIES_FILENAME,
    PLOTS_DIRECTORY_NAME,
    PROFESSOR_ARTICLE_DIAGNOSTICS_FILENAME,
    PROFESSOR_ARTICLE_INPUT_FILENAME,
    REGRESSION_INPUT_DIAGNOSTICS_FILENAME,
    ROBUSTNESS_COEFFICIENTS_FILENAME,
    ROBUSTNESS_FIT_STATISTICS_FILENAME,
)

RegressionModelType = Literal[
    "ols", "olshc3", "pr", "nb", "nbhc3", "zinb", "huber", "fractional_logit", "ppml"
]
RegressionOutcomeType = Literal["relative", "absolute"]
OLS_MODEL_TYPES = frozenset({"ols", "olshc3"})
NEGATIVE_BINOMIAL_MODEL_TYPES = frozenset({"nb", "nbhc3"})


def is_ols_model_type(model_type: str) -> bool:
    """Return whether a configured model uses the linear OLS estimator."""
    return model_type in OLS_MODEL_TYPES


@dataclass(frozen=True, slots=True)
class RegressionOutcomeMetadata:
    """Describe the report grouping and label for one model outcome.

    :param outcome_type: Whether the dependent variable is relative or absolute.
    :param alias: Human-readable label used in regression reports.
    """

    outcome_type: RegressionOutcomeType
    alias: str


@dataclass(frozen=True, slots=True)
class RegressionModelConfig:
    """Configure a single ordered regression model specification.

    :param model_id: Stable model identifier, such as ``m9``.
    :param terms: Ordered formula terms or field-share tokens.
    """

    model_id: str
    terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RegressionOutputs:
    """Derive fixed regression artifact paths from their output directory.

    :param directory: Root directory for all regression-generated artifacts.
    """

    directory: Path

    @property
    def regression_input(self) -> Path:
        """Return the researcher-level input consumed by later analyses."""
        return self.directory / AUTHOR_REGRESSION_INPUT_FILENAME

    @property
    def professor_article_input(self) -> Path:
        """Return the prepared professor/publication records path."""
        return self.directory / PROFESSOR_ARTICLE_INPUT_FILENAME

    @property
    def professor_article_diagnostics(self) -> Path:
        """Return the professor/publication build diagnostics path."""
        return self.directory / PROFESSOR_ARTICLE_DIAGNOSTICS_FILENAME

    @property
    def asjc_field_share_diagnostics(self) -> Path:
        """Return the ASJC field-share diagnostics path."""
        return self.directory / ASJC_FIELD_SHARE_DIAGNOSTICS_FILENAME

    @property
    def regression_input_diagnostics(self) -> Path:
        """Return the researcher-level input diagnostics path."""
        return self.directory / REGRESSION_INPUT_DIAGNOSTICS_FILENAME

    @property
    def model_coefficients(self) -> Path:
        """Return the long-form model coefficients path."""
        return self.directory / MODEL_COEFFICIENTS_FILENAME

    @property
    def robustness_model_coefficients(self) -> Path:
        """Return coefficients for configured non-OLS robustness models."""
        return self.directory / ROBUSTNESS_COEFFICIENTS_FILENAME

    @property
    def robustness_model_fit_statistics(self) -> Path:
        """Return fit statistics for configured non-OLS robustness models."""
        return self.directory / ROBUSTNESS_FIT_STATISTICS_FILENAME

    @property
    def model_fit_statistics(self) -> Path:
        """Return the model fit and convergence statistics path."""
        return self.directory / MODEL_FIT_STATISTICS_FILENAME

    @property
    def dependent_variable_descriptives(self) -> Path:
        """Return the dependent-variable descriptives path."""
        return self.directory / DEPENDENT_VARIABLE_DESCRIPTIVES_FILENAME

    @property
    def aic_bic_comparison(self) -> Path:
        """Return the AIC/BIC comparison path."""
        return self.directory / AIC_BIC_COMPARISON_FILENAME

    @property
    def aic_bic_ranked(self) -> Path:
        """Return the ranked AIC/BIC comparison path."""
        return self.directory / AIC_BIC_RANKED_FILENAME

    @property
    def best_models_by_aic(self) -> Path:
        """Return the best-models-by-AIC path."""
        return self.directory / BEST_MODELS_BY_AIC_FILENAME

    @property
    def best_models_by_bic(self) -> Path:
        """Return the best-models-by-BIC path."""
        return self.directory / BEST_MODELS_BY_BIC_FILENAME

    @property
    def inbreeding_discipline_spec_comparison(self) -> Path:
        """Return the discipline specification comparison path."""
        return self.directory / INBREEDING_DISCIPLINE_SPEC_COMPARISON_FILENAME

    @property
    def model_formulas(self) -> Path:
        """Return the fitted formula report path."""
        return self.directory / MODEL_FORMULAS_FILENAME

    @property
    def model_summaries(self) -> Path:
        """Return the fitted model summaries path."""
        return self.directory / MODEL_SUMMARIES_FILENAME

    @property
    def latex_dir(self) -> Path:
        """Return the collection directory for generated LaTeX tables."""
        return self.directory / LATEX_DIRECTORY_NAME

    @property
    def plots_dir(self) -> Path:
        """Return the collection directory for regression plots."""
        return self.directory / PLOTS_DIRECTORY_NAME

    def all_paths(self) -> tuple[Path, ...]:
        """Return all configured regression output paths.

        :return: Fixed output files followed by the LaTeX and plot collection roots.
        """
        return (
            self.regression_input,
            self.professor_article_input,
            self.professor_article_diagnostics,
            self.asjc_field_share_diagnostics,
            self.regression_input_diagnostics,
            self.model_coefficients,
            self.robustness_model_coefficients,
            self.robustness_model_fit_statistics,
            self.model_fit_statistics,
            self.dependent_variable_descriptives,
            self.aic_bic_comparison,
            self.aic_bic_ranked,
            self.best_models_by_aic,
            self.best_models_by_bic,
            self.inbreeding_discipline_spec_comparison,
            self.model_formulas,
            self.model_summaries,
            self.latex_dir,
            self.plots_dir,
        )


@dataclass(frozen=True, slots=True)
class RegressionConfig:
    """Resolved upstream/external inputs and scientific regression settings.

    :param article_author_file_path: External article-author dataset.
    :param mobility_author_proxy: Author-level artifact produced by mobility.
    :param outputs: Explicit regression-owned artifact paths.
    :param article_impact_file_path: Optional external article impact dataset.
    :param disruption_indicator_file_path: Optional external disruption indicators.
    :param dependent_variables: Outcome names and their estimator families.
    :param outcome_metadata: Outcome groupings and report labels owned by regression.
    :param model_specs: Optional ordered model terms; defaults to the source's ten models.
    :param zinb_inflation_formulas: Optional ZINB inflation formulas by outcome.
    :param zinb_regularization: Regularization strength used to initialize the inflation model.
    :param zinb_fit_method: ZINB optimizer method.
    :param zinb_max_iterations: ZINB iteration limit.
    :param zinb_dispersion_power: NB variance power used by ZINB.
    :param mle_failure_policy: Whether diagnostic failures raise or warn.
    :param mle_gradient_tolerance: Maximum per-observation score norm.
    :param covariance_type: Optional statsmodels covariance estimator.
    :param drop_missing: Drop rows missing values required by a model formula.
    :param min_publications: Minimum publication threshold used for sample selection.
    :param min_first_year: Optional inclusive first-publication-year cutoff.
    :param professor_title_pattern: Case-insensitive professor-title selection pattern.
    :param exclude_titles: Title classes excluded from model fitting.
    :param rebuild_input: Rebuild regression artifacts from configured stage inputs.
    :param print_model_summaries: Print fitted summaries to standard output.
    """

    article_author_file_path: Path
    mobility_author_proxy: Path
    outputs: RegressionOutputs
    article_impact_file_path: Path | None = None
    disruption_indicator_file_path: Path | None = None
    dependent_variables: Mapping[str, RegressionModelType] = field(default_factory=dict)
    outcome_metadata: Mapping[str, RegressionOutcomeMetadata] = field(
        default_factory=lambda: {
            outcome: RegressionOutcomeMetadata(
                outcome_type=cast(RegressionOutcomeType, outcome_type), alias=alias
            )
            for outcome, (outcome_type, alias) in DEFAULT_REGRESSION_OUTCOME_METADATA.items()
        }
    )
    model_specs: tuple[RegressionModelConfig, ...] | None = None
    zinb_inflation_formulas: Mapping[str, str] = field(default_factory=dict)
    zinb_regularization: float = 3.0
    zinb_fit_method: str = "bfgs"
    zinb_max_iterations: int = 1000
    zinb_dispersion_power: int = 2
    mle_failure_policy: Literal["error", "warn"] = "error"
    mle_gradient_tolerance: float = 1e-4
    covariance_type: str | None = None
    drop_missing: bool = True
    min_publications: int = 4
    min_first_year: int | None = None
    professor_title_pattern: str = "prof"
    exclude_titles: tuple[str, ...] = ("FH-Ass. Prof.",)
    rebuild_input: bool = True
    print_model_summaries: bool = False

    def __post_init__(self) -> None:
        """Validate ranges without altering configured values.

        :raises ValueError: If model options or outputs are invalid.
        """
        if self.min_publications < 1:
            raise ValueError("min_publications must be at least one")
        if self.min_first_year is not None and self.min_first_year < 0:
            raise ValueError("min_first_year must be nonnegative or None")
        if self.zinb_regularization < 0:
            raise ValueError("zinb_regularization must be nonnegative")
        if self.zinb_max_iterations < 1:
            raise ValueError("zinb_max_iterations must be positive")
        if self.zinb_dispersion_power < 1:
            raise ValueError("zinb_dispersion_power must be positive")
        if self.mle_gradient_tolerance <= 0:
            raise ValueError("mle_gradient_tolerance must be positive")
        if self.professor_title_pattern == "":
            raise ValueError("professor_title_pattern must not be empty")
        if len(set(self.outputs.all_paths())) != len(self.outputs.all_paths()):
            raise ValueError("Regression output paths must be distinct")
        model_ids = [spec.model_id for spec in self.model_specs or ()]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Regression model IDs must be unique")
        missing_metadata = set(self.dependent_variables) - self.outcome_metadata.keys()
        if missing_metadata:
            raise ValueError(
                "Missing regression outcome metadata for: " + ", ".join(sorted(missing_metadata))
            )

    @property
    def relative_performance_outcomes(self) -> tuple[str, ...]:
        """Return configured outcomes classified as relative performance measures."""
        return tuple(
            outcome
            for outcome, metadata in self.outcome_metadata.items()
            if metadata.outcome_type == "relative"
        )

    @property
    def absolute_performance_outcomes(self) -> tuple[str, ...]:
        """Return configured outcomes classified as absolute performance measures."""
        return tuple(
            outcome
            for outcome, metadata in self.outcome_metadata.items()
            if metadata.outcome_type == "absolute"
        )

    @property
    def outcome_aliases(self) -> dict[str, str]:
        """Return configured display aliases for model outcomes."""
        return {outcome: metadata.alias for outcome, metadata in self.outcome_metadata.items()}

    @property
    def author_regression_input_path(self) -> Path:
        """Return the regression artifact consumed by downstream analysis stages."""
        return self.outputs.regression_input

    @property
    def professor_article_output_path(self) -> Path:
        """Return the regression-owned professor-publication artifact path."""
        return self.outputs.professor_article_input

    def check_paths_exist(self) -> None:
        """Validate paths needed to rebuild the regression input.

        :raises FileNotFoundError: If an external input or upstream mobility artifact is absent.
        """
        paths = (
            ("[data].article_author_file_path", "external dataset", self.article_author_file_path),
            (
                "[mobility.outputs].author_inbreeding_proxy",
                "mobility",
                self.mobility_author_proxy,
            ),
            ("[data].article_impact_file_path", "external dataset", self.article_impact_file_path),
            (
                "[data].disruption_indicator_file_path",
                "external dataset",
                self.disruption_indicator_file_path,
            ),
        )
        missing = [
            f"{key} (producer: {producer}): {path}"
            for key, producer, path in paths
            if path is not None and not path.is_file()
        ]
        if missing:
            raise FileNotFoundError("Missing regression inputs:\n- " + "\n- ".join(missing))

    def check_regression_input_exists(self) -> None:
        """Verify the existing regression artifact when rebuilding is disabled.

        :raises FileNotFoundError: If the configured regression input is absent.
        """
        if not self.outputs.regression_input.is_file():
            raise FileNotFoundError(
                "Missing generated regression artifact from producer regression: "
                f"[regression.outputs].regression_input: {self.outputs.regression_input}"
            )

    def load_inputs(self) -> LoadedRegressionInputs:
        """Load a previously generated researcher-level regression input.

        :return: Regression data wrapped for pipeline consumers.
        :raises FileNotFoundError: If the configured regression artifact is absent.
        """
        from acib.regression.input import read_table

        self.check_regression_input_exists()
        return LoadedRegressionInputs(data=read_table(self.outputs.regression_input))


def load_regression_config(path: str | Path) -> RegressionConfig:
    """Load regression settings and resolve shared/upstream inputs.

    :param path: Shared pipeline TOML file.
    :return: Resolved regression configuration.
    """
    from acib.pipeline.config import load_regression_config as load_config

    return load_config(path)


@dataclass(frozen=True, slots=True)
class LoadedRegressionInputs:
    """Loaded regression artifact returned to pipeline consumers.

    :param data: Researcher-level regression input table.
    """

    data: pd.DataFrame
