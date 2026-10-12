# Copyright © 2026 TU Wien

"""Configuration models for statistical matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from acib.regression.constants import DEFAULT_REGRESSION_OUTCOME_METADATA
from acib.statistical_matching.constants import (
    ADJUSTED_OUTCOME_MEANS_FILENAME,
    ATT_ESTIMAND,
    BALANCE_SUMMARY_FILENAME,
    BALANCE_THRESHOLD,
    COVARIATE_BALANCE_FILENAME,
    DEFAULT_CEM_CUTPOINTS,
    DEFAULT_COVARIATES,
    DEFAULT_METHODS,
    DEFAULT_OUTCOMES,
    DEFAULT_TITLE_EXCLUSIONS,
    LATEX_DIRECTORY_NAME,
    MATCHING_FORMULAS_FILENAME,
    METHOD_DIAGNOSTICS_FILENAME,
    OUTCOME_EFFECTS_LONG_FILENAME,
    OUTCOME_EFFECTS_WIDE_FILENAME,
    OUTCOME_STANDARD_ERRORS_WIDE_FILENAME,
    PLOTS_DIRECTORY_NAME,
    PROPENSITY_SCORES_BY_METHOD_FILENAME,
    PROPENSITY_SCORES_FILENAME,
    TREATED_VALUE,
    TREATMENT_VARIABLE,
)


@dataclass(frozen=True, slots=True)
class StatisticalMatchingOutputs:
    """Output directory owned by statistical matching."""

    directory: Path

    @property
    def balance_summary(self) -> Path:
        """Return the balance-summary artifact path."""
        return self.directory / BALANCE_SUMMARY_FILENAME

    @property
    def covariate_balance(self) -> Path:
        """Return the covariate-balance artifact path."""
        return self.directory / COVARIATE_BALANCE_FILENAME

    @property
    def outcome_effects_long(self) -> Path:
        """Return the long outcome-effects artifact path."""
        return self.directory / OUTCOME_EFFECTS_LONG_FILENAME

    @property
    def adjusted_outcome_means(self) -> Path:
        """Return the adjusted-outcome-means artifact path."""
        return self.directory / ADJUSTED_OUTCOME_MEANS_FILENAME

    @property
    def outcome_effects_wide(self) -> Path:
        """Return the wide outcome-effects artifact path."""
        return self.directory / OUTCOME_EFFECTS_WIDE_FILENAME

    @property
    def outcome_standard_errors_wide(self) -> Path:
        """Return the wide outcome-standard-errors artifact path."""
        return self.directory / OUTCOME_STANDARD_ERRORS_WIDE_FILENAME

    @property
    def method_diagnostics(self) -> Path:
        """Return the method-diagnostics artifact path."""
        return self.directory / METHOD_DIAGNOSTICS_FILENAME

    @property
    def propensity_scores(self) -> Path:
        """Return the propensity-scores artifact path."""
        return self.directory / PROPENSITY_SCORES_FILENAME

    @property
    def propensity_scores_by_method(self) -> Path:
        """Return the per-method propensity-scores artifact path."""
        return self.directory / PROPENSITY_SCORES_BY_METHOD_FILENAME

    @property
    def matching_formulas(self) -> Path:
        """Return the matching-formulas report path."""
        return self.directory / MATCHING_FORMULAS_FILENAME

    @property
    def latex_dir(self) -> Path:
        """Return the LaTeX report directory."""
        return self.directory / LATEX_DIRECTORY_NAME

    @property
    def plots_dir(self) -> Path:
        """Return the plot directory."""
        return self.directory / PLOTS_DIRECTORY_NAME

    def all_paths(self) -> tuple[Path, ...]:
        """Return every configured output path.

        :return: Paths in dataclass field order.
        """
        return (
            self.balance_summary,
            self.covariate_balance,
            self.outcome_effects_long,
            self.adjusted_outcome_means,
            self.outcome_effects_wide,
            self.outcome_standard_errors_wide,
            self.method_diagnostics,
            self.propensity_scores,
            self.propensity_scores_by_method,
            self.matching_formulas,
            self.latex_dir,
            self.plots_dir,
        )


@dataclass(frozen=True, slots=True)
class StatisticalMatchingConfig:
    """Resolved upstream input, settings, and output contracts."""

    input_path: Path
    outputs: StatisticalMatchingOutputs
    outcomes: tuple[str, ...] = DEFAULT_OUTCOMES
    outcome_types: dict[str, str] = field(
        default_factory=lambda: {
            name: metadata[0] for name, metadata in DEFAULT_REGRESSION_OUTCOME_METADATA.items()
        }
    )
    outcome_aliases: dict[str, str] = field(
        default_factory=lambda: {
            name: metadata[1] for name, metadata in DEFAULT_REGRESSION_OUTCOME_METADATA.items()
        }
    )
    covariates: tuple[str, ...] = DEFAULT_COVARIATES
    methods: tuple[str, ...] = DEFAULT_METHODS
    treatment_variable: str = TREATMENT_VARIABLE
    treated_value: bool = TREATED_VALUE
    estimand: str = ATT_ESTIMAND
    exclude_titles: tuple[str, ...] = DEFAULT_TITLE_EXCLUSIONS
    cem_cutpoints: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            key: tuple(values) for key, values in DEFAULT_CEM_CUTPOINTS.items()
        }
    )
    balance_threshold: float = BALANCE_THRESHOLD
    covariance_type: str = "HC3"
    drop_missing: bool = True

    def __post_init__(self) -> None:
        """Reject invalid thresholds, selections, or output paths.

        :raises ValueError: If settings are ambiguous or unsupported.
        """
        if not self.balance_threshold > 0:
            raise ValueError("Matching balance_threshold must be positive")
        if len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError("Selected matching outcomes must be unique")
        if len(set(self.methods)) != len(self.methods) or not set(self.methods) <= set(
            DEFAULT_METHODS
        ):
            raise ValueError("Matching methods must be unique and supported")
        paths = self.outputs.all_paths()
        if len(set(paths)) != len(paths):
            raise ValueError("Statistical-matching output paths must be distinct")

    def check_input_exists(self) -> None:
        """Require the producer-owned regression artifact to be present.

        :raises FileNotFoundError: If the configured input file is missing.
        """
        if not self.input_path.is_file():
            raise FileNotFoundError(
                "Missing generated artifact from producer regression: "
                f"[regression.outputs].regression_input: {self.input_path}"
            )
