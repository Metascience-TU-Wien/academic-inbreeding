# Copyright © 2026 TU Wien

"""Configuration models for statistical matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from acib.regression.constants import DEFAULT_REGRESSION_OUTCOME_METADATA
from acib.statistical_matching.constants import (
    ATT_ESTIMAND,
    BALANCE_THRESHOLD,
    DEFAULT_CEM_CUTPOINTS,
    DEFAULT_COVARIATES,
    DEFAULT_METHODS,
    DEFAULT_OUTCOMES,
    DEFAULT_TITLE_EXCLUSIONS,
    TREATED_VALUE,
    TREATMENT_VARIABLE,
)


@dataclass(frozen=True, slots=True)
class StatisticalMatchingOutputs:
    """Explicit files and collection roots owned by matching."""

    balance_summary: Path
    covariate_balance: Path
    outcome_effects_long: Path
    adjusted_outcome_means: Path
    outcome_effects_wide: Path
    outcome_standard_errors_wide: Path
    method_diagnostics: Path
    propensity_scores: Path
    propensity_scores_by_method: Path
    matching_formulas: Path
    latex_dir: Path
    plots_dir: Path

    def all_paths(self) -> tuple[Path, ...]:
        """Return every configured output path.

        :return: Paths in dataclass field order.
        """
        return tuple(getattr(self, name) for name in self.__dataclass_fields__)


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
