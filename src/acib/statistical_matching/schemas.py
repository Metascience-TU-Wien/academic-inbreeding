# Copyright © 2026 TU Wien

"""Input and output artifact contracts for statistical matching."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from acib.constants import IS_INBRED, RESEARCHER_ID
from acib.regression.schemas import RegressionInputOutputSchema


class StatisticalMatchingInputSchema(RegressionInputOutputSchema):
    """Validate regression fields while allowing intentionally de-identified input."""

    sc_id: Series[Any] | None = pa.Field(
        default=None, alias=RESEARCHER_ID, nullable=False, unique=True
    )


def validate_matching_input(data: pd.DataFrame, required: set[str]) -> pd.DataFrame:
    """Validate producer fields and configured dynamic fields.

    :param data: Regression artifact contents.
    :param required: Configured fields that must be present.
    :return: Validated data, retaining extra columns.
    :raises ValueError: If a configured field is missing.
    """
    validated = StatisticalMatchingInputSchema.validate(data)
    missing = required - set(validated.columns)
    if missing:
        raise ValueError(f"Regression input schema missing configured columns: {sorted(missing)}")
    return validated


class MatchingBalanceSummarySchema(pa.DataFrameModel):
    """Contract for one balance/ESS summary per method."""

    method: Series[str] = pa.Field(nullable=False)
    estimand: Series[str] = pa.Field(nullable=False)
    n_non_inbred: Series[int] = pa.Field(nullable=False)
    n_inbred: Series[int] = pa.Field(nullable=False)
    ess_non_inbred: Series[Any] = pa.Field(nullable=False)
    ess_inbred: Series[Any] = pa.Field(nullable=False)
    max_abs_smd: Series[Any] = pa.Field(nullable=False)
    adequate_balance: Series[bool] = pa.Field(nullable=False)

    class Config:
        """Allow future diagnostic columns."""

        strict = False
        coerce = False


class MatchingCovariateBalanceSchema(pa.DataFrameModel):
    """Contract for raw and adjusted balance rows."""

    method: Series[str] = pa.Field(nullable=False)
    covariate: Series[str] = pa.Field(nullable=False)
    diff_unadjusted: Series[Any] = pa.Field(nullable=False)
    diff_adjusted: Series[Any] = pa.Field(nullable=False)
    abs_diff_adjusted: Series[Any] = pa.Field(nullable=False)
    balanced: Series[bool] = pa.Field(nullable=False)

    class Config:
        """Allow balance metadata to pass through."""

        strict = False
        coerce = False


class MatchingOutcomeEffectsSchema(pa.DataFrameModel):
    """Contract for long-form outcome effects."""

    method: Series[str] = pa.Field(nullable=False)
    estimand: Series[str] = pa.Field(nullable=False)
    outcome: Series[str] = pa.Field(nullable=False)
    n: Series[int] = pa.Field(nullable=False)
    estimate: Series[Any] = pa.Field(nullable=True)
    std_error: Series[Any] = pa.Field(nullable=True)
    p_value: Series[Any] = pa.Field(nullable=True)
    conf_low: Series[Any] = pa.Field(nullable=True)
    conf_high: Series[Any] = pa.Field(nullable=True)
    estimate_display: Series[Any] = pa.Field(nullable=True)
    std_error_display: Series[Any] = pa.Field(nullable=True)
    conf_low_display: Series[Any] = pa.Field(nullable=True)
    conf_high_display: Series[Any] = pa.Field(nullable=True)
    stars: Series[str] = pa.Field(nullable=False)

    class Config:
        """Allow additional effect compatibility columns."""

        strict = False
        coerce = False


class MatchingAdjustedMeansSchema(pa.DataFrameModel):
    """Contract for treatment-specific adjusted means."""

    method: Series[str] = pa.Field(nullable=False)
    estimand: Series[str] = pa.Field(nullable=False)
    outcome: Series[str] = pa.Field(nullable=False)
    is_inbred: Series[bool] = pa.Field(alias=IS_INBRED, nullable=False)
    n: Series[int] = pa.Field(nullable=False)
    estimate: Series[Any] = pa.Field(nullable=True)
    std_error: Series[Any] = pa.Field(nullable=True)
    conf_low: Series[Any] = pa.Field(nullable=True)
    conf_high: Series[Any] = pa.Field(nullable=True)

    class Config:
        """Allow display-scale columns and additional inference metadata."""

        strict = False
        coerce = False


class MatchingWideEffectsSchema(pa.DataFrameModel):
    """Contract for wide outcome tables with dynamic method columns."""

    outcome: Series[str] = pa.Field(nullable=False)

    class Config:
        """Allow any configured method columns."""

        strict = False
        coerce = False


class MatchingDiagnosticsSchema(pa.DataFrameModel):
    """Contract for method diagnostics with method-specific extra fields."""

    method: Series[str] = pa.Field(nullable=False)

    class Config:
        """Allow diagnostic columns specific to each method."""

        strict = False
        coerce = False


class MatchingPropensitySchema(pa.DataFrameModel):
    """Contract for pooled propensity-score rows."""

    is_inbred: Series[bool] = pa.Field(alias=IS_INBRED, nullable=False)
    propensity_score: Series[Any] = pa.Field(nullable=False)

    class Config:
        """Allow additional source score fields."""

        strict = False
        coerce = False


class MatchingPropensityByMethodSchema(pa.DataFrameModel):
    """Contract for method-level scores and weights."""

    method: Series[str] = pa.Field(nullable=False)
    is_inbred: Series[bool] = pa.Field(alias=IS_INBRED, nullable=False)
    propensity_score: Series[Any] = pa.Field(nullable=False)
    weights: Series[Any] = pa.Field(nullable=False)

    class Config:
        """Allow compatible method-specific columns."""

        strict = False
        coerce = False


def validate_output(
    name: str, frame: pd.DataFrame, *, treatment_variable: str = IS_INBRED
) -> pd.DataFrame:
    """Validate a generated table against its artifact-specific contract.

    :param name: Artifact key selecting its schema.
    :param frame: Table to validate.
    :param treatment_variable: Treatment column used by score tables.
    :return: The validated input table.
    :raises KeyError: If no schema is registered for the artifact key.
    """
    schemas: dict[str, type[pa.DataFrameModel]] = {
        "balance_summary": MatchingBalanceSummarySchema,
        "covariate_balance": MatchingCovariateBalanceSchema,
        "outcome_effects_long": MatchingOutcomeEffectsSchema,
        "adjusted_outcome_means": MatchingAdjustedMeansSchema,
        "outcome_effects_wide": MatchingWideEffectsSchema,
        "outcome_standard_errors_wide": MatchingWideEffectsSchema,
        "method_diagnostics": MatchingDiagnosticsSchema,
        "propensity_scores": MatchingPropensitySchema,
        "propensity_scores_by_method": MatchingPropensityByMethodSchema,
    }
    candidate = frame
    if name in {"propensity_scores", "propensity_scores_by_method"}:
        candidate = frame.rename(columns={treatment_variable: IS_INBRED})
    schemas[name].validate(candidate)
    return frame
