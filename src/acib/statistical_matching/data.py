# Copyright © 2026 TU Wien

"""Input preparation and covariate encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
from patsy import dmatrix

from acib.constants import PRIMARY_FIELD
from acib.statistical_matching.constants import TITLE, TITLE_REFERENCE


def prepare_matching_data(
    data: pd.DataFrame, *, treatment_variable: str, exclude_titles: tuple[str, ...]
) -> pd.DataFrame:
    """Normalize treatment and matching fields in a copy of the input.

    :param data: Source observations.
    :param treatment_variable: Treatment indicator field.
    :param exclude_titles: Titles to remove before matching.
    :return: Prepared observations with a fresh index.
    :raises KeyError: If the treatment field is absent.
    """
    result = data.copy().replace([np.inf, -np.inf], np.nan)
    if exclude_titles and TITLE in result:
        result = result.loc[~result[TITLE].isin(exclude_titles)].copy()
    if treatment_variable not in result:
        raise KeyError(f"Missing treatment column: {treatment_variable}")
    result[treatment_variable] = normalize_bool_series(result[treatment_variable])
    for column in (TITLE, PRIMARY_FIELD):
        if column in result:
            result[column] = result[column].astype("category")
    return result.reset_index(drop=True)


def normalize_bool_series(series: pd.Series) -> pd.Series:
    """Parse supported booleans, numbers, and common textual spellings.

    :param series: Values to normalize.
    :return: Boolean treatment values.
    :raises ValueError: If a non-null value cannot be parsed.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.map(_bool_value)
    if normalized.isna().any():
        bad = sorted({str(value) for value in series[normalized.isna()].dropna().unique()})
        raise ValueError(f"Cannot coerce treatment values to bool: {bad}")
    return normalized.astype(bool)


def _bool_value(value: object) -> bool | float:
    """Convert one supported scalar to a boolean or missing value."""
    if value is None or value is pd.NA or value is pd.NaT:
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in {"true", "t", "1", "yes", "y"}:
            return True
        if word in {"false", "f", "0", "no", "n"}:
            return False
    return np.nan


def matching_frame(
    data: pd.DataFrame, *, treatment_variable: str, covariates: tuple[str, ...], drop_missing: bool
) -> pd.DataFrame:
    """Check required fields and optionally remove incomplete rows.

    :param data: Prepared observations.
    :param treatment_variable: Treatment field.
    :param covariates: Required matching covariates.
    :param drop_missing: Whether to exclude rows missing required values.
    :return: Matching observations retaining other source fields.
    :raises KeyError: If a required field is absent.
    """
    required = [treatment_variable, *covariates]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise KeyError(f"Missing matching input column(s): {missing}")
    result = data.copy()
    return result.dropna(subset=required).reset_index(drop=True) if drop_missing else result


def encoded_term(data: pd.DataFrame, column: str) -> str:
    """Return formula syntax for a covariate.

    :param data: Observations used to inspect the field type.
    :param column: Covariate name.
    :return: Patsy term for the covariate.
    """
    series = data[column]
    if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series):
        if column == TITLE and TITLE_REFERENCE in set(series.dropna().astype(str)):
            return f'C({column}, Treatment(reference="{TITLE_REFERENCE}"))'
        return f"C({column})"
    return column


def propensity_formula(*, treatment: str, data: pd.DataFrame, covariates: tuple[str, ...]) -> str:
    """Build the logistic treatment formula.

    :param treatment: Treatment field name.
    :param data: Observations used to determine term encoding.
    :param covariates: Formula covariates.
    :return: Patsy-compatible formula string.
    """
    return f"{treatment} ~ " + " + ".join(encoded_term(data, value) for value in covariates)


def encoded_covariates(data: pd.DataFrame, covariates: tuple[str, ...]) -> pd.DataFrame:
    """Build dummy-coded covariates for entropy balancing.

    :param data: Source observations.
    :param covariates: Fields to encode.
    :return: Numeric design matrix without an intercept column.
    """
    formula = " + ".join(encoded_term(data, value) for value in covariates)
    return dmatrix(formula, data=data, return_type="dataframe").drop(
        columns=["Intercept"], errors="ignore"
    )
