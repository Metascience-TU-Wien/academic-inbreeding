# Copyright © 2026 TU Wien

"""Prepare regression data and calculate dependent-variable descriptives."""

from __future__ import annotations

import numpy as np
import pandas as pd

from acib.constants import IS_INBRED, PRIMARY_FIELD, TITLE


def prepare_regression_data(
    data: pd.DataFrame,
    *,
    exclude_titles: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Normalize regression input columns before formula fitting.

    :param data: Raw regression input data.
    :param exclude_titles: Title classes removed from the regression sample.
    :return: Data prepared for statsmodels formula fitting.
    """
    result = data.copy()
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result.loc[:, numeric_columns] = result.loc[:, numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    if exclude_titles and TITLE in result:
        result = result[~result[TITLE].isin(exclude_titles)].copy()
    for column in (TITLE, PRIMARY_FIELD):
        if column in result:
            result[column] = result[column].fillna("").astype("category")
    if IS_INBRED in result:
        result[IS_INBRED] = result[IS_INBRED].astype("category")
    return result


def model_frame(data: pd.DataFrame, *, formula: str, drop_missing: bool) -> pd.DataFrame:
    """Select and validate the columns needed for one model formula.

    :param data: Prepared regression input data.
    :param formula: Statsmodels formula used to identify required columns.
    :param drop_missing: Whether to drop rows with missing formula-column values.
    :return: Model-specific data frame containing only formula columns.
    """
    columns = required_columns_from_formula(formula)
    missing = sorted(column for column in columns if column not in data.columns)
    if missing:
        raise KeyError(f"Missing regression input column(s) for formula {formula!r}: {missing}")
    frame = data.loc[:, columns].copy()
    return frame.dropna() if drop_missing else frame


def required_columns_from_formula(formula: str) -> list[str]:
    """Extract source data-frame column names from a statsmodels formula.

    :param formula: Statsmodels formula string.
    :return: Sorted source data-frame column names required by the formula.
    """
    tokens = set()
    clean = formula.replace("C(", "(").replace("I(", "(")
    for token in (
        clean.replace("~", "+")
        .replace("**", "+")
        .replace("*", "+")
        .replace(":", "+")
        .replace("(", "+")
        .replace(")", "+")
        .split("+")
    ):
        token = token.strip()
        if token and not token.replace(".", "", 1).isdigit():
            tokens.add(token)
    return sorted(tokens)


def dependent_variable_descriptives(
    data: pd.DataFrame,
    dependent_variables: tuple[str, ...],
) -> pd.DataFrame:
    """Build mean, median, and standard-deviation rows for outcomes.

    :param data: Prepared regression data.
    :param dependent_variables: Dependent variables to summarize.
    :return: Descriptive-statistics table.
    """
    rows = []
    for dependent_variable in dependent_variables:
        series = pd.to_numeric(data[dependent_variable], errors="coerce")
        rows.append(
            {
                "dependent_variable": dependent_variable,
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(ddof=1),
                "n": int(series.notna().sum()),
            }
        )
    return pd.DataFrame(rows)
