# Copyright © 2026 TU Wien

"""Covariate-balance diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from acib.constants import DOMAIN_SHARE_COLUMNS, FIELD_SHARE_COLUMNS


def _matrix(data: pd.DataFrame, covariates: tuple[str, ...]) -> pd.DataFrame:
    """Encode configured covariates as numeric balance columns.

    :param data: Source observations.
    :param covariates: Fields to encode.
    :return: Numeric design matrix aligned to the input index.
    """
    columns = list(covariates)
    if set(covariates) & set(DOMAIN_SHARE_COLUMNS):
        columns.extend(c for c in DOMAIN_SHARE_COLUMNS if c in data and c not in columns)
    if set(covariates) & set(FIELD_SHARE_COLUMNS):
        columns.extend(c for c in FIELD_SHARE_COLUMNS if c in data and c not in columns)
    encoded: dict[str, pd.Series] = {}
    for column in columns:
        if column not in data:
            continue
        values = data[column]
        if isinstance(values.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(values):
            categories = (
                values.cat.categories
                if isinstance(values.dtype, pd.CategoricalDtype)
                else sorted(values.dropna().unique())
            )
            for category in categories:
                label = str(category)
                encoded[f"{column}[{label}]"] = values.astype(str).eq(label).astype(float)
        else:
            encoded[column] = pd.to_numeric(values, errors="coerce").astype(float)
    return pd.DataFrame(encoded, index=data.index)


def effective_sample_size(weights: pd.Series) -> float:
    """Calculate effective sample size from weights.

    :param weights: Observation weights.
    :return: Squared-sum effective sample size.
    """
    values = weights.to_numpy(dtype=float)
    return float(values.sum() ** 2 / np.square(values).sum()) if np.square(values).sum() else 0.0


def covariate_balance(
    original: pd.DataFrame,
    adjusted: pd.DataFrame,
    weights: pd.Series,
    *,
    treatment: str,
    covariates: tuple[str, ...],
    method: str,
    threshold: float,
) -> pd.DataFrame:
    """Calculate original and adjusted standardized mean differences.

    :param original: Unadjusted observations.
    :param adjusted: Matched or weighted observations.
    :param weights: Weights aligned to adjusted observations.
    :param treatment: Treatment indicator column.
    :param covariates: Covariates to assess.
    :param method: Balancing method label.
    :param threshold: Maximum absolute difference for balance.
    :return: One balance diagnostic row per encoded covariate.
    """
    baseline = _matrix(original, covariates)
    adjusted_values = _matrix(adjusted, covariates).reindex(
        columns=baseline.columns, fill_value=0.0
    )
    t0 = original[treatment].astype(bool)
    t1 = adjusted[treatment].astype(bool)
    rows = []
    for column in baseline:
        values = baseline[column]
        pooled = np.sqrt((values[t0].var(ddof=0) + values[~t0].var(ddof=0)) / 2)
        if pooled == 0 or np.isnan(pooled):
            before = after = 0.0
        else:
            before = (values[t0].mean() - values[~t0].mean()) / pooled
            treated_mean = np.average(adjusted_values.loc[t1, column], weights=weights.loc[t1])
            control_mean = np.average(adjusted_values.loc[~t1, column], weights=weights.loc[~t1])
            after = (treated_mean - control_mean) / pooled
        rows.append(
            {
                "method": method,
                "covariate": column,
                "diff_unadjusted": before,
                "diff_adjusted": after,
                "abs_diff_adjusted": abs(after),
                "balanced": abs(after) < threshold,
            }
        )
    return pd.DataFrame(rows)
