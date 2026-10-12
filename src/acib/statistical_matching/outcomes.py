# Copyright © 2026 TU Wien

"""Weighted outcome-effect estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from acib.statistical_matching.constants import PP_OUTCOMES


def estimate_outcome(
    data: pd.DataFrame,
    weights: pd.Series,
    *,
    outcome: str,
    treatment: str,
    method: str,
    estimand: str,
    covariance: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Estimate a weighted treatment effect and treatment-specific means.

    :param data: Balanced observations.
    :param weights: Weights aligned to observations.
    :param outcome: Outcome field.
    :param treatment: Treatment indicator field.
    :param method: Balancing method label.
    :param estimand: Estimand label.
    :param covariance: Robust covariance estimator.
    :return: Effect record and adjusted-mean records.
    :raises ValueError: If no outcome observations are available.
    """
    mask = data[outcome].notna()
    frame = data.loc[mask].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No non-missing observations for outcome {outcome!r}.")
    target = frame[treatment].astype(float)
    fit = (
        sm.WLS(
            frame[outcome].astype(float),
            sm.add_constant(target, has_constant="add"),
            weights=weights.loc[mask].reset_index(drop=True),
        )
        .fit()
        .get_robustcov_results(cov_type=covariance)
    )
    test = fit.t_test([0.0, 1.0])
    low, high = np.ravel(test.conf_int())
    estimate, se, p = (
        float(np.ravel(test.effect)[0]),
        float(np.ravel(test.sd)[0]),
        float(np.ravel(test.pvalue)[0]),
    )
    scale = 100.0 if outcome in PP_OUTCOMES else 1.0
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    row = {
        "method": method,
        "outcome": outcome,
        "n": len(frame),
        "estimate": estimate,
        "std_error": se,
        "p_value": p,
        "conf_low": low,
        "conf_high": high,
        "ci_lower": low,
        "ci_upper": high,
        "estimate_display": estimate * scale,
        "std_error_display": se * scale,
        "conf_low_display": low * scale,
        "conf_high_display": high * scale,
        "ci_lower_display": low * scale,
        "ci_upper_display": high * scale,
        "stars": stars,
    }
    means = []
    for is_inbred, vector in ((False, [1.0, 0.0]), (True, [1.0, 1.0])):
        result = fit.t_test(vector)
        mean, err = float(np.ravel(result.effect)[0]), float(np.ravel(result.sd)[0])
        ci = np.ravel(result.conf_int())
        means.append(
            {
                "method": method,
                "estimand": estimand,
                "outcome": outcome,
                "is_inbred": is_inbred,
                "n": len(frame),
                "estimate": mean,
                "std_error": err,
                "conf_low": ci[0],
                "conf_high": ci[1],
                "estimate_display": mean * scale,
                "std_error_display": err * scale,
                "conf_low_display": ci[0] * scale,
                "conf_high_display": ci[1] * scale,
            }
        )
    return row, means
