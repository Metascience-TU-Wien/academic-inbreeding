# Copyright © 2026 TU Wien

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy import stats

from acib.constants import HAZEN_PERCENTILE, IS_INBRED
from acib.regression.config import is_ols_model_type
from acib.regression.models import (
    average_comparison_for_binary_indicator,
    average_predictions_for_binary_indicator,
    coefficient_rows_from_fit,
    fit_row_from_fit,
    format_regression_summary,
)
from acib.regression.specs import RegressionModelSpec


@dataclass(frozen=True, slots=True)
class FittedRegressionModel:
    """Store a fitted regression result and expose its reportable outputs.

    :param dependent_variable: Outcome modeled by the fit.
    :param spec: Formula specification associated with the fit.
    :param formula: Formula used to create the fitted model.
    :param fit: Statsmodels fitted result.
    :param model_data: Rows and columns supplied to the fit.
    :param mle_diagnostics: Optional maximum-likelihood diagnostics.
    :param model_type: Model family label.
    :param inflation_formula: Optional zero-inflation formula.
    :param field_share_reference: Omitted field-share reference.
    :param share_control_type: Label for the share-control scheme.
    """

    dependent_variable: str
    spec: RegressionModelSpec
    formula: str
    fit: Any
    model_data: pd.DataFrame
    mle_diagnostics: dict[str, Any] | None = None
    model_type: str = "ols"
    inflation_formula: str | None = None
    field_share_reference: str | None = None
    share_control_type: str | None = None

    @property
    def model_id(self) -> str:
        """Return the stable model-specification identifier."""
        return self.spec.model_id

    @property
    def coefficients(self) -> pd.DataFrame:
        """Return coefficient estimates and their model-family label."""
        result = pd.DataFrame(
            coefficient_rows_from_fit(self.fit, self.dependent_variable, self.model_id)
        )
        result["model_type"] = self.model_type
        if self.model_type in {"fractional_logit", "ppml"}:
            result = pd.concat(
                [result, pd.DataFrame([self.average_marginal_effect(IS_INBRED)])],
                ignore_index=True,
            )
        return result

    def average_marginal_effect(self, variable: str) -> dict[str, Any]:
        """Estimate the discrete average effect of a binary variable.

        :param variable: Binary predictor varied in the prediction scenarios.
        :return: Coefficient-table row with delta-method inference.
        :raises ValueError: If the predictor is absent.
        """
        if variable not in self.model_data:
            raise ValueError(f"Marginal-effect variable {variable!r} is absent")
        scenarios = []
        for value in (False, True):
            frame = self.model_data.copy()
            frame[variable] = value
            scenarios.append(
                np.asarray(
                    build_design_matrices(
                        [self.fit.model._acib_design_info], frame, return_type="dataframe"
                    )[0],
                    dtype=float,
                )
            )
        params = np.asarray(self.fit.params, dtype=float)
        predict = self.fit.model.predict

        def mean_contrast(parameters: np.ndarray) -> float:
            high = np.asarray(predict(parameters, exog=scenarios[1]), dtype=float)
            low = np.asarray(predict(parameters, exog=scenarios[0]), dtype=float)
            return float(np.mean(high - low))

        estimate = mean_contrast(params)
        gradient = np.empty_like(params)
        for index in range(len(params)):
            step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(params[index]))
            upper = params.copy()
            lower = params.copy()
            upper[index] += step
            lower[index] -= step
            gradient[index] = (mean_contrast(upper) - mean_contrast(lower)) / (2 * step)
        covariance = np.asarray(self.fit.cov_params(), dtype=float)
        variance = float(gradient @ covariance @ gradient)
        std_error = float(np.sqrt(max(variance, 0.0)))
        z_value = estimate / std_error if std_error else np.nan
        critical = float(stats.norm.ppf(0.975))
        scale = 100.0 if self.dependent_variable == HAZEN_PERCENTILE else 1.0
        return {
            "dependent_variable": self.dependent_variable,
            "model_id": self.model_id,
            "term": f"AME[{variable}]",
            "estimate": estimate * scale,
            "std_error": std_error * scale,
            "p_value": float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan,
            "conf_low": (estimate - critical * std_error) * scale,
            "conf_high": (estimate + critical * std_error) * scale,
            "n_obs": len(self.model_data),
            "model_type": self.model_type,
        }

    @property
    def fit_statistics(self) -> dict[str, Any]:
        """Return fit statistics, design diagnostics, and MLE diagnostics."""
        row = fit_row_from_fit(
            self.fit,
            self.dependent_variable,
            self.model_id,
            self.formula,
        )
        exog = self.fit.model.exog
        row.update(
            {
                "design_columns": int(exog.shape[1]),
                "design_rank": int(np.linalg.matrix_rank(exog)),
                "design_full_rank": bool(np.linalg.matrix_rank(exog) == exog.shape[1]),
                "field_share_reference": self.field_share_reference,
                "share_control_type": self.share_control_type,
                "model_type": self.model_type,
                "inflation_formula": self.inflation_formula,
            }
        )
        if self.mle_diagnostics:
            row.update(self.mle_diagnostics)
        return row

    @property
    def summary_text(self) -> str:
        """Return a formatted summary of the fitted model."""
        summary = format_regression_summary(
            fit=self.fit,
            data=self.model_data,
            dependent_variable=self.dependent_variable,
            model_id=self.model_id,
            formula=self.formula,
            model_type=self.model_type,
            inflation_formula=self.inflation_formula,
        )
        if self.mle_diagnostics:
            summary += "\n\nMLE optimizer diagnostics:\n" + "\n".join(
                f"{key}: {value}" for key, value in self.mle_diagnostics.items()
            )
        return summary

    def average_predictions(self, variable: str) -> pd.DataFrame:
        """Return average predictions for both values of a binary variable.

        :param variable: Binary indicator varied in the prediction scenarios.
        :return: Prediction rows, or an empty frame for non-OLS models.
        """
        if not is_ols_model_type(self.model_type):
            return pd.DataFrame()
        return pd.DataFrame(
            average_predictions_for_binary_indicator(
                fit=self.fit,
                data=self.model_data,
                variable=variable,
            )
        )

    def average_comparison(self, variable: str) -> pd.DataFrame:
        """Return the estimated true-minus-false average comparison.

        :param variable: Binary indicator used for the comparison.
        :return: A one-row comparison frame, or an empty frame when unavailable.
        """
        if not is_ols_model_type(self.model_type):
            return pd.DataFrame()
        row = average_comparison_for_binary_indicator(
            fit=self.fit,
            data=self.model_data,
            variable=variable,
        )
        return pd.DataFrame([row]) if row is not None else pd.DataFrame()
