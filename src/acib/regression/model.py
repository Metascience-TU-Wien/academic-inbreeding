# Copyright © 2026 TU Wien

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

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
        return result

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
