# Copyright © 2026 TU Wien

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy import stats

from acib.regression.config import is_ols_model_type
from acib.regression.data import required_columns_from_formula
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
    _cached_marginal_effects: list[dict[str, Any]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def model_id(self) -> str:
        """Return the stable model-specification identifier."""
        return self.spec.model_id

    @property
    def coefficients(self) -> pd.DataFrame:
        """Return raw parameters and separately typed reportable effects."""
        result = pd.DataFrame(
            coefficient_rows_from_fit(self.fit, self.dependent_variable, self.model_id)
        )
        result["effect_type"] = "coefficient"
        result["effect_units"] = self._coefficient_units
        result["model_type"] = self.model_type
        result["covariance_estimator"] = self.fit.model._acib_covariance_estimator
        result["outcome_transformation"] = self.fit.model._acib_outcome_transformation
        if self.model_type in {"frac", "ppml", "nb"}:
            effects = pd.DataFrame(self.marginal_effects())
            if not effects.empty:
                result = pd.concat([result, effects], ignore_index=True, sort=False)
        return result

    @property
    def _coefficient_units(self) -> str:
        """Return the scale on which fitted parameters are expressed."""
        if self.model_type in {"frac", "ppml", "nb", "pr", "zinb"}:
            return "link_scale"
        if self.dependent_variable == "hazen_perc_med":
            return "percentile_points"
        if self.dependent_variable.startswith("pp_top_"):
            return "percentage_points"
        return "outcome_units"

    def marginal_effects(self) -> list[dict[str, Any]]:
        """Calculate AMEs for every continuous or categorical formula predictor.

        :return: AME rows with delta-method inference and reporting metadata.
        """
        if self.model_type not in {"frac", "ppml", "nb"}:
            return []
        if self._cached_marginal_effects is not None:
            return self._cached_marginal_effects
        outcome = self.dependent_variable
        predictors = [
            column for column in required_columns_from_formula(self.formula) if column != outcome
        ]
        rows: list[dict[str, Any]] = []
        for variable in predictors:
            if variable not in self.model_data:
                continue
            series = self.model_data[variable]
            if _is_categorical_predictor(series) or _is_formula_categorical_predictor(
                self.fit, variable
            ):
                levels = list(series.dropna().unique())
                if len(levels) < 2:
                    continue
                reference = _categorical_reference(self.fit, variable, series, levels)
                for level in levels:
                    if level == reference:
                        continue
                    rows.append(
                        self._marginal_effect_row(
                            variable=variable,
                            contrast_level=level,
                            reference_level=reference,
                        )
                    )
            else:
                rows.append(self._marginal_effect_row(variable=variable))
        object.__setattr__(self, "_cached_marginal_effects", rows)
        return rows

    def _marginal_effect_row(
        self,
        *,
        variable: str,
        contrast_level: Any | None = None,
        reference_level: Any | None = None,
    ) -> dict[str, Any]:
        """Calculate one derivative or discrete category contrast.

        :param variable: Binary predictor varied in the prediction scenarios.
        :param contrast_level: Category value compared with its reference.
        :param reference_level: Reference category for a discrete contrast.
        :return: Typed AME row with delta-method inference.
        """
        high_design, low_design, effect_scale = self._effect_design_matrices(
            variable=variable,
            contrast_level=contrast_level,
            reference_level=reference_level,
        )
        params = np.asarray(self.fit.params, dtype=float)
        estimate = (
            self._effect_value(parameters=params, high=high_design, low=low_design) / effect_scale
        )
        if not np.isfinite(estimate):
            raise ValueError(
                f"Non-finite AME estimate for {self.dependent_variable} "
                f"{self.model_id} predictor {variable}"
            )
        gradient = self._effect_gradient(
            parameters=params,
            high=high_design,
            low=low_design,
            effect_scale=effect_scale,
        )
        covariance = np.asarray(self.fit.cov_params(), dtype=float)
        if not np.isfinite(gradient).all() or not np.isfinite(covariance).all():
            raise ValueError(
                f"Non-finite delta-method inputs for {self.dependent_variable} "
                f"{self.model_id} AME of {variable}"
            )
        with np.errstate(over="raise", invalid="raise"):
            variance = float(gradient @ covariance @ gradient)
        if not np.isfinite(variance):
            raise ValueError(
                f"Non-finite delta-method variance for {self.dependent_variable} "
                f"{self.model_id} AME of {variable}"
            )
        std_error = float(np.sqrt(max(variance, 0.0)))
        z_value = estimate / std_error if std_error else np.nan
        critical = float(stats.norm.ppf(0.975))
        scale = float(self.fit.model._acib_reporting_scale)
        contrast = (
            f"{contrast_level!s} vs {reference_level!s}"
            if reference_level is not None
            else "derivative"
        )
        binary_discrete = (
            reference_level is not None and len(self.model_data[variable].dropna().unique()) == 2
        )
        term = (
            f"AME[{variable}]"
            if reference_level is None or binary_discrete
            else f"AME[{variable}: {contrast}]"
        )
        return {
            "dependent_variable": self.dependent_variable,
            "model_id": self.model_id,
            "term": term,
            "estimate": estimate * scale,
            "std_error": std_error * scale,
            "p_value": float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan,
            "conf_low": (estimate - critical * std_error) * scale,
            "conf_high": (estimate + critical * std_error) * scale,
            "n_obs": len(self.model_data),
            "model_type": self.model_type,
            "effect_type": "AME",
            "effect_units": self._ame_units,
            "covariance_estimator": self.fit.model._acib_covariance_estimator,
            "outcome_transformation": self.fit.model._acib_outcome_transformation,
            "predictor": variable,
            "contrast": contrast,
        }

    @property
    def _ame_units(self) -> str:
        if self.dependent_variable == "hazen_perc_med":
            return "percentile_points"
        if self.dependent_variable.startswith("pp_top_"):
            return "percentage_points"
        if self.dependent_variable.startswith("P_top") and self.model_type == "nb":
            return "rounded_publication_count"
        return "outcome_units"

    def _effect_value(
        self,
        *,
        parameters: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> float:
        """Evaluate a prediction contrast on precomputed design matrices."""
        return self._mean_prediction(parameters, high) - self._mean_prediction(parameters, low)

    def _effect_values(
        self,
        *,
        parameters: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> np.ndarray:
        """Evaluate prediction contrasts for a batch of perturbed parameter vectors.

        :param parameters: Matrix whose columns are parameter vectors.
        :param high: Design matrix for the higher or contrasted scenario.
        :param low: Design matrix for the lower or reference scenario.
        :return: One mean prediction contrast per parameter vector.
        """
        design_columns = high.shape[1]
        high_linear = high @ parameters[:design_columns]
        low_linear = low @ parameters[:design_columns]
        if self.model_type == "nb":
            with np.errstate(over="raise", invalid="raise"):
                high_predictions = np.exp(high_linear)
                low_predictions = np.exp(low_linear)
        else:
            inverse_link = self.fit.model.family.link.inverse
            high_predictions = inverse_link(high_linear)
            low_predictions = inverse_link(low_linear)
        return np.mean(high_predictions, axis=0) - np.mean(low_predictions, axis=0)

    def _effect_gradient(
        self,
        *,
        parameters: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        effect_scale: float,
    ) -> np.ndarray:
        """Calculate the delta-method gradient, using the exact NB log-link derivative.

        :param parameters: Fitted model parameters.
        :param high: Design matrix for the higher or contrasted scenario.
        :param low: Design matrix for the lower or reference scenario.
        :param effect_scale: Difference scale for continuous-predictor derivatives.
        :return: Gradient in fitted-parameter order.
        """
        if self.model_type != "nb":
            steps = 1e-5 * np.maximum(1.0, np.abs(parameters))
            perturbations = np.diag(steps)
            upper = parameters[:, np.newaxis] + perturbations
            lower = parameters[:, np.newaxis] - perturbations
            return (
                self._effect_values(parameters=upper, high=high, low=low)
                - self._effect_values(parameters=lower, high=high, low=low)
            ) / (2 * steps * effect_scale)

        design_columns = high.shape[1]
        with np.errstate(over="raise", invalid="raise"):
            high_mean = np.exp(high @ parameters[:design_columns])
            low_mean = np.exp(low @ parameters[:design_columns])
            gradient = np.zeros_like(parameters)
            gradient[:design_columns] = (high.T @ high_mean - low.T @ low_mean) / (
                len(high) * effect_scale
            )
        return gradient

    def _effect_design_matrices(
        self,
        *,
        variable: str,
        contrast_level: Any | None,
        reference_level: Any | None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Build design matrices once for both scenarios of an AME calculation.

        :param variable: Predictor varied in the two prediction scenarios.
        :param contrast_level: Category compared with its reference, if categorical.
        :param reference_level: Reference category, if categorical.
        :return: Higher/lower design matrices and the scale divisor for their difference.
        """
        base_data = self._prediction_data()
        if reference_level is None:
            values = pd.to_numeric(base_data[variable], errors="coerce").to_numpy(float)
            step = 1e-5 * max(1.0, float(np.nanstd(values)))
            lower = base_data
            upper = lower.copy()
            lower[variable] = values - step
            upper[variable] = values + step
            high_frame, low_frame = upper, lower
            effect_scale = 2 * step
        else:
            high_frame = base_data
            low_frame = high_frame.copy()
            high_frame[variable] = contrast_level
            low_frame[variable] = reference_level
            effect_scale = 1.0
        design_info = self.fit.model._acib_design_info
        high_design = build_design_matrices([design_info], high_frame, return_type="dataframe")[0]
        low_design = build_design_matrices([design_info], low_frame, return_type="dataframe")[0]
        return (
            np.asarray(high_design, dtype=float),
            np.asarray(low_design, dtype=float),
            effect_scale,
        )

    def _prediction_data(self) -> pd.DataFrame:
        """Return the predictor frame in the coordinate system used at fit time."""
        data = self.model_data.copy()
        center = getattr(self.fit.model, "_acib_first_year_center", np.nan)
        if np.isfinite(center) and "first_year" in data:
            data["first_year"] = pd.to_numeric(data["first_year"], errors="coerce") - center
        return data

    def _mean_prediction(self, parameters: np.ndarray, design: np.ndarray) -> float:
        """Return the mean prediction for a precomputed design matrix."""
        try:
            with np.errstate(over="raise", invalid="raise"):
                predictions = self.fit.model.predict(parameters, exog=design)
        except FloatingPointError as exc:
            raise ValueError("Non-finite prediction while computing a regression AME") from exc
        if not np.isfinite(predictions).all():
            raise ValueError("Non-finite prediction while computing a regression AME")
        return float(np.mean(np.asarray(predictions, dtype=float)))

    def reportable_effect(self, term_name: str) -> dict[str, Any] | None:
        """Return the estimator-appropriate effect corresponding to a table term.

        :param term_name: Patsy term name requested by an existing table row.
        :return: Coefficient or AME record, or ``None`` when the term has no standalone effect.
        """
        if self.model_type not in {"frac", "ppml", "nb"}:
            return None
        if term_name.startswith("I("):
            return None
        categorical = re.fullmatch(r"C\(([^)]+)\)\[T\.(.*)\]", term_name)
        if categorical:
            predictor, level = categorical.groups()
            if predictor not in self.model_data:
                return None
            rows = [row for row in self.marginal_effects() if row["predictor"] == predictor]
            if len(self.model_data[predictor].dropna().unique()) == 2:
                return next((row for row in rows if row["term"] == f"AME[{predictor}]"), None)
            return next(
                (row for row in rows if row["term"].startswith(f"AME[{predictor}: {level} vs ")),
                None,
            )
        if term_name.startswith("C("):
            return None
        rows = [row for row in self.marginal_effects() if row["predictor"] == term_name]
        return rows[0] if rows else None

    def categorical_contrast(
        self, *, variable: str, level: Any, reference: Any
    ) -> dict[str, Any] | None:
        """Calculate an average discrete change against a specified category.

        :param variable: Categorical predictor to vary.
        :param level: Category whose predicted outcome is evaluated.
        :param reference: Reference category for the contrast.
        :return: AME row with delta-method inference, or ``None`` for unavailable levels.
        """
        if self.model_type not in {"frac", "ppml", "nb"}:
            return None
        if variable not in self.model_data:
            return None
        levels = set(self.model_data[variable].dropna().unique())
        if level not in levels or reference not in levels:
            return None
        return self._marginal_effect_row(
            variable=variable,
            contrast_level=level,
            reference_level=reference,
        )

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
                "covariance_estimator": self.fit.model._acib_covariance_estimator,
                "outcome_transformation": self.fit.model._acib_outcome_transformation,
                "reporting_scale": self.fit.model._acib_reporting_scale,
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


def _is_categorical_predictor(series: pd.Series) -> bool:
    """Return whether observed values represent categories rather than a numeric scale."""
    return (
        isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series.dtype)
        or pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
    )


def _is_formula_categorical_predictor(fit: Any, variable: str) -> bool:
    """Return whether Patsy encoded a source variable as a categorical factor."""
    return any(
        factor.name() == f"C({variable})"
        or (factor_info.type == "categorical" and factor.name() == variable)
        for factor, factor_info in fit.model._acib_design_info.factor_infos.items()
    )


def _categorical_reference(
    fit: Any, variable: str, series: pd.Series, observed_levels: list[Any]
) -> Any:
    """Return the treatment reference encoded by Patsy for a categorical predictor."""
    for factor, factor_info in fit.model._acib_design_info.factor_infos.items():
        factor_name = factor.name()
        if factor_name == f"C({variable})" and factor_info.categories:
            return factor_info.categories[0]
    if isinstance(series.dtype, pd.CategoricalDtype):
        categories = list(series.cat.categories)
        if categories:
            return categories[0]
    return min(observed_levels, key=str)
