# Copyright © 2026 TU Wien

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats

from acib.regression.config import RegressionOutputs
from acib.regression.constants import (
    DEFAULT_REGRESSION_DEPENDENT_VARIABLES,
)
from acib.regression.data import (
    model_frame,
    prepare_regression_data,
    required_columns_from_formula,
)
from acib.regression.specs import RegressionModelSpec, regression_model_specs

__all__ = [
    "RegressionModelSpec",
    "fit_regression_model",
    "fit_regression_models",
    "regression_model_specs",
    "required_columns_from_formula",
]

_logger = structlog.getLogger(__name__)

DEFAULT_DEPENDENT_VARIABLES = list(DEFAULT_REGRESSION_DEPENDENT_VARIABLES)


def fit_regression_models(
    *,
    data: pd.DataFrame,
    dependent_variables: list[str] | None = None,
    model_specs: list[RegressionModelSpec] | None = None,
    covariance_type: str | None = None,
    drop_missing: bool = True,
    model_types: dict[str, str] | None = None,
    model_type: str = "ols",
    zinb_inflation_formulas: dict[str, str] | None = None,
    mle_failure_policy: str = "error",
    mle_gradient_tolerance: float = 1e-4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """
    Fit configured regression model families and collect model outputs.

    :param data: Regression input data with one row per author.
    :param dependent_variables: Dependent variables to fit against each model specification.
    :param model_specs: Optional model specifications; built-in defaults are used when omitted.
    :param covariance_type: Optional statsmodels covariance type, e.g. ``HC3``.
    :param drop_missing: Whether to drop rows with missing formula-column values before fitting.
    :param model_types: Optional model family by dependent variable; defaults to OLS.
    :param model_type: Default family for outcomes not listed in ``model_types``.
    :param zinb_inflation_formulas: Optional ZINB inflation RHS by dependent variable.
    :param mle_failure_policy: ``error`` to reject failed MLE fits or ``warn`` to retain them.
    :param mle_gradient_tolerance: Maximum acceptable per-observation score infinity norm.
    :return: Coefficient table, fit-statistics table, and fitted formula mapping.
    """
    dependent_variables = dependent_variables or DEFAULT_DEPENDENT_VARIABLES
    coefficient_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    formulas: dict[str, str] = {}
    prepared = prepare_regression_data(data)
    model_specs = model_specs or regression_model_specs()
    for dependent_variable in dependent_variables:
        outcome_model_type = (model_types or {}).get(dependent_variable, model_type)
        inflation_formula = (zinb_inflation_formulas or {}).get(dependent_variable, "1")
        for spec in model_specs:
            formula = spec.formula(dependent_variable, data=prepared)
            formulas[f"{dependent_variable}_{spec.model_id}"] = formula
            model_data = model_frame(prepared, formula=formula, drop_missing=drop_missing)
            if outcome_model_type == "zinb":
                inflation_columns = required_columns_from_formula(inflation_formula)
                missing = sorted(column for column in inflation_columns if column not in prepared)
                if missing:
                    raise KeyError(
                        f"Missing regression input column(s) for ZINB inflation formula "
                        f"{inflation_formula!r}: {missing}"
                    )
                for column in inflation_columns:
                    if column not in model_data:
                        model_data[column] = prepared.loc[model_data.index, column]
                if drop_missing:
                    model_data = model_data.dropna()
            if model_data.empty:
                _logger.warning("Skipping regression model with no complete rows", formula=formula)
                continue
            fit = fit_regression_model(
                model_type=outcome_model_type,
                formula=formula,
                data=model_data,
                inflation_formula=inflation_formula,
                covariance_type=covariance_type,
                mle_gradient_tolerance=mle_gradient_tolerance,
            )
            diagnostics = {}
            if outcome_model_type in {"nb", "zinb"}:
                diagnostics = validate_mle_fit(
                    fit,
                    model_id=spec.model_id,
                    dependent_variable=dependent_variable,
                    model_type=outcome_model_type,
                    policy=mle_failure_policy,
                    gradient_tolerance=mle_gradient_tolerance,
                )
            log_regression_summary(
                fit=fit,
                data=model_data,
                dependent_variable=dependent_variable,
                model_id=spec.model_id,
                formula=formula,
                model_type=outcome_model_type,
                inflation_formula=(inflation_formula if outcome_model_type == "zinb" else None),
            )
            coefficient_rows_for_model = coefficient_rows_from_fit(
                fit, dependent_variable, spec.model_id
            )
            for row in coefficient_rows_for_model:
                row["model_type"] = outcome_model_type
                row.update(diagnostics)
            coefficient_rows.extend(coefficient_rows_for_model)
            fit_row = fit_row_from_fit(fit, dependent_variable, spec.model_id, formula)
            fit_row["model_type"] = outcome_model_type
            fit_row["inflation_formula"] = (
                inflation_formula if outcome_model_type == "zinb" else None
            )
            fit_row.update(diagnostics)
            fit_rows.append(fit_row)
            if outcome_model_type == "zinb":
                formulas[f"{dependent_variable}_{spec.model_id}_inflation"] = inflation_formula
    return pd.DataFrame(coefficient_rows), pd.DataFrame(fit_rows), formulas


def import_statsmodels_formula_api() -> Any:
    """
    Import statsmodels formula API only when regressions are fitted.

    :return: Lazily imported ``statsmodels.formula.api`` module.
    """
    try:
        import statsmodels.formula.api as smf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "statsmodels is required to run regressions. Install project dependencies first."
        ) from exc
    return smf


def fit_regression_model(
    *,
    model_type: str,
    formula: str,
    data: pd.DataFrame,
    inflation_formula: str = "1",
    covariance_type: str | None = None,
    zinb_regularization: float = 3.0,
    zinb_fit_method: str = "bfgs",
    zinb_max_iterations: int = 1000,
    zinb_dispersion_power: int = 2,
    mle_gradient_tolerance: float = 1e-4,
) -> Any:
    """Fit one configured OLS, Poisson, NB, or ZINB model.

    NB fits retain the original BFGS-then-Newton candidate ordering and
    diagnostics-based candidate selection. ZINB starts are initialized from
    regularized inflation and negative-binomial GLM fits.

    :param model_type: Supported model family identifier.
    :param formula: Statsmodels formula for the count or linear model.
    :param data: Model-specific observations.
    :param inflation_formula: RHS for the zero-inflation component.
    :param covariance_type: Optional statsmodels covariance estimator.
    :param zinb_regularization: Penalty used for the inflation-start logit.
    :param zinb_fit_method: Optimizer used for the final ZINB fit.
    :param zinb_max_iterations: Maximum ZINB optimizer iterations.
    :param zinb_dispersion_power: NB parameterization power for ZINB.
    :param mle_gradient_tolerance: Per-observation score tolerance.
    :return: Fitted statsmodels result.
    :raises ValueError: If the model family or ZINB initialization is invalid.
    :raises RuntimeError: If both NB optimizers fail to produce a candidate.
    """
    smf = import_statsmodels_formula_api()
    fit_options = {"cov_type": covariance_type} if covariance_type else {}
    if model_type == "ols":
        return smf.ols(formula=formula, data=data).fit(**fit_options)
    if model_type not in {"pr", "nb", "zinb"}:
        raise ValueError(f"Unsupported regression model type: {model_type!r}")

    import statsmodels.api as sm
    from patsy import dmatrices

    fit_data = data
    first_year_center = np.nan
    rhs = formula.split("~", maxsplit=1)[1]
    rhs_terms = [term.strip() for term in rhs.split("+")]
    if model_type == "nb" and "first_year" in rhs_terms and "first_year" in data:
        fit_data = data.copy()
        first_year = pd.to_numeric(fit_data["first_year"], errors="coerce")
        first_year_center = float(first_year.mean())
        # Recenter this linear term to stabilize MLE; slopes and tests are invariant,
        # while the intercept is interpreted at the sample's mean first-publication year.
        fit_data["first_year"] = first_year - first_year_center

    endog_frame, exog_frame = dmatrices(formula, data=fit_data, return_type="dataframe")
    endog = endog_frame.iloc[:, 0]
    if model_type == "pr":
        return sm.GLM(endog, exog_frame, family=sm.families.Poisson()).fit(**fit_options)

    from statsmodels.discrete.discrete_model import NegativeBinomial

    if model_type == "nb":
        candidates = []
        errors: list[str] = []
        for method in ("bfgs", "newton"):
            try:
                with warnings.catch_warnings(record=True) as fit_warnings:
                    warnings.simplefilter("always")
                    fit = NegativeBinomial(endog, exog_frame).fit(
                        start_params=None,
                        method=method,
                        maxiter=1000,
                        disp=False,
                        **fit_options,
                    )
                fit._nmp_optimizer = method
                fit._nmp_alpha_start = negative_binomial_alpha_start(fit, method=method)
                fit._nmp_centered_first_year = first_year_center
                fit._nmp_optimizer_warnings = tuple(
                    f"{type(item.message).__name__}: {item.message}" for item in fit_warnings
                )
                candidates.append(fit)
                if not mle_fit_diagnostics(fit, gradient_tolerance=mle_gradient_tolerance)[
                    "mle_failure_reasons"
                ]:
                    return fit
            except Exception as exc:  # noqa: BLE001 - retain optimizer fallback policy
                errors.append(f"{method}: {type(exc).__name__}: {exc}")
        if not candidates:
            raise RuntimeError("Negative Binomial optimizers failed: " + "; ".join(errors))
        return min(
            candidates,
            key=lambda candidate: mle_candidate_score(
                candidate,
                gradient_tolerance=mle_gradient_tolerance,
            ),
        )

    from patsy import dmatrix
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    from statsmodels.discrete.discrete_model import Logit

    inflation_exog = dmatrix(inflation_formula, data=data, return_type="dataframe")
    inflation_start = Logit((endog == 0).astype(int), inflation_exog).fit_regularized(
        alpha=zinb_regularization,
        disp=False,
    )
    count_start = sm.GLM(
        endog,
        exog_frame,
        family=sm.families.NegativeBinomial(alpha=1.0),
    ).fit()
    mean_y = float(np.mean(endog))
    variance_y = float(np.var(endog))
    alpha_start = float(np.clip((variance_y - mean_y) / (mean_y**2 + 1e-12), 1e-2, 10.0))
    zinb = ZeroInflatedNegativeBinomialP(
        endog,
        exog_frame,
        exog_infl=inflation_exog,
        inflation="logit",
        p=zinb_dispersion_power,
    )
    inflation_map = dict(zip(inflation_exog.columns, np.asarray(inflation_start.params)))
    count_map = dict(zip(exog_frame.columns, np.asarray(count_start.params)))
    start_by_name: dict[str, float] = {}
    for name in zinb.exog_names:
        if name == "alpha":
            start_by_name[name] = alpha_start
        elif name == "lnalpha":
            start_by_name[name] = float(np.log(alpha_start))
        elif name.startswith("inflate_"):
            source_name = name.removeprefix("inflate_")
            if source_name in inflation_map:
                start_by_name[name] = float(inflation_map[source_name])
        elif name in count_map:
            start_by_name[name] = float(count_map[name])
    if len(start_by_name) != len(zinb.exog_names):
        missing = sorted(set(zinb.exog_names) - set(start_by_name))
        raise ValueError(f"Could not initialize ZINB parameters: {missing}")
    zinb_start = np.asarray([start_by_name[name] for name in zinb.exog_names])
    fit = zinb.fit(
        start_params=zinb_start,
        method=zinb_fit_method,
        maxiter=zinb_max_iterations,
        disp=False,
        **(fit_options or {"cov_type": "HC0"}),
    )
    fit._nmp_optimizer = zinb_fit_method
    fit._nmp_alpha_start = alpha_start
    return fit


class RegressionFitError(RuntimeError):
    """Raised when a maximum-likelihood regression fit fails diagnostics."""


def mle_fit_diagnostics(
    fit: Any,
    *,
    gradient_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Summarize optimizer status, parameter validity, and score diagnostics.

    :param fit: Fitted maximum-likelihood result.
    :param gradient_tolerance: Maximum acceptable normalized score norm.
    :return: Diagnostic values and a joined failure-reason string.
    """
    retvals = getattr(fit, "mle_retvals", {}) or {}
    raw_iterations = retvals.get("iterations")
    iterations = int(raw_iterations) if raw_iterations is not None else np.nan
    raw_calls = retvals.get("fcalls", retvals.get("gcalls"))
    function_calls = int(raw_calls) if raw_calls is not None else np.nan
    settings = getattr(fit, "mle_settings", {}) or {}
    raw_iteration_limit = settings.get("maxiter")
    iteration_limit = int(raw_iteration_limit) if raw_iteration_limit is not None else np.nan
    converged_value = retvals.get("converged", getattr(fit, "converged", None))
    converged = bool(converged_value) if converged_value is not None else None
    try:
        score = np.asarray(
            fit.model.score(np.asarray(fit.params, dtype=float)), dtype=float
        ).reshape(-1)
        gradient_norm = float(np.linalg.norm(score, ord=np.inf) / max(float(fit.nobs), 1.0))
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
        gradient_norm = np.nan
    try:
        parameters_finite = bool(np.isfinite(np.asarray(fit.params, dtype=float)).all())
        standard_errors_finite = bool(np.isfinite(np.asarray(fit.bse, dtype=float)).all())
        log_likelihood_finite = bool(np.isfinite(float(fit.llf)))
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
        parameters_finite = standard_errors_finite = log_likelihood_finite = False

    reasons = []
    if converged is not True:
        reasons.append(f"optimizer convergence status is {converged!r}")
    if not np.isfinite(gradient_norm):
        reasons.append("gradient norm is non-finite or unavailable")
    elif gradient_norm > gradient_tolerance:
        reasons.append(
            f"gradient norm {gradient_norm:.3g} exceeds tolerance {gradient_tolerance:.3g}"
        )
    if not parameters_finite:
        reasons.append("one or more parameter estimates are non-finite")
    if not standard_errors_finite:
        reasons.append("one or more standard errors are non-finite")
    if not log_likelihood_finite:
        reasons.append("log-likelihood is non-finite")
    if np.isfinite(iterations) and iterations < 1:
        reasons.append("optimizer reported no iterations")
    iteration_limit_reached = bool(
        np.isfinite(iteration_limit)
        and (
            (np.isfinite(iterations) and iterations >= iteration_limit)
            or (
                not np.isfinite(iterations)
                and np.isfinite(function_calls)
                and function_calls >= iteration_limit
            )
        )
    )
    if iteration_limit_reached:
        reasons.append("optimizer reached its configured iteration/call limit")

    params = getattr(fit, "params", pd.Series(dtype=float))
    alpha_estimate = float(params["alpha"]) if "alpha" in params else np.nan
    alpha_start = getattr(fit, "_nmp_alpha_start", np.nan)

    return {
        "mle_optimizer": getattr(fit, "_nmp_optimizer", settings.get("optimizer")),
        "mle_converged": converged,
        "mle_iterations": iterations,
        "mle_function_calls": function_calls,
        "mle_iteration_limit": iteration_limit,
        "mle_iteration_limit_reached": iteration_limit_reached,
        "mle_alpha_start": alpha_start,
        "mle_alpha_estimate": alpha_estimate,
        "mle_centered_first_year": getattr(fit, "_nmp_centered_first_year", np.nan),
        "mle_gradient_norm": gradient_norm,
        "mle_gradient_tolerance": gradient_tolerance,
        "mle_parameters_finite": parameters_finite,
        "mle_standard_errors_finite": standard_errors_finite,
        "mle_log_likelihood_finite": log_likelihood_finite,
        "mle_failure_reasons": "; ".join(reasons),
        "mle_optimizer_warnings": "; ".join(getattr(fit, "_nmp_optimizer_warnings", ())),
    }


def negative_binomial_alpha_start(fit: Any, *, method: str) -> float:
    """Recover the NB dispersion start from optimizer settings.

    :param fit: Fitted negative-binomial result.
    :param method: Optimizer used for the fit.
    :return: Dispersion start, or ``nan`` when it cannot be recovered.
    """
    settings = getattr(fit, "mle_settings", {}) or {}
    start_params = np.asarray(settings.get("start_params", []), dtype=float)
    if start_params.size == 0 or not np.isfinite(start_params[-1]):
        return np.nan
    if method in {"newton", "ncg"}:
        return float(start_params[-1])
    return float(np.exp(start_params[-1]))


def mle_candidate_score(fit: Any, *, gradient_tolerance: float) -> tuple[int, float]:
    """Rank optimizer candidates by diagnostic failures and gradient norm.

    :param fit: Candidate fitted result.
    :param gradient_tolerance: Maximum acceptable normalized score norm.
    :return: Failure count followed by gradient norm (infinity if unavailable).
    """
    diagnostics = mle_fit_diagnostics(fit, gradient_tolerance=gradient_tolerance)
    gradient = diagnostics["mle_gradient_norm"]
    return (
        len([reason for reason in diagnostics["mle_failure_reasons"].split("; ") if reason]),
        float(gradient) if np.isfinite(gradient) else float("inf"),
    )


def validate_mle_fit(
    fit: Any,
    *,
    model_id: str,
    dependent_variable: str,
    model_type: str,
    policy: str = "error",
    gradient_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Apply the configured error-or-warning policy to an MLE result.

    :param fit: Fitted maximum-likelihood result.
    :param model_id: Stable model specification identifier.
    :param dependent_variable: Modeled outcome name.
    :param model_type: Model family label.
    :param policy: Either ``error`` or ``warn``.
    :param gradient_tolerance: Maximum acceptable normalized score norm.
    :return: Diagnostics for the fit.
    :raises ValueError: If the policy is unsupported.
    :raises RegressionFitError: If diagnostics fail under the error policy.
    """
    if policy not in {"error", "warn"}:
        raise ValueError(f"Unsupported MLE failure policy: {policy!r}")
    diagnostics = mle_fit_diagnostics(fit, gradient_tolerance=gradient_tolerance)
    reasons = diagnostics["mle_failure_reasons"]
    if reasons:
        message = (
            f"{model_type.upper()} fit for {dependent_variable}/{model_id} failed "
            f"optimizer diagnostics: {reasons}; diagnostics={diagnostics}"
        )
        if policy == "error":
            raise RegressionFitError(message)
        if policy == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    return diagnostics


def coefficient_rows_from_fit(
    fit: Any, dependent_variable: str, model_id: str
) -> list[dict[str, Any]]:
    """
    Convert fitted model coefficients into long-form output rows.

    :param fit: Fitted statsmodels result object.
    :param dependent_variable: Dependent variable name.
    :param model_id: Model specification identifier.
    :return: Long-form coefficient rows for CSV output.
    """
    conf_int = fit.conf_int()
    rows = []
    for term in fit.params.index:
        rows.append(
            {
                "dependent_variable": dependent_variable,
                "model_id": model_id,
                "term": term,
                "estimate": fit.params[term],
                "std_error": fit.bse[term],
                "t_value": getattr(fit, "tvalues", fit.params)[term],
                "p_value": fit.pvalues[term],
                "conf_low": conf_int.loc[term, 0],
                "conf_high": conf_int.loc[term, 1],
                "n_obs": int(fit.nobs),
                "r_squared": getattr(fit, "rsquared", np.nan),
                "adj_r_squared": getattr(fit, "rsquared_adj", np.nan),
                "aic": fit_aic(fit),
                "bic": fit_bic(fit),
            }
        )
    return rows


def fit_row_from_fit(
    fit: Any, dependent_variable: str, model_id: str, formula: str
) -> dict[str, Any]:
    """
    Convert fitted model diagnostics into one fit-statistics output row.

    :param fit: Fitted statsmodels result object.
    :param dependent_variable: Dependent variable name.
    :param model_id: Model specification identifier.
    :param formula: Formula used to fit the model.
    :return: Fit-statistics row for CSV output.
    """
    return {
        "dependent_variable": dependent_variable,
        "model_id": model_id,
        "formula": formula,
        "n_obs": int(fit.nobs),
        "r_squared": getattr(fit, "rsquared", np.nan),
        "adj_r_squared": getattr(fit, "rsquared_adj", np.nan),
        "aic": fit_aic(fit),
        "bic": fit_bic(fit),
        "f_statistic": getattr(fit, "fvalue", np.nan),
        "f_pvalue": getattr(fit, "f_pvalue", np.nan),
    }


def r_lm_aic(fit: Any) -> float:
    """
    Return AIC using R's ``AIC.lm`` parameter count.

    Statsmodels OLS reports AIC with the fitted regression coefficients only. R's ``AIC.lm``
    additionally counts the Gaussian residual scale parameter, so the value is larger by 2.
    """
    return float(fit.aic + 2)


def r_lm_bic(fit: Any) -> float:
    """
    Return BIC using R's ``BIC.lm`` parameter count.

    R's ``BIC.lm`` counts one additional residual scale parameter compared with statsmodels,
    increasing BIC by ``log(nobs)``.
    """
    return float(fit.bic + np.log(float(fit.nobs)))


def fit_aic(fit: Any) -> float:
    """Return AIC using the source-compatible OLS parameter count."""
    return r_lm_aic(fit) if hasattr(fit, "rsquared") else float(fit.aic)


def fit_bic(fit: Any) -> float:
    """Return BIC using the source-compatible OLS parameter count."""
    return r_lm_bic(fit) if hasattr(fit, "rsquared") else float(fit.bic)


def log_regression_summary(
    *,
    fit: Any,
    data: pd.DataFrame,
    dependent_variable: str,
    model_id: str,
    formula: str,
    model_type: str = "ols",
    inflation_formula: str | None = None,
) -> None:
    """
    Log a formatted R-style regression summary for one fitted model.

    :param fit: Fitted statsmodels result object.
    :param data: Model-specific dataframe used for fitting.
    :param dependent_variable: Dependent variable name.
    :param model_id: Model specification identifier.
    :param formula: Formula used to fit the model.
    :return: None.
    """
    summary = format_regression_summary(
        fit=fit,
        data=data,
        dependent_variable=dependent_variable,
        model_id=model_id,
        formula=formula,
        model_type=model_type,
        inflation_formula=inflation_formula,
    )
    _logger.info(f"Regression model summary\n{summary}")


def format_regression_summary(
    *,
    fit: Any,
    data: pd.DataFrame,
    dependent_variable: str,
    model_id: str,
    formula: str,
    model_type: str = "ols",
    inflation_formula: str | None = None,
) -> str:
    """
    Build the complete formatted regression summary text.

    :param fit: Fitted statsmodels result object.
    :param data: Model-specific dataframe used for fitting.
    :param dependent_variable: Dependent variable name.
    :param model_id: Model specification identifier.
    :param formula: Formula used to fit the model.
    :return: R-style formatted regression summary.
    """
    if model_type != "ols":
        extra = f"\nInflation formula: {inflation_formula}" if inflation_formula else ""
        return (
            f"Dependent variable: {dependent_variable} | Model: {model_id} ({model_type})\n"
            f"Formula: {formula}{extra}\n\n{fit.summary().as_text()}"
        )
    dependent = pd.to_numeric(data[dependent_variable], errors="coerce")
    average_predictions = average_predictions_for_binary_indicator(
        fit=fit,
        data=data,
        variable="is_inbred",
    )
    average_comparison = average_comparison_for_binary_indicator(
        fit=fit,
        data=data,
        variable="is_inbred",
    )
    sections = [
        f"Dependent variable: {dependent_variable} | Model: {model_id}",
        "",
        "Call:",
        f"lm(formula = {formula}, data = df)",
        "",
        format_residuals(fit),
        "",
        format_coefficients(fit),
        "---",
        "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1",
        "",
        format_fit_statistics(fit),
    ]
    if average_predictions:
        sections.extend(["", format_average_predictions(average_predictions, variable="is_inbred")])
    if average_comparison is not None:
        sections.extend(["", format_average_comparison(average_comparison)])
    sections.extend(["", format_dependent_summary(dependent_variable, dependent)])
    return "\n".join(sections)


def format_residuals(fit: Any) -> str:
    """
    Format residual quantiles like R's ``summary.lm`` output.

    :param fit: Fitted statsmodels result object.
    :return: R-style residual quantile section.
    """
    quantiles = pd.Series(fit.resid).quantile([0, 0.25, 0.5, 0.75, 1])
    labels = ["Min", "1Q", "Median", "3Q", "Max"]
    values = [
        quantiles.loc[0],
        quantiles.loc[0.25],
        quantiles.loc[0.5],
        quantiles.loc[0.75],
        quantiles.loc[1],
    ]
    return "\n".join(
        [
            "Residuals:",
            " ".join(f"{label:>10}" for label in labels),
            " ".join(f"{format_float(value, digits=5):>10}" for value in values),
        ]
    )


def format_coefficients(fit: Any) -> str:
    """
    Format coefficient estimates, uncertainty, tests, and significance codes.

    :param fit: Fitted statsmodels result object.
    :return: R-style coefficient table with significance codes.
    """
    rows = coefficient_summary_rows(fit)
    term_width = max(len(""), *(len(row["term"]) for row in rows))
    lines = [
        "Coefficients:",
        (
            f"{'':<{term_width}} {'Estimate':>12} {'Std. Error':>12} "
            f"{'t value':>9} {'Pr(>|t|)':>10}   "
        ),
    ]
    for row in rows:
        lines.append(
            f"{row['term']:<{term_width}} "
            f"{format_float(row['estimate'], digits=7):>12} "
            f"{format_float(row['std_error'], digits=7):>12} "
            f"{format_float(row['t_value'], digits=3):>9} "
            f"{format_p_value(row['p_value']):>10} "
            f"{row['significance']}"
        )
    return "\n".join(lines)


def format_fit_statistics(fit: Any) -> str:
    """
    Format residual error, R-squared, and F-statistic diagnostics.

    :param fit: Fitted statsmodels result object.
    :return: R-style fit-statistics section.
    """
    return "\n".join(
        [
            (
                f"Residual standard error: {format_float(np.sqrt(fit.scale), digits=4)} "
                f"on {int(fit.df_resid)} degrees of freedom"
            ),
            (
                f"Multiple R-squared:  {format_float(fit.rsquared, digits=5)},\t"
                f"Adjusted R-squared:  {format_float(fit.rsquared_adj, digits=5)}"
            ),
            (
                f"F-statistic: {format_float(fit.fvalue, digits=4)} on {int(fit.df_model)} "
                f"and {int(fit.df_resid)} DF,  p-value: {format_p_value(fit.f_pvalue)}"
            ),
        ]
    )


def format_average_predictions(rows: list[dict[str, Any]], *, variable: str) -> str:
    """
    Format average prediction rows for a binary scenario variable.

    :param rows: Average-prediction rows.
    :param variable: Binary variable used for the prediction scenarios.
    :return: Formatted average-prediction table.
    """
    lines = [
        f"Average predictions for {variable}:",
        (
            f"{variable:>10} {'Estimate':>10} {'Std. Error':>11} {'z':>8} "
            f"{'Pr(>|z|)':>10} {'2.5 %':>9} {'97.5 %':>9}"
        ),
    ]
    for row in rows:
        lines.append(
            f"{row[variable]!s:>10} "
            f"{format_float(row['estimate'], digits=4):>10} "
            f"{format_float(row['std_error'], digits=4):>11} "
            f"{format_float(row['z_value'], digits=3):>8} "
            f"{format_p_value(row['p_value']):>10} "
            f"{format_float(row['conf_low'], digits=4):>9} "
            f"{format_float(row['conf_high'], digits=4):>9}"
        )
    lines.append("Type: response")
    return "\n".join(lines)


def format_average_comparison(row: dict[str, Any]) -> str:
    """
    Format the average comparison table for a binary scenario variable.

    :param row: Average-comparison row.
    :return: Formatted average-comparison table.
    """
    return "\n".join(
        [
            f"Average comparison for {row['term']}:",
            (
                f"{'Estimate':>10} {'Std. Error':>11} {'z':>8} "
                f"{'Pr(>|z|)':>10} {'2.5 %':>9} {'97.5 %':>9}"
            ),
            (
                f"{format_float(row['estimate'], digits=4):>10} "
                f"{format_float(row['std_error'], digits=4):>11} "
                f"{format_float(row['z_value'], digits=3):>8} "
                f"{format_p_value(row['p_value']):>10} "
                f"{format_float(row['conf_low'], digits=4):>9} "
                f"{format_float(row['conf_high'], digits=4):>9}"
            ),
            f"Term: {row['term']}",
            "Type: response",
            f"Comparison: {row['comparison']}",
        ]
    )


def format_dependent_summary(dependent_variable: str, dependent: pd.Series) -> str:
    """
    Format simple descriptive statistics for the dependent variable.

    :param dependent_variable: Dependent variable name.
    :param dependent: Dependent variable values used for fitting.
    :return: Formatted mean, median, and standard deviation section.
    """
    return "\n".join(
        [
            f"mean(df[[{dependent_variable!r}]])   {format_float(dependent.mean(), digits=7)}",
            f"median(df[[{dependent_variable!r}]]) {format_float(dependent.median(), digits=7)}",
            f"sd(df[[{dependent_variable!r}]])     {format_float(dependent.std(ddof=1), digits=7)}",
        ]
    )


def coefficient_summary_rows(fit: Any) -> list[dict[str, Any]]:
    """
    Build coefficient rows used by the formatted log summary.

    :param fit: Fitted statsmodels result object.
    :return: Coefficient rows used by the formatted log summary.
    """
    rows = []
    for term in fit.params.index:
        p_value = float(fit.pvalues[term])
        rows.append(
            {
                "term": term,
                "estimate": float(fit.params[term]),
                "std_error": float(fit.bse[term]),
                "t_value": float(fit.tvalues[term]),
                "p_value": p_value,
                "significance": significance_code(p_value),
            }
        )
    return rows


def average_predictions_for_binary_indicator(
    *,
    fit: Any,
    data: pd.DataFrame,
    variable: str,
) -> list[dict[str, Any]]:
    """
    Estimate average predictions after setting a binary variable to false and true.

    :param fit: Fitted statsmodels result object.
    :param data: Model-specific dataframe used for fitting.
    :param variable: Binary variable for the prediction scenarios.
    :return: Average-prediction rows for ``False`` and ``True`` scenarios.
    """
    if variable not in data.columns:
        return []
    return [
        average_prediction_row(fit=fit, data=data, variable=variable, value=value)
        for value in (False, True)
    ]


def average_prediction_row(
    *,
    fit: Any,
    data: pd.DataFrame,
    variable: str,
    value: bool,
) -> dict[str, Any]:
    """
    Estimate one average prediction scenario for a binary variable.

    :param fit: Fitted statsmodels result object.
    :param data: Model-specific dataframe used for fitting.
    :param variable: Binary variable for the prediction scenario.
    :param value: Scenario value assigned to the binary variable.
    :return: Average-prediction row with estimate, standard error, z-value, p-value,
    and confidence interval.
    """
    scenario = data.copy()
    scenario[variable] = value
    exog = design_matrix_for_fit(fit=fit, data=scenario)
    weights = np.asarray(exog).mean(axis=0)
    estimate, std_error = linear_combination_estimate(fit=fit, weights=weights)
    z_value = estimate / std_error if std_error else np.nan
    p_value = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    return {
        variable: value,
        "estimate": estimate,
        "std_error": std_error,
        "z_value": z_value,
        "p_value": p_value,
        "conf_low": estimate - 1.96 * std_error,
        "conf_high": estimate + 1.96 * std_error,
    }


def average_comparison_for_binary_indicator(
    *,
    fit: Any,
    data: pd.DataFrame,
    variable: str,
) -> dict[str, Any] | None:
    """
    Estimate the average ``True - False`` comparison for a binary variable.

    :param fit: Fitted statsmodels result object.
    :param data: Model-specific dataframe used for fitting.
    :param variable: Binary variable used for the ``True - False`` comparison.
    :return: Average-comparison row, or ``None`` when the variable is unavailable.
    """
    if variable not in data.columns:
        return None
    false_data = data.copy()
    true_data = data.copy()
    false_data[variable] = False
    true_data[variable] = True
    false_weights = np.asarray(design_matrix_for_fit(fit=fit, data=false_data)).mean(axis=0)
    true_weights = np.asarray(design_matrix_for_fit(fit=fit, data=true_data)).mean(axis=0)
    estimate, std_error = linear_combination_estimate(fit=fit, weights=true_weights - false_weights)
    z_value = estimate / std_error if std_error else np.nan
    p_value = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    return {
        "term": variable,
        "comparison": "True - False",
        "estimate": estimate,
        "std_error": std_error,
        "z_value": z_value,
        "p_value": p_value,
        "conf_low": estimate - 1.96 * std_error,
        "conf_high": estimate + 1.96 * std_error,
    }


def design_matrix_for_fit(*, fit: Any, data: pd.DataFrame) -> Any:
    """
    Recreate the fitted model's design matrix for scenario data.

    :param fit: Fitted statsmodels result object.
    :param data: Scenario dataframe.
    :return: Patsy design matrix aligned to the fitted model.
    """
    from patsy import build_design_matrices

    design_info = fit.model.data.design_info
    return build_design_matrices([design_info], data, return_type="dataframe")[0]


def linear_combination_estimate(*, fit: Any, weights: np.ndarray) -> tuple[float, float]:
    """
    Estimate a linear combination of fitted model parameters.

    :param fit: Fitted statsmodels result object.
    :param weights: Linear-combination weights over model parameters.
    :return: Estimate and standard error for the linear combination.
    """
    covariance = np.asarray(fit.cov_params())
    estimate = float(weights @ np.asarray(fit.params))
    variance = float(weights @ covariance @ weights.T)
    return estimate, float(np.sqrt(max(variance, 0)))


def format_float(value: Any, *, digits: int) -> str:
    """
    Format a numeric value for compact regression-summary output.

    :param value: Numeric value to format.
    :param digits: Number of decimal digits for fixed-point formatting.
    :return: Compact formatted number.
    """
    if value is None or pd.isna(value):
        return "NA"
    number = float(value)
    if not np.isfinite(number):
        return str(number)
    if number != 0 and (abs(number) < 1e-4 or abs(number) >= 1e5):
        return f"{number:.{max(digits - 1, 1)}e}"
    text = f"{number:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_p_value(value: Any) -> str:
    """
    Format p-values using R-like thresholds and notation.

    :param value: P-value to format.
    :return: R-style formatted p-value.
    """
    if value is None or pd.isna(value):
        return "NA"
    p_value = float(value)
    if not np.isfinite(p_value):
        return str(p_value)
    if p_value < 0.001:
        return "0.00"
    return f"{p_value:.5f}".rstrip("0").rstrip(".")


def significance_code(p_value: float) -> str:
    """
    Map a p-value to an R-style significance code.

    :param p_value: P-value.
    :return: R-style significance code.
    """
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "."
    return ""


def write_regression_outputs(
    *,
    coefficients: pd.DataFrame,
    fit_statistics: pd.DataFrame,
    formulas: dict[str, str],
    outputs: RegressionOutputs,
) -> None:
    """Write the core regression outputs to their configured artifacts.

    :param coefficients: Coefficient table returned by ``fit_regression_models``.
    :param fit_statistics: Fit-statistics table returned by ``fit_regression_models``.
    :param formulas: Mapping of model keys to formula strings.
    :param outputs: Regression stage's explicit output paths.
    """
    for frame, path in (
        (coefficients, outputs.model_coefficients),
        (fit_statistics, outputs.model_fit_statistics),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    outputs.model_formulas.parent.mkdir(parents=True, exist_ok=True)
    outputs.model_formulas.write_text(
        "\n".join(f"{key}: {formula}" for key, formula in sorted(formulas.items())) + "\n",
        encoding="utf-8",
    )
