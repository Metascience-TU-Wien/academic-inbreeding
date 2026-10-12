# Copyright © 2026 TU Wien

"""Treatment balancing strategies used by statistical matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from acib.regression.models import import_statsmodels_formula_api
from acib.statistical_matching.config import StatisticalMatchingConfig
from acib.statistical_matching.data import encoded_covariates, propensity_formula


@dataclass(frozen=True, slots=True)
class BalancedSample:
    """Data and weights for one balancing method."""

    method_id: str
    estimand: str
    data: pd.DataFrame
    weights: pd.Series
    diagnostics: dict[str, Any] = field(default_factory=dict)


def fit_propensity_scores(data: pd.DataFrame, config: StatisticalMatchingConfig) -> np.ndarray:
    """Fit logistic propensity scores and clip them to source bounds.

    :param data: Matching observations.
    :param config: Matching settings.
    :return: Clipped fitted probabilities in row order.
    """
    smf = import_statsmodels_formula_api()
    fit_data = data.copy()
    fit_data[config.treatment_variable] = fit_data[config.treatment_variable].astype(int)
    fit = smf.logit(
        propensity_formula(
            treatment=config.treatment_variable, data=data, covariates=config.covariates
        ),
        data=fit_data,
    ).fit(disp=False)
    return np.clip(np.asarray(fit.predict(fit_data), dtype=float), 1e-6, 1 - 1e-6)


def fit_strategy(
    method: str, data: pd.DataFrame, config: StatisticalMatchingConfig
) -> BalancedSample:
    """Fit one configured balancing strategy.

    :param method: Strategy identifier.
    :param data: Matching observations.
    :param config: Matching settings.
    :return: Selected observations, weights, and diagnostics.
    :raises KeyError: If the strategy is unsupported.
    :raises ValueError: If the strategy cannot produce a valid sample.
    """
    treatment = data[config.treatment_variable].astype(bool)
    if method in {"PSM", "IPTW"}:
        scores = fit_propensity_scores(data, config)
        if method == "IPTW":
            return _sample(
                method, config.estimand, data, np.where(treatment, 1.0, scores / (1 - scores))
            )
        controls = set(np.flatnonzero(~treatment.to_numpy()).tolist())
        positions: list[int] = []
        for treated in np.flatnonzero(treatment.to_numpy()):
            if not controls:
                break
            available = np.array(sorted(controls), dtype=int)
            control = int(available[np.abs(scores[available] - scores[treated]).argmin()])
            controls.remove(control)
            positions.extend((int(treated), control))
        if not positions:
            raise ValueError("PSM could not form any treated-control pairs.")
        sample = data.iloc[positions].copy().reset_index(drop=True)
        weights = pd.Series(1.0, index=sample.index, name="weights")
        sample["propensity_score"] = scores[positions]
        sample["weights"] = weights
        return BalancedSample(
            method, config.estimand, sample, weights, {"n_pairs": len(positions) // 2}
        )
    if method == "CEM":
        strata = pd.DataFrame(index=data.index)
        cuts = config.cem_cutpoints or {}
        for column in config.covariates:
            strata[column] = (
                pd.cut(data[column], [-np.inf, *cuts[column], np.inf], include_lowest=True)
                if column in cuts
                else data[column].astype(str)
            )
        weights = pd.Series(0.0, index=data.index, name="weights")
        matched_strata = 0
        for positions in strata.groupby(list(strata.columns), observed=True).groups.values():
            idx = pd.Index(positions)
            treated = treatment.loc[idx]
            nt, nc = int(treated.sum()), int((~treated).sum())
            if nt and nc:
                matched_strata += 1
                weights.loc[idx[treated.to_numpy()]] = 1.0
                weights.loc[idx[(~treated).to_numpy()]] = nt / nc
        keep = weights.gt(0)
        if not keep.any():
            raise ValueError("CEM found no strata containing both treatment groups.")
        sample = data.loc[keep].copy().reset_index(drop=True)
        weights = weights.loc[keep].reset_index(drop=True)
        sample["weights"] = weights
        return BalancedSample(
            method, config.estimand, sample, weights, {"matched_strata": matched_strata}
        )
    if method == "EB":
        design = encoded_covariates(data, config.covariates).astype(float)
        control, treated = design.loc[~treatment], design.loc[treatment]
        if control.empty or treated.empty:
            raise ValueError("Entropy balancing requires treated and control observations.")
        selected = []
        dropped = []
        for column in control:
            values = control[column]
            mean = float(treated[column].mean())
            low, high = float(values.min()), float(values.max())
            if np.isclose(values.std(ddof=0), 0.0):
                if not np.isclose(mean, low):
                    dropped.append(column)
            elif low - 1e-10 <= mean <= high + 1e-10:
                selected.append(column)
            else:
                dropped.append(column)
        if selected:
            controls_weight = _entropy_weights(
                control[selected], treated[selected], float(treatment.sum())
            )
        else:
            controls_weight = pd.Series(float(treatment.sum()) / len(control), index=control.index)
        weights = pd.Series(1.0, index=data.index, name="weights")
        weights.loc[controls_weight.index] = controls_weight
        sample = data.copy().reset_index(drop=True)
        weights = weights.reset_index(drop=True)
        sample["weights"] = weights
        return BalancedSample(
            method,
            config.estimand,
            sample,
            weights,
            {"dropped_entropy_balance_columns": ";".join(dropped)},
        )
    raise KeyError(f"Unknown statistical matching method: {method}")


def _entropy_weights(control: pd.DataFrame, treated: pd.DataFrame, total: float) -> pd.Series:
    """Compute exponential-tilt weights using BFGS.

    :param control: Encoded control observations.
    :param treated: Encoded treated observations.
    :param total: Target sum of control weights.
    :return: Calibrated control weights.
    :raises ValueError: If optimization fails to converge.
    """
    mean = control.mean()
    sd = control.std(ddof=0).replace(0.0, 1.0)
    x = ((control - mean) / sd).to_numpy(dtype=float)
    target = ((treated.mean() - mean) / sd).to_numpy()

    def objective(beta: np.ndarray) -> float:
        """Evaluate the entropy-balancing objective."""
        return float(logsumexp(x @ beta) - target @ beta)

    def gradient(beta: np.ndarray) -> np.ndarray:
        """Evaluate the objective gradient."""
        eta = x @ beta
        return np.exp(eta - logsumexp(eta)) @ x - target

    fit = minimize(
        objective,
        np.zeros(x.shape[1]),
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 10000},
    )
    if not fit.success and np.abs(gradient(fit.x)).max() > 1e-5:
        raise ValueError(f"Entropy balancing did not converge: {fit.message}")
    probs = np.exp(x @ fit.x - logsumexp(x @ fit.x))
    return pd.Series(total * probs, index=control.index, name="weights")


def _sample(method: str, estimand: str, data: pd.DataFrame, values: np.ndarray) -> BalancedSample:
    """Attach weights and package a full-sample balancing result."""
    weights = pd.Series(values, index=data.index, name="weights")
    sample = data.copy().reset_index(drop=True)
    weights = weights.reset_index(drop=True)
    sample["weights"] = weights
    return BalancedSample(method, estimand, sample, weights)
