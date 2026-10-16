# Copyright © 2026 TU Wien

"""Model-fit comparison tables used by regression reporting."""

from __future__ import annotations

import pandas as pd


def aic_bic_table(fit_statistics: pd.DataFrame) -> pd.DataFrame:
    """Build the AIC/BIC comparison table from model fit statistics.

    :param fit_statistics: Fit-statistics table with AIC and BIC columns.
    :return: AIC/BIC comparison table in input row order.
    """
    return fit_statistics.loc[:, ["dependent_variable", "model_id", "aic", "bic"]].rename(
        columns={"model_id": "model", "aic": "AIC", "bic": "BIC"}
    )


def ranked_aic_bic_table(fit_statistics: pd.DataFrame) -> pd.DataFrame:
    """Rank models within outcomes by AIC and BIC.

    :param fit_statistics: Fit-statistics table with AIC and BIC columns.
    :return: AIC/BIC table with within-outcome ranks.
    """
    result = aic_bic_table(fit_statistics)
    result["AIC_rank"] = result.groupby("dependent_variable")["AIC"].rank(method="min")
    result["BIC_rank"] = result.groupby("dependent_variable")["BIC"].rank(method="min")
    return result


def best_models_by_metric(fit_statistics: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    """Select the best model per outcome for one fit metric.

    :param fit_statistics: Model fit-statistics table.
    :param metric: Either ``AIC`` or ``BIC``.
    :return: Best model rows per dependent variable, sorted by outcome.
    """
    ranked = ranked_aic_bic_table(fit_statistics)
    idx = ranked.groupby("dependent_variable")[metric].idxmin()
    idx = idx.dropna().astype(int)
    if idx.empty:
        return ranked.iloc[0:0].copy()
    return ranked.loc[idx].sort_values("dependent_variable").reset_index(drop=True)


def inbreeding_discipline_spec_comparison(
    *, coefficients: pd.DataFrame, fit_statistics: pd.DataFrame
) -> pd.DataFrame:
    """Compare the inbreeding coefficient across disciplinary specifications.

    :param coefficients: Long-form model coefficient table.
    :param fit_statistics: Model fit and specification diagnostics.
    :return: Selected coefficient rows with their discipline-control metadata.
    """
    if coefficients.empty:
        return pd.DataFrame()
    rows = coefficients[
        coefficients["model_id"].isin(["m7", "m8", "m9"])
        & coefficients["term"].eq("C(is_inbred)[T.True]")
    ].copy()
    if rows.empty:
        return rows
    stats = fit_statistics[
        [
            "dependent_variable",
            "model_id",
            "field_share_reference",
            "share_control_type",
            "design_full_rank",
        ]
    ].copy()
    rows = rows.merge(stats, on=["dependent_variable", "model_id"], how="left")
    rows["discipline_specification"] = rows["model_id"].map(
        {
            "m7": "no_field_control",
            "m8": "detailed_field_shares",
            "m9": "detailed_field_shares_plus_first_year",
        }
    )
    return rows[
        [
            "dependent_variable",
            "discipline_specification",
            "model_id",
            "estimate",
            "std_error",
            "conf_low",
            "conf_high",
            "p_value",
            "n_obs",
            "share_control_type",
            "field_share_reference",
            "design_full_rank",
        ]
    ].sort_values(["dependent_variable", "model_id"])
