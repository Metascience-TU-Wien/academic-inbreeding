# Copyright © 2026 TU Wien

"""Behavioral tests for the statistical-matching suite."""

from pathlib import Path

import pandas as pd
import pytest

from acib.statistical_matching.constants import DEFAULT_METHODS
from acib.statistical_matching.data import encoded_covariates
from acib.statistical_matching.reporting import statistical_matching_console_table
from acib.statistical_matching.schemas import validate_matching_input, validate_output
from acib.statistical_matching.strategies import fit_strategy
from acib.statistical_matching.suite import StatisticalMatchingSuite, fit_suite

from .conftest import matching_config


def test_suite_runs_all_methods_and_emits_complete_results(
    synthetic_matching_data: pd.DataFrame, tmp_path: Path
) -> None:
    config = matching_config(tmp_path)
    result = StatisticalMatchingSuite(data=synthetic_matching_data, config=config).fit()
    assert tuple(result.balance_summary.method) == DEFAULT_METHODS
    assert set(result.balanced_samples) == set(DEFAULT_METHODS)
    assert len(result.outcome_effects) == 32
    assert len(result.adjusted_means) == 64
    assert set(result.covariate_balance.method) == set(DEFAULT_METHODS)
    assert result.formulas["balancing"].startswith("is_inbred ~ academic_age")
    absolute = result.latex_tables["matching_results_absolute.tex"]
    assert (
        r"P(top 10\%), P(top 5\%), and P(top 1\%) are professor-level sums of "
        "publication-level fractional indicators for their respective top citation-impact groups."
    ) in absolute


def test_display_scales_only_pp_effects(
    synthetic_matching_data: pd.DataFrame, tmp_path: Path
) -> None:
    result = fit_suite(synthetic_matching_data, matching_config(tmp_path))
    pp = result.outcome_effects.query("outcome == 'pp_top_10'").iloc[0]
    mwpr = result.outcome_effects.query("outcome == 'hazen_perc_med'").iloc[0]
    assert pp.estimate_display == pp.estimate * 100
    assert mwpr.estimate_display == mwpr.estimate


def test_absolute_outcomes_reporting_and_order(
    synthetic_matching_data: pd.DataFrame, tmp_path: Path
) -> None:
    outcomes = ("hazen_perc_med", "pp_top_10", "n_public", "P_top10")
    aliases = {"P_top10": r"P(top 10\%)", "n_public": "Publications"}
    types = {
        "hazen_perc_med": "relative",
        "pp_top_10": "relative",
        "n_public": "absolute",
        "P_top10": "absolute",
    }
    result = fit_suite(
        synthetic_matching_data,
        matching_config(tmp_path, outcomes=outcomes, outcome_types=types, outcome_aliases=aliases),
    )
    p_row = result.outcome_effects.query("method == 'PSM' and outcome == 'P_top10'").iloc[0]
    console = statistical_matching_console_table(
        result.outcome_effects,
        covariance_type=result.covariance_type,
        outcomes=outcomes,
        aliases=aliases,
    )
    assert p_row.estimate_display == p_row.estimate
    assert "P(top 10%) ATT" in console
    assert "P outcomes are unrounded sums of fractional publication counts." in console
    assert "PP outcomes are reported in percentage points." in console
    relative, absolute = result.latex_tables.values()
    assert r"\label{tab:matching_results_relative}" in relative
    assert r"\label{tab:matching_results_absolute}" in absolute
    assert r"\quad Robust SE" in relative and r"\quad Robust SE" in absolute
    assert r"Std.\ err." not in relative + absolute
    assert "HC3 heteroskedasticity-robust standard errors (Robust SE)" in relative
    assert r"\(p\)-values (\(P\)) based on \(t\)-tests" in relative
    assert r"PP(top \(x\%\)) outcomes are reported in percentage points." in relative
    assert "lower absolute publication output" in absolute
    assert r"P(top 10\%)" in absolute
    assert "Publications is the number of publications per professor." in absolute
    assert (
        r"P(top 10\%) is the professor-level sum of publication-level fractional "
        "indicators for its top citation-impact group."
    ) in absolute
    assert "P(top 10%) [count]" not in absolute
    aliases_ordered = ["Publications", r"P(top 10\%)"]
    assert [absolute.index(alias) for alias in aliases_ordered] == sorted(
        absolute.index(alias) for alias in aliases_ordered
    )


def test_effects_expose_confidence_intervals(
    synthetic_matching_data: pd.DataFrame, tmp_path: Path
) -> None:
    effects = fit_suite(synthetic_matching_data, matching_config(tmp_path)).outcome_effects
    row = effects.iloc[0]
    assert {"ci_lower", "ci_upper", "ci_lower_display", "ci_upper_display"} <= set(effects)
    assert row.ci_lower == row.conf_low
    assert row.ci_upper == row.conf_high
    assert row.ci_lower_display == row.conf_low_display
    assert row.ci_upper_display == row.conf_high_display


def test_full_professor_is_categorical_reference(
    synthetic_matching_data: pd.DataFrame, tmp_path: Path
) -> None:
    config = matching_config(tmp_path, covariates=("academic_age", "title"))
    encoded = encoded_covariates(synthetic_matching_data, ("title",))
    result = fit_suite(synthetic_matching_data, config)
    assert 'Treatment(reference="Full Prof.")' in result.formulas["balancing"]
    assert not any("[T.Full Prof.]" in column for column in encoded)
    assert any("[T.Ao. Prof.]" in column for column in encoded)


def test_treated_value_setting_remains_inert_as_in_source(
    synthetic_matching_data: pd.DataFrame, tmp_path: Path
) -> None:
    config = matching_config(tmp_path, treated_value=False)

    result = StatisticalMatchingSuite(data=synthetic_matching_data, config=config).fit()

    assert result.balance_summary.n_inbred.tolist() == [96, 96, 96, 96]


def test_balance_includes_omitted_reference_terms(
    synthetic_matching_data: pd.DataFrame, tmp_path: Path
) -> None:
    from acib.statistical_matching.balance import covariate_balance

    data = synthetic_matching_data.assign(
        life_sci_share=0.2,
        health_sci_share=0.1,
        phys_sci_share=0.3,
        soc_hum_share=0.2,
        multi_area_share=0.2,
    )
    covariates = (
        "academic_age",
        "title",
        "first_year",
        "life_sci_share",
        "health_sci_share",
        "soc_hum_share",
        "multi_area_share",
    )
    result = covariate_balance(
        data,
        data,
        pd.Series(1.0, index=data.index),
        treatment="is_inbred",
        covariates=covariates,
        method="test",
        threshold=0.10,
    )
    covars = set(result.covariate)
    assert {"phys_sci_share", "title[Full Prof.]", "title[Ao. Prof.]"} <= covars


def test_writes_configured_artifacts(synthetic_matching_data: pd.DataFrame, tmp_path: Path) -> None:
    config = matching_config(tmp_path)
    synthetic_matching_data.assign(sc_id=range(len(synthetic_matching_data))).to_csv(
        config.input_path, index=False
    )
    from acib.statistical_matching.workflow import run_statistical_matching_suite

    run_statistical_matching_suite(config)
    assert all(path.exists() for path in config.outputs.all_paths())
    assert (config.outputs.latex_dir / "matching_results_relative.tex").exists()
    assert (config.outputs.latex_dir / "matching_results_absolute.tex").exists()
    assert (config.outputs.plots_dir / "covariate_balance.png").exists()
    assert (config.outputs.plots_dir / "absolute_outcome_effects.png").exists()
    assert (config.outputs.plots_dir / "propensity_score_overlap.png").exists()
    assert (config.outputs.plots_dir / "propensity_score_overlap_by_method.png").exists()


def test_latex_and_console_reports_contain_dynamic_estimates(
    synthetic_matching_data: pd.DataFrame, tmp_path: Path
) -> None:
    result = fit_suite(synthetic_matching_data, matching_config(tmp_path))
    row = result.outcome_effects.query("method == 'PSM' and outcome == 'hazen_perc_med'").iloc[0]
    latex = result.latex_table
    assert r"\label{tab:matching_results_relative}" in latex
    assert r"\begin{tabular*}{0.84\textwidth}" in latex
    assert r"\multicolumn{1}{c}{\textbf{PSM}}" in latex
    assert r"\multicolumn{1}{c}{\textbf{IPTW}}" in latex
    assert f"{row.estimate_display:.2f}" in latex
    assert f"[{row.conf_low_display:.2f}, {row.conf_high_display:.2f}]" in latex
    assert "Statistical matching / weighting results" in result.console_table
    assert "mwPR ATT" in result.console_table and "PP(top 10%) ATT" in result.console_table


def test_source_derived_scientific_golden_values(
    synthetic_matching_data: pd.DataFrame, tmp_path: Path
) -> None:
    result = fit_suite(synthetic_matching_data, matching_config(tmp_path))
    row = result.outcome_effects.query("method == 'PSM' and outcome == 'hazen_perc_med'").iloc[0]
    assert row.estimate == pytest.approx(-3.0, abs=1e-10)
    assert row.std_error == pytest.approx(0.2029223250, abs=1e-8)
    assert row.ci_lower == pytest.approx(-3.4002700049, abs=1e-8)
    assert row.ci_upper == pytest.approx(-2.5997299951, abs=1e-8)
    for sample in result.balanced_samples.values():
        treated = sample.data.is_inbred.astype(bool)
        assert sample.weights.loc[treated].sum() == pytest.approx(96.0)
        assert sample.weights.loc[~treated].sum() == pytest.approx(96.0)
    assert result.balance_summary.max_abs_smd.abs().max() == pytest.approx(0.0, abs=1e-12)


def test_threshold_is_strict_and_boundary_is_imbalanced() -> None:
    from acib.statistical_matching.balance import covariate_balance

    frame = pd.DataFrame({"t": [True, True, False, False], "x": [-0.95, 1.05, -1.05, 0.95]})
    result = covariate_balance(
        frame,
        frame,
        pd.Series(1.0, index=frame.index),
        treatment="t",
        covariates=("x",),
        method="boundary",
        threshold=0.10,
    )
    assert result.iloc[0].abs_diff_adjusted == pytest.approx(0.10)
    assert not result.iloc[0].balanced


def test_zero_pooled_variance_preserves_source_zero_smd_behavior() -> None:
    from acib.statistical_matching.balance import covariate_balance

    frame = pd.DataFrame({"treated": [True, True, False, False], "constant": [0, 0, 1, 1]})
    result = covariate_balance(
        frame,
        frame,
        pd.Series(1.0, index=frame.index),
        treatment="treated",
        covariates=("constant",),
        method="zero-variance",
        threshold=0.10,
    )
    assert result.loc[0, "diff_unadjusted"] == 0.0
    assert result.loc[0, "diff_adjusted"] == 0.0


def test_psm_resolves_propensity_ties_by_control_input_order(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "row_id": range(8),
            "is_inbred": [True, True, True, True, False, False, False, False],
            "age": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    config = matching_config(
        tmp_path,
        outcomes=("hazen_perc_med",),
        covariates=("age",),
        methods=("PSM",),
    )

    sample = fit_strategy("PSM", data, config)

    assert sample.data.row_id.tolist() == [0, 4, 1, 5, 2, 6, 3, 7]
    assert sample.diagnostics["n_pairs"] == 4


def test_psm_stops_after_controls_are_exhausted(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "row_id": range(5),
            "is_inbred": [True, True, True, False, False],
            "age": [0, 1, 0, 0, 1],
        }
    )
    config = matching_config(
        tmp_path,
        outcomes=("hazen_perc_med",),
        covariates=("age",),
        methods=("PSM",),
    )

    sample = fit_strategy("PSM", data, config)

    assert sample.diagnostics["n_pairs"] == 2
    assert sample.data.row_id.tolist() == [0, 3, 1, 4]


def test_matching_input_requires_producer_and_dynamic_fields(
    synthetic_matching_data: pd.DataFrame,
) -> None:
    producer = synthetic_matching_data.assign(
        sc_id=range(len(synthetic_matching_data)), producer_extra=1
    )
    assert len(validate_matching_input(producer, {"P_top10"})) == len(synthetic_matching_data)
    with pytest.raises(ValueError, match="missing configured columns"):
        validate_matching_input(producer.drop(columns="P_top10"), {"P_top10"})


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(
            {
                "sc_id": [1, 1],
                "is_inbred": [False, True],
                "title": ["Full Prof."] * 2,
            }
        ),
        pd.DataFrame({"sc_id": [1], "is_inbred": ["True"], "title": ["Full Prof."]}),
    ],
    ids=("duplicate-producer-researcher-id", "invalid-producer-treatment-type"),
)
def test_matching_input_enforces_regression_producer_contract(frame: pd.DataFrame) -> None:
    from pandera.errors import SchemaError

    with pytest.raises(SchemaError):
        validate_matching_input(frame, {"is_inbred"})


def test_output_schema_rejects_malformed_artifact() -> None:
    from pandera.errors import SchemaError

    with pytest.raises(SchemaError):
        validate_output("balance_summary", pd.DataFrame({"method": ["PSM"]}))


@pytest.mark.parametrize("method", ["PSM", "CEM"])
def test_matching_methods_reject_no_overlap(method: str, tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "is_inbred": [True, True, False, False],
            "academic_age": [1, 1, 20, 20],
            "title": ["A"] * 4,
            "primary_field": ["X"] * 4,
            "first_year": [2000] * 4,
            "hazen_perc_med": [1.0] * 4,
        }
    )
    config = matching_config(
        tmp_path,
        outcomes=("hazen_perc_med",),
        covariates=("academic_age",),
        methods=(method,),
        cem_cutpoints={"academic_age": (5,)},
    )
    if method == "PSM":
        data["is_inbred"] = True
    with pytest.raises(ValueError):
        fit_strategy(method, data, config)


def test_entropy_balancing_handles_no_feasible_covariate_columns(
    synthetic_matching_data: pd.DataFrame, tmp_path: Path
) -> None:
    data = synthetic_matching_data.copy()
    data.loc[data.is_inbred, "academic_age"] = 100
    config = matching_config(
        tmp_path,
        outcomes=("hazen_perc_med",),
        covariates=("academic_age",),
        methods=("EB",),
    )
    from acib.statistical_matching.data import matching_frame, prepare_matching_data

    prepared = matching_frame(
        prepare_matching_data(data, treatment_variable="is_inbred", exclude_titles=()),
        treatment_variable="is_inbred",
        covariates=config.covariates,
        drop_missing=True,
    )
    sample = fit_strategy("EB", prepared, config)
    assert sample.weights.notna().all()
