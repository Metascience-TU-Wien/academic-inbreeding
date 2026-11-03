# Copyright © 2026 TU Wien

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pytest import CaptureFixture

from acib.regression.analysis_config import RegressionSuiteConfig
from acib.regression.constants import (
    DOMAIN_SHARE_COLUMNS,
    FIELD_SHARE_COLUMNS,
    FIELD_SHARE_TERMS,
    PAPER_REGRESSION_MODEL_TERMS,
)
from acib.regression.fields import (
    impact_asjc_subject_area,
    included_domain_share_terms,
    included_field_share_terms,
    professor_field_shares,
    unmatched_impact_class_name_labels,
)
from acib.regression.input import (
    build_professor_article_input,
    build_regression_input,
    prepare_author_summary,
)
from acib.regression.latex import (
    LatexModelColumn,
    format_p_value_cell,
    model_comparison_latex_table,
    model_inference_note,
    outcome_header,
    performance_table_note,
    table_preamble,
)
from acib.regression.models import (
    RegressionFitError,
    _validate_glm_fit,
    coefficient_rows_from_fit,
    fit_regression_model,
    fit_regression_models,
    mle_fit_diagnostics,
    prepare_model_estimation_data,
    regression_model_specs,
    required_columns_from_formula,
    validate_mle_fit,
)
from acib.regression.specs import RegressionModelSpec
from acib.regression.suite import RegressionSuite
from acib.regression.tables import best_models_by_metric


def tiny_authors(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "author_ids": ids,
            "selected_disambiguated_cluster_id": [f"cluster-{item}" for item in ids],
            "person_title": ["Univ.-Prof. Dr."] * len(ids),
            "is_inbreeding_proxy": [True] * len(ids),
            "n_unique": [4] * len(ids),
            "academic_age_years": [10] * len(ids),
            "first_date": ["2010-01-01"] * len(ids),
        }
    )


def test_build_regression_input_aggregates_outcomes_and_subject_counts() -> None:
    articles = pd.DataFrame(
        {
            "eid": ["e1", "e2", "e1", "e4"],
            "pubyear": [2000, 2001, 2000, 2024],
            "author_ids": ["a1"] * 4,
            "leiden_p_top_1": [0.0, 1.0, 0.0, np.nan],
            "leiden_p_top_5": [1.0, 1.0, 1.0, np.nan],
            "leiden_p_top_10": [1.0, 0.0, 1.0, np.nan],
            "ptop1_frac": [0.025, 0.05, 0.025, np.nan],
            "ptop5_frac": [0.3, 0.1, 0.3, np.nan],
            "ptop10_frac": [0.3, 0.2, 0.3, np.nan],
            "hazen_perc_med": [40.0, 60.0, 40.0, np.nan],
            "di1": [0.1, 0.5, 0.1, np.nan],
        }
    )
    authors = tiny_authors(["a1"])
    authors["article_general_asjc_subject_area_publication_counts"] = [
        (
            "{1500: {'category': 'General Chemical Engineering', 'n': 2}, "
            "1600: {'category': 'General Chemistry', 'n': 1}}"
        )
    ]
    result = build_regression_input(author_articles=articles, authors=authors)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["pp_top_10"] == 0.5
    assert row["P_top10"] == pytest.approx(0.5)
    assert row["P_top5"] == pytest.approx(0.4)
    assert row["P_top1"] == pytest.approx(0.075)
    assert row["n_public"] == 4
    assert row["n_public_impact"] == 2
    assert row["hazen_perc_med"] == 50
    assert row["di1"] == pytest.approx(0.3)
    assert row["primary_field"] == "General Chemical Engineering"
    assert row["n_public_chemical_engineering"] == 2
    assert row["n_public_chemistry"] == 1


def test_build_regression_input_preserves_distinct_researcher_ids_per_cluster() -> None:
    articles = pd.DataFrame(
        {
            "eid": ["e1"],
            "author_ids": ["a1"],
            "pp_top_10": [1.0],
            "hazen_perc_med": [90.0],
        }
    )
    authors = tiny_authors(["a1", "a1"])
    authors["selected_disambiguated_cluster_id"] = ["cluster-1", "cluster-1"]
    authors["sc_id"] = [101, 102]
    authors["is_inbreeding_proxy"] = [True, False]
    result = build_regression_input(author_articles=articles, authors=authors)
    assert result[["sc_id", "selected_disambiguated_cluster_id", "is_inbred"]].to_dict(
        orient="records"
    ) == [
        {
            "sc_id": 101,
            "selected_disambiguated_cluster_id": "cluster-1",
            "is_inbred": True,
        },
        {
            "sc_id": 102,
            "selected_disambiguated_cluster_id": "cluster-1",
            "is_inbred": False,
        },
    ]


def test_author_summary_keeps_zero_denominator_coaffiliation_missing() -> None:
    authors = tiny_authors(["a1"])
    authors["n_unique"] = [0]
    authors["n_coaffiliations"] = [0]
    authors["n_coaffiliation_publications"] = [0]
    assert pd.isna(prepare_author_summary(authors).loc[0, "coaffiliation_publication_share"])


def test_build_input_applies_inclusive_first_year_cutoff() -> None:
    articles = pd.DataFrame(
        {
            "eid": ["old", "new"],
            "author_ids": ["a", "b"],
            "pp_top_10": [1.0, 0.0],
            "hazen_perc_med": [90.0, 50.0],
        }
    )
    authors = tiny_authors(["a", "b"])
    authors["first_date"] = ["1995-01-01", "1996-01-01"]
    result = build_regression_input(author_articles=articles, authors=authors, min_first_year=1996)
    assert result["first_year"].tolist() == [1996]


def test_build_input_drops_rows_without_impact_values() -> None:
    articles = pd.DataFrame(
        {
            "eid": ["matched", "unmatched"],
            "author_ids": ["a1", "a1"],
            "hazen_perc_med": [80.0, np.nan],
            "pp_top_10": [1.0, np.nan],
            "pp_top_5": [1.0, np.nan],
            "pp_top_1": [0.0, np.nan],
        }
    )
    result = build_regression_input(author_articles=articles, authors=tiny_authors(["a1"]))
    assert result[["hazen_perc_med", "pp_top_10", "pp_top_5", "pp_top_1"]].to_dict(
        orient="records"
    ) == [{"hazen_perc_med": 80.0, "pp_top_10": 1.0, "pp_top_5": 1.0, "pp_top_1": 0.0}]


def test_field_shares_fractionally_weight_each_publication() -> None:
    data = pd.DataFrame(
        {
            "sc_id": [1, 1, 2],
            "eid": ["p1", "p2", "p3"],
            "article_general_asjc_subject_areas": [
                {2700: "Medicine", 3200: "Psychology"},
                {2700: "Medicine"},
                {},
            ],
        }
    )
    result = professor_field_shares(data)
    weights = result.publication_contributions.groupby(["sc_id", "eid"])["field_weight"].sum()
    shares = result.shares.set_index("sc_id")
    assert np.allclose(weights.loc[1], 1.0)
    assert weights.loc[(2, "p3")] == 0.0
    assert shares.loc[1, "med_share"] == 0.75
    assert shares.loc[1, "psych_share"] == 0.25
    assert shares.loc[2, list(FIELD_SHARE_COLUMNS)].sum() == 0.0


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("{Medicine,Oncology}", {2700: "Medicine"}),
        ("Earth and Planetary Sciences (all)", {1900: "Earth and Planetary Sciences"}),
        (
            "{Business and International Management,Software}",
            {1400: "Business, Management and Accounting", 1700: "Computer Science"},
        ),
        ("Ecology", {2300: "Environmental Science"}),
    ],
)
def test_impact_class_names_map_to_major_asjc_fields(label: str, expected: dict[int, str]) -> None:
    assert impact_asjc_subject_area(label) == expected
    assert unmatched_impact_class_name_labels(label) == ()


def test_impact_class_name_collapses_detailed_categories() -> None:
    assert impact_asjc_subject_area(
        "{Oncology,Cardiology and Cardiovascular Medicine,Computer Science Applications}"
    ) == {1700: "Computer Science", 2700: "Medicine"}


def test_field_shares_use_article_impact_class_names() -> None:
    articles = pd.DataFrame(
        {
            "eid": ["e1", "e2"],
            "author_ids": ["a1", "a1"],
            "hazen_perc_med": [70.0, 80.0],
            "pp_top_10": [1.0, 0.0],
        }
    )
    impact = pd.DataFrame(
        {"item_id": ["e1", "e2"], "class_name": ["Medicine", "Computer Science (all)"]}
    )
    authors = tiny_authors(["a1"])
    result = build_regression_input(
        author_articles=articles, authors=authors, article_impacts=impact
    )
    assert result.loc[0, "med_share"] == 0.5
    assert result.loc[0, "comp_sci_share"] == 0.5
    assert result.loc[0, "health_sci_share"] == 0.5
    assert result.loc[0, "phys_sci_share"] == 0.5


def test_official_comma_containing_impact_labels_map_without_unmatched_values() -> None:
    expected = {
        "Ecology, Evolution, Behavior and Systematics": {
            1100: "Agricultural and Biological Sciences"
        },
        "Endocrinology, Diabetes and Metabolism": {2700: "Medicine"},
        "Pediatrics, Perinatology and Child Health": {2700: "Medicine"},
        "Geography, Planning and Development": {3300: "Social Sciences"},
        "Renewable Energy, Sustainability and the Environment": {2100: "Energy"},
        "Safety, Risk, Reliability and Quality": {2200: "Engineering"},
        "Electronic, Optical and Magnetic Materials": {2500: "Materials Science"},
        "Surfaces, Coatings and Films": {2500: "Materials Science"},
        "Radiology, Nuclear Medicine and Imaging": {2700: "Medicine"},
        "Atomic and Molecular Physics, and Optics": {3100: "Physics and Astronomy"},
        "Physical Therapy, Sports Therapy and Rehabilitation": {3600: "Health Professions"},
    }
    for label, expected_fields in expected.items():
        assert impact_asjc_subject_area(label) == expected_fields
        assert unmatched_impact_class_name_labels(label) == ()


def test_short_impact_labels_remain_distinct_from_compound_label_mappings() -> None:
    assert impact_asjc_subject_area("Ecology") == {2300: "Environmental Science"}
    assert impact_asjc_subject_area("Endocrinology") == {
        1300: "Biochemistry, Genetics and Molecular Biology"
    }
    assert impact_asjc_subject_area("Pediatrics") == {2900: "Nursing"}


def test_broad_field_shares_are_fractional() -> None:
    data = pd.DataFrame(
        {
            "sc_id": [1, 1, 2],
            "eid": ["p1", "p2", "p3"],
            "impact_asjc_subject_area": [
                {2700: "Medicine", 3200: "Psychology"},
                {1000: "Multidisciplinary"},
                {1700: "Computer Science", 2200: "Engineering"},
            ],
        }
    )
    shares = professor_field_shares(data, field_column="impact_asjc_subject_area").shares.set_index(
        "sc_id"
    )
    assert shares.loc[1, "health_sci_share"] == 0.25
    assert shares.loc[1, "soc_hum_share"] == 0.25
    assert shares.loc[1, "multi_area_share"] == 0.5
    assert shares.loc[2, "phys_sci_share"] == 1.0
    assert np.allclose(shares.loc[:, list(DOMAIN_SHARE_COLUMNS)].sum(axis=1), 1.0)


def test_professor_field_shares_add_broad_area_shares() -> None:
    data = pd.DataFrame(
        {
            "sc_id": [1, 1, 2],
            "eid": ["p1", "p2", "p3"],
            "impact_asjc_subject_area": [
                {2700: "Medicine", 3200: "Psychology"},
                {1000: "Multidisciplinary"},
                {1700: "Computer Science", 2200: "Engineering"},
            ],
        }
    )
    result = professor_field_shares(data, field_column="impact_asjc_subject_area")
    shares = result.shares.set_index("sc_id")
    assert shares.loc[1, "health_sci_share"] == 0.25
    assert shares.loc[1, "soc_hum_share"] == 0.25
    assert shares.loc[1, "multi_area_share"] == 0.5
    assert shares.loc[2, "phys_sci_share"] == 1.0
    assert np.allclose(shares.loc[:, list(DOMAIN_SHARE_COLUMNS)].sum(axis=1), 1.0)


def test_field_share_terms_omit_one_reference_category() -> None:
    terms = included_field_share_terms(pd.DataFrame({"med_share": [0.8], "psych_share": [0.2]}))
    assert "med_share" not in terms
    assert set(terms) == {"psych_share"}


def test_domain_share_terms_omit_one_reference_category() -> None:
    data = pd.DataFrame(
        {column: [1.0 if i == 0 else 0.0] for i, column in enumerate(DOMAIN_SHARE_COLUMNS)}
    )
    terms = included_domain_share_terms(data)
    assert len(terms) == len(DOMAIN_SHARE_COLUMNS) - 1
    assert "life_sci_share" not in terms


def test_professor_article_input_deduplicates_only_researcher_publication_pairs() -> None:
    articles = pd.DataFrame(
        {
            "eid": ["shared", "shared", "solo"],
            "doi": ["10/shared"] * 2 + ["10/solo"],
            "author_ids": ["a1", "a2", "a1"],
            "hazen_perc_med": [75.0, 75.0, 25.0],
            "leiden_p_top_1": [0.0] * 3,
            "leiden_p_top_5": [1.0, 1.0, 0.0],
            "leiden_p_top_10": [1.0, 1.0, 0.0],
        }
    )
    authors = tiny_authors(["a1", "a2"])
    authors["sc_id"] = [1, 2]
    authors["person_title"] = ["Univ.-Prof. Dr.", "Assistenzprofessorin"]
    authors["is_inbreeding_proxy"] = [False, True]
    result = build_professor_article_input(author_articles=articles, authors=authors)
    assert result[["eid", "sc_id", "title", "is_inbred"]].to_dict(orient="records") == [
        {"eid": "shared", "sc_id": 1, "title": "Full Prof.", "is_inbred": False},
        {"eid": "shared", "sc_id": 2, "title": "Ass. Prof.", "is_inbred": True},
        {"eid": "solo", "sc_id": 1, "title": "Full Prof.", "is_inbred": False},
    ]
    assert result.loc[result.eid.eq("shared"), "doi"].tolist() == ["10/shared"] * 2


def test_professor_article_input_allows_missing_subject_counts() -> None:
    articles = pd.DataFrame(
        {
            "eid": ["p1"],
            "doi": ["10/p1"],
            "author_ids": ["a1"],
            "hazen_perc_med": [75.0],
            "leiden_p_top_1": [0.0],
            "leiden_p_top_5": [1.0],
            "leiden_p_top_10": [1.0],
        }
    )
    result = build_professor_article_input(author_articles=articles, authors=tiny_authors(["a1"]))
    assert result.loc[0, "title"] == "Full Prof."
    assert "primary_field" not in result


def test_professor_article_input_explodes_authors_and_merges_impact_data() -> None:
    articles = pd.DataFrame(
        {
            "eid": ["shared"],
            "doi": ["10/shared"],
            "author_ids": ["1;2;3"],
            "subject_areas": ['[{"area": "Medicine", "code": 2700}]'],
        }
    )
    impact = pd.DataFrame(
        {
            "item_id": ["shared"],
            "class_name": ["{Medicine,Oncology}"],
            "hazen_perc_med": [75.0],
            "ptop1_frac": [0.0],
            "ptop5_frac": [1.0],
            "ptop10_frac": [1.0],
        }
    )
    authors = tiny_authors(["1", "2"])
    authors["sc_id"] = [101, 102]
    authors["person_title"] = ["Univ.-Prof. Dr.", "Ass.-Prof. Dr."]
    authors["is_inbreeding_proxy"] = [False, True]
    result = build_professor_article_input(
        author_articles=articles, authors=authors, article_impacts=impact
    )
    assert result[["eid", "sc_id", "title", "hazen_perc_med", "pp_top_10"]].to_dict(
        orient="records"
    ) == [
        {
            "eid": "shared",
            "sc_id": 101,
            "title": "Full Prof.",
            "hazen_perc_med": 75.0,
            "pp_top_10": 1.0,
        },
        {
            "eid": "shared",
            "sc_id": 102,
            "title": "Ass. Prof.",
            "hazen_perc_med": 75.0,
            "pp_top_10": 1.0,
        },
    ]
    assert result["ptop10_frac"].tolist() == [1.0, 1.0]
    assert result["impact_asjc_subject_area"].tolist() == [{2700: "Medicine"}] * 2


def test_required_formula_columns_include_categorical_and_squared_terms() -> None:
    formula = regression_model_specs()[6].formula("pp_top_10")
    assert required_columns_from_formula(formula) == [
        "academic_age",
        "coaffiliation_publication_share",
        "is_inbred",
        "mean_coauthors_per_publication",
        "n_away_years_abroad",
        "n_away_years_domestic",
        "pp_top_10",
        "title",
    ]


def test_default_model_specs_match_source_order_and_terms() -> None:
    specs = regression_model_specs()
    assert [spec.model_id for spec in specs] == list(PAPER_REGRESSION_MODEL_TERMS)
    assert len(specs) == 10
    assert specs[-1] == RegressionModelSpec("m10", PAPER_REGRESSION_MODEL_TERMS["m10"])
    assert FIELD_SHARE_TERMS in specs[-1].terms
    assert "C(primary_field)" not in PAPER_REGRESSION_MODEL_TERMS["m8"]
    assert "C(primary_field)" not in PAPER_REGRESSION_MODEL_TERMS["m9"]


def test_regression_suite_excludes_configured_titles() -> None:
    data = pd.DataFrame(
        {
            "outcome": [0.1, 0.2, 0.25, 0.35, 0.5],
            "academic_age": [1, 2, 3, 4, 5],
            "title": [
                "Ao. Prof.",
                "Full Prof.",
                "FH-Ass. Prof.",
                "Ass. Prof.",
                "FH-Prof.",
            ],
        }
    )
    result = RegressionSuite(
        data=data,
        config=RegressionSuiteConfig(
            dependent_variables=("outcome",),
            model_specs=(RegressionModelSpec("m1", ("academic_age",)),),
            exclude_titles=("FH-Ass. Prof.",),
        ),
    ).fit()
    assert result.fit_statistics.loc[0, "n_obs"] == 4


@pytest.mark.parametrize("family", ["ols", "olshc3", "huber", "frac", "ppml", "nb", "pr", "zinb"])
def test_suite_fits_configured_model_families(family: str) -> None:
    rng = np.random.default_rng(2026)
    x = rng.normal(size=500)
    mean = np.exp(0.4 + 0.2 * x)
    y = rng.negative_binomial(2, 2 / (2 + mean))
    outcome_name = "share" if family == "frac" else "count"
    if family == "frac":
        y = np.clip(1 / (1 + np.exp(-(-0.3 + 0.2 * x))) + rng.normal(0, 0.02, size=len(x)), 0, 1)
    result = RegressionSuite(
        data=pd.DataFrame({outcome_name: y, "x": x}),
        config=RegressionSuiteConfig(
            dependent_variables=(outcome_name,),
            model_specs=(RegressionModelSpec("m1", ("x",)),),
            model_types={outcome_name: family},
            zinb_inflation_formulas={outcome_name: "1"},
            exclude_titles=(),
        ),
    ).fit()
    fitted = result.models[(outcome_name, "m1")]
    assert fitted.model_type == family
    assert set(result.coefficients.term) >= {"Intercept", "x"}
    assert result.summaries[f"{outcome_name}_m1"]
    if family == "olshc3":
        assert fitted.fit.cov_type == "HC3"
        assert "HC3 (heteroskedasticity-robust)" in fitted.summary_text
    if family == "zinb":
        assert fitted.fit.mle_retvals["converged"]
    if family == "nb":
        assert fitted.fit.cov_type == "HC3"
        assert fitted.mle_diagnostics is not None
        assert fitted.mle_diagnostics["mle_alpha_start"] != pytest.approx(0.5)
        assert fitted.mle_diagnostics["mle_alpha_start"] != pytest.approx(
            fitted.mle_diagnostics["mle_alpha_estimate"]
        )


def test_mle_diagnostics_reject_large_gradient_even_when_converged() -> None:
    fit = SimpleNamespace(
        mle_retvals={"converged": True, "iterations": 1, "fcalls": 3},
        mle_settings={"maxiter": 1000, "optimizer": "lbfgs"},
        params=np.array([1.0]),
        bse=np.array([0.1]),
        llf=-5.0,
        nobs=10,
        model=SimpleNamespace(score=lambda _: np.array([1.0])),
    )
    diagnostics = mle_fit_diagnostics(fit, gradient_tolerance=1e-4)
    assert diagnostics["mle_converged"] is True
    assert diagnostics["mle_iterations"] == 1
    assert diagnostics["mle_function_calls"] == 3
    assert diagnostics["mle_gradient_norm"] == pytest.approx(0.1)
    with pytest.raises(RegressionFitError, match="gradient norm"):
        validate_mle_fit(fit, model_id="m7", dependent_variable="n_public", model_type="nb")


def test_mle_warning_policy_retains_diagnostic_failure() -> None:
    fit = SimpleNamespace(
        mle_retvals={"converged": True, "iterations": 1},
        mle_settings={"maxiter": 1000, "optimizer": "lbfgs"},
        params=np.array([1.0]),
        bse=np.array([0.1]),
        llf=-5.0,
        nobs=10,
        model=SimpleNamespace(score=lambda _: np.array([1.0])),
    )
    with pytest.warns(RuntimeWarning, match="gradient norm"):
        diagnostic = validate_mle_fit(
            fit,
            model_id="m7",
            dependent_variable="n_public",
            model_type="nb",
            policy="warn",
        )
    assert "gradient norm" in diagnostic["mle_failure_reasons"]


def test_best_fit_table_allows_non_likelihood_estimators_without_aic_bic() -> None:
    fit_statistics = pd.DataFrame(
        {
            "dependent_variable": ["outcome", "outcome"],
            "model_id": ["m8", "m9"],
            "aic": [np.nan, np.nan],
            "bic": [np.nan, np.nan],
        }
    )

    best = best_models_by_metric(fit_statistics, metric="AIC")

    assert best.empty
    assert list(best.columns) == [
        "dependent_variable",
        "model",
        "AIC",
        "BIC",
        "AIC_rank",
        "BIC_rank",
    ]


def test_poisson_pseudo_likelihood_accepts_fractional_counts() -> None:
    fit = fit_regression_model(
        model_type="pr",
        formula="count ~ x",
        data=pd.DataFrame({"count": [0.0, 0.5, 1.0, 1.5, 2.0, 0.0, 0.5, 1.0], "x": np.arange(8.0)}),
    )
    assert fit.nobs == 8


def test_huber_regression_matches_statsmodels_reference_with_robust_covariance() -> None:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    x = np.linspace(-2.0, 2.0, 80)
    y = 1.5 + 0.7 * x
    y[-1] += 35.0
    data = pd.DataFrame({"outcome": y, "x": x})

    fitted = fit_regression_model(model_type="huber", formula="outcome ~ x", data=data)
    expected = smf.rlm(formula="outcome ~ x", data=data, M=sm.robust.norms.HuberT()).fit()

    assert type(fitted.model).__name__ == "RLM"
    pd.testing.assert_series_equal(fitted.params, expected.params)
    pd.testing.assert_series_equal(fitted.bse, expected.bse)
    assert fitted.bse["x"] != pytest.approx(
        fit_regression_model(model_type="ols", formula="outcome ~ x", data=data).bse["x"]
    )


def test_fractional_logit_matches_hc3_reference_and_rejects_out_of_range() -> None:
    import statsmodels.api as sm
    from patsy import dmatrices

    x = np.linspace(-1.5, 1.5, 100)
    y = 1.0 / (1.0 + np.exp(-(-0.4 + 0.9 * x))) + 0.01 * np.sin(x * 7)
    data = pd.DataFrame({"share": y, "x": x})
    fitted = fit_regression_model(model_type="frac", formula="share ~ x", data=data)
    endog, exog = dmatrices("share ~ x", data, return_type="dataframe")
    expected = sm.GLM(
        endog.iloc[:, 0],
        exog,
        family=sm.families.Binomial(link=sm.families.links.Logit()),
    ).fit(cov_type="HC3")

    pd.testing.assert_series_equal(fitted.params, expected.params)
    pd.testing.assert_series_equal(fitted.bse, expected.bse)
    assert fitted.cov_type == "HC3"
    rows = {row["term"]: row for row in coefficient_rows_from_fit(fitted, "share", "m1")}
    for term in fitted.params.index:
        assert rows[term]["conf_low"] == pytest.approx(expected.conf_int().loc[term, 0])
        assert rows[term]["conf_high"] == pytest.approx(expected.conf_int().loc[term, 1])
        assert rows[term]["p_value"] == pytest.approx(expected.pvalues[term])
    with pytest.raises(ValueError, match=r"bounded in \[0, 1\]"):
        fit_regression_model(
            model_type="frac",
            formula="share ~ x",
            data=pd.DataFrame({"share": [0.1, 0.3, 1.01, 0.7], "x": range(4)}),
        )


def test_ppml_matches_robust_poisson_reference_and_accepts_fractional_counts() -> None:
    import statsmodels.api as sm
    from patsy import dmatrices

    x = np.linspace(-1.0, 1.0, 100)
    y = np.exp(0.25 + 0.4 * x) + (np.arange(100) % 3) / 4
    data = pd.DataFrame({"count": y, "x": x})
    fitted = fit_regression_model(model_type="ppml", formula="count ~ x", data=data)
    endog, exog = dmatrices("count ~ x", data, return_type="dataframe")
    expected = sm.GLM(endog.iloc[:, 0], exog, family=sm.families.Poisson()).fit(cov_type="HC3")

    pd.testing.assert_series_equal(fitted.params, expected.params)
    pd.testing.assert_series_equal(fitted.bse, expected.bse)
    assert fitted.cov_type == "HC3"
    rows = {row["term"]: row for row in coefficient_rows_from_fit(fitted, "count", "m1")}
    for term in fitted.params.index:
        assert rows[term]["conf_low"] == pytest.approx(expected.conf_int().loc[term, 0])
        assert rows[term]["conf_high"] == pytest.approx(expected.conf_int().loc[term, 1])
        assert rows[term]["p_value"] == pytest.approx(expected.pvalues[term])
    with pytest.raises(ValueError, match="nonnegative"):
        fit_regression_model(
            model_type="ppml",
            formula="count ~ x",
            data=pd.DataFrame({"count": [0.2, 0.5, -0.1, 1.0], "x": range(4)}),
        )


def test_estimator_response_preprocessing_preserves_source_and_nb_rounds_half_up() -> None:
    source = pd.DataFrame({"P_top10": [3.7, 3.2, 2.5, 2.4], "x": [1.5, 2.5, 3.5, 4.5]})
    original = source.copy(deep=True)

    nb_data, note, scale = prepare_model_estimation_data(
        model_type="nb", formula="P_top10 ~ x", data=source
    )
    ppml_data, ppml_note, ppml_scale = prepare_model_estimation_data(
        model_type="ppml", formula="P_top10 ~ x", data=source
    )

    assert nb_data["P_top10"].tolist() == [4.0, 3.0, 3.0, 2.0]
    assert nb_data["x"].tolist() == source["x"].tolist()
    assert "half-up" in note
    assert scale == 1.0
    pd.testing.assert_frame_equal(ppml_data, source)
    assert ppml_note == "none" and ppml_scale == 1.0
    pd.testing.assert_frame_equal(source, original)

    model_input = pd.DataFrame(
        {
            "P_top10": np.tile([0.2, 1.3, 2.5, 3.7, 4.4], 20),
            "x": np.linspace(-1, 1, 100),
        }
    )
    original_model_input = model_input.copy(deep=True)
    fitted = fit_regression_model(model_type="nb", formula="P_top10 ~ x", data=model_input)
    np.testing.assert_array_equal(fitted.model.endog, np.floor(model_input.P_top10 + 0.5))
    assert "half-up" in fitted.model._acib_outcome_transformation
    pd.testing.assert_frame_equal(model_input, original_model_input)
    pd.testing.assert_frame_equal(source, original)


def test_estimator_preprocessing_accepts_fractional_endpoints_and_scales_mwpr() -> None:
    data = pd.DataFrame({"hazen_perc_med": [0.0, 40.0, 100.0], "x": [1.0, 2.0, 3.0]})
    frac_data, note, scale = prepare_model_estimation_data(
        model_type="frac", formula="hazen_perc_med ~ x", data=data
    )

    assert frac_data["hazen_perc_med"].tolist() == [0.0, 0.4, 1.0]
    assert scale == 100.0
    assert "divided by 100" in note
    pd.testing.assert_series_equal(
        data["hazen_perc_med"],
        pd.Series([0.0, 40.0, 100.0], name="hazen_perc_med"),
    )


def test_fractional_logit_keeps_exact_zero_and_one_observations() -> None:
    rng = np.random.default_rng(523)
    x = rng.normal(size=140)
    y = np.clip(1 / (1 + np.exp(-(-0.3 + 0.4 * x))) + rng.normal(0, 0.2, len(x)), 0, 1)
    y[::12] = 0.0
    y[5::12] = 1.0
    data = pd.DataFrame({"share": y, "x": x})

    fitted = fit_regression_model(model_type="frac", formula="share ~ x", data=data)

    assert fitted.nobs == len(data)
    assert 0.0 in fitted.model.endog
    assert 1.0 in fitted.model.endog
    assert fitted.cov_type == "HC3"


@pytest.mark.parametrize(
    ("converged", "parameters", "covariance", "fitted_values"),
    [
        (False, [1.0], [[1.0]], [1.0]),
        (True, [np.inf], [[1.0]], [1.0]),
        (True, [1.0], [[np.nan]], [1.0]),
        (True, [1.0], [[1.0]], [np.inf]),
    ],
)
def test_ppml_numerical_diagnostics_reject_invalid_glm_fits(
    converged: bool,
    parameters: list[float],
    covariance: list[list[float]],
    fitted_values: list[float],
) -> None:
    fit = SimpleNamespace(
        converged=converged,
        params=np.asarray(parameters),
        cov_params=lambda: np.asarray(covariance),
        fittedvalues=np.asarray(fitted_values),
    )
    with pytest.raises(RegressionFitError, match="ppml"):
        _validate_glm_fit(fit, "ppml")


@pytest.mark.parametrize("family", ("huber", "frac", "ppml"))
def test_alternative_estimators_keep_the_ols_sample_and_scale_mwpr(family: str) -> None:
    rng = np.random.default_rng(9812)
    n = 100
    age = rng.normal(size=n)
    outcome = 52.0 + age * 4 + rng.normal(size=n)
    outcome[[5, 22]] = np.nan
    age[7] = np.nan
    data = pd.DataFrame({"hazen_perc_med": outcome, "academic_age": age})
    spec = (RegressionModelSpec("m1", ("academic_age",)),)

    ols = (
        RegressionSuite(
            data=data,
            config=RegressionSuiteConfig(
                dependent_variables=("hazen_perc_med",),
                model_specs=spec,
                model_types={"hazen_perc_med": "ols"},
                exclude_titles=(),
            ),
        )
        .fit()
        .models[("hazen_perc_med", "m1")]
    )
    alternative = (
        RegressionSuite(
            data=data,
            config=RegressionSuiteConfig(
                dependent_variables=("hazen_perc_med",),
                model_specs=spec,
                model_types={"hazen_perc_med": family},
                exclude_titles=(),
            ),
        )
        .fit()
        .models[("hazen_perc_med", "m1")]
    )

    assert alternative.model_data.index.tolist() == ols.model_data.index.tolist()
    expected_outcome = ols.model_data["hazen_perc_med"]
    if family == "frac":
        expected_outcome = expected_outcome / 100.0
    np.testing.assert_allclose(alternative.fit.model.endog, expected_outcome)
    pd.testing.assert_series_equal(
        alternative.model_data["hazen_perc_med"], ols.model_data["hazen_perc_med"]
    )


@pytest.mark.parametrize(
    ("family", "outcome_column"),
    (("frac", "share"), ("ppml", "count")),
)
def test_glm_ame_and_delta_method_match_independent_calculation(
    family: str, outcome_column: str
) -> None:
    from patsy import build_design_matrices
    from scipy import stats

    rng = np.random.default_rng(472)
    n = 180
    age = rng.normal(size=n)
    inbred = np.arange(n) % 2 == 0
    probability = 1 / (1 + np.exp(-(-0.8 + 0.3 * age + 0.65 * inbred)))
    fractional_outcome = np.clip(probability + rng.normal(scale=0.025, size=n), 0.01, 0.99)
    count_outcome = rng.poisson(np.exp(-0.2 + 0.25 * age + 0.45 * inbred))
    outcome = fractional_outcome if family == "frac" else count_outcome
    data = pd.DataFrame({outcome_column: outcome, "academic_age": age, "is_inbred": inbred})
    fitted = (
        RegressionSuite(
            data=data,
            config=RegressionSuiteConfig(
                dependent_variables=(outcome_column,),
                model_specs=(RegressionModelSpec("m1", ("academic_age", "C(is_inbred)")),),
                model_types={outcome_column: family},
                exclude_titles=(),
            ),
        )
        .fit()
        .models[(outcome_column, "m1")]
    )

    design_info = fitted.fit.model._acib_design_info
    scenarios = []
    for value in (False, True):
        frame = fitted.model_data.copy()
        frame["is_inbred"] = value
        scenarios.append(np.asarray(build_design_matrices([design_info], frame)[0], dtype=float))
    params = np.asarray(fitted.fit.params, dtype=float)

    def contrast(beta: np.ndarray) -> float:
        predicted_inbred = fitted.fit.model.predict(beta, exog=scenarios[1])
        predicted_not_inbred = fitted.fit.model.predict(beta, exog=scenarios[0])
        return float(np.mean(predicted_inbred - predicted_not_inbred))

    estimate = contrast(params)
    step_size = 1e-5
    gradient = np.empty_like(params)
    for index in range(len(params)):
        upper, lower = params.copy(), params.copy()
        upper[index] += step_size
        lower[index] -= step_size
        gradient[index] = (contrast(upper) - contrast(lower)) / (2 * step_size)
    standard_error = float(np.sqrt(gradient @ np.asarray(fitted.fit.cov_params()) @ gradient))
    z_value = estimate / standard_error
    ame = fitted.coefficients.query("term == 'AME[is_inbred]'").iloc[0]

    assert ame.estimate == pytest.approx(estimate, rel=1e-7)
    assert ame.std_error == pytest.approx(standard_error, rel=1e-5)
    assert ame.p_value == pytest.approx(2 * stats.norm.sf(abs(z_value)), rel=1e-5)
    assert ame.conf_low == pytest.approx(estimate - stats.norm.ppf(0.975) * standard_error)
    assert ame.conf_high == pytest.approx(estimate + stats.norm.ppf(0.975) * standard_error)


def test_negative_binomial_ames_match_independent_binary_and_continuous_references() -> None:
    from patsy import build_design_matrices
    from scipy import stats

    from acib.regression.latex import LatexModelColumn

    rng = np.random.default_rng(176)
    count = 260
    age = rng.normal(size=count)
    inbred = np.arange(count) % 2 == 0
    titles = np.asarray(["Full Prof.", "Ao. Prof.", "Ass. Prof."])[np.arange(count) % 3]
    mean = np.exp(-0.2 + 0.2 * age + 0.35 * inbred + 0.2 * (titles == "Ao. Prof."))
    outcome = rng.negative_binomial(2.5, 2.5 / (2.5 + mean))
    data = pd.DataFrame(
        {"count": outcome, "academic_age": age, "is_inbred": inbred, "title": titles}
    )
    model = (
        RegressionSuite(
            data=data,
            config=RegressionSuiteConfig(
                dependent_variables=("count",),
                model_specs=(
                    RegressionModelSpec("m1", ("academic_age", "C(is_inbred)", "C(title)")),
                ),
                model_types={"count": "nb"},
                exclude_titles=(),
            ),
        )
        .fit()
        .models[("count", "m1")]
    )
    data_design = model.fit.model._acib_design_info

    def mean_prediction(frame: pd.DataFrame, beta: np.ndarray) -> float:
        design = build_design_matrices([data_design], frame)[0]
        return float(np.mean(model.fit.model.predict(beta, exog=design)))

    parameter = np.asarray(model.fit.params, dtype=float)
    gradient = np.empty_like(parameter)
    step = 1e-5
    for index in range(parameter.size):
        high, low = parameter.copy(), parameter.copy()
        high[index] += step
        low[index] -= step
        high_data, low_data = model.model_data.copy(), model.model_data.copy()
        high_data["is_inbred"] = True
        low_data["is_inbred"] = False
        gradient[index] = (
            mean_prediction(high_data, high)
            - mean_prediction(low_data, high)
            - mean_prediction(high_data, low)
            + mean_prediction(low_data, low)
        ) / (2 * step)
    binary_effect = mean_prediction(
        model.model_data.assign(is_inbred=True), parameter
    ) - mean_prediction(model.model_data.assign(is_inbred=False), parameter)
    covariance = np.asarray(model.fit.cov_params())
    binary_se = float(np.sqrt(gradient @ covariance @ gradient))
    binary_row = model.coefficients.query("term == 'AME[is_inbred]'").iloc[0]
    assert binary_row.estimate == pytest.approx(binary_effect, rel=1e-6)
    assert binary_row.std_error == pytest.approx(binary_se, rel=1e-4)
    assert binary_row.p_value == pytest.approx(
        2 * stats.norm.sf(abs(binary_effect / binary_se)), rel=1e-4
    )

    title_effect = LatexModelColumn("count", model).title_contrast(
        "Ao. Prof.", reference="Full Prof."
    )
    assert title_effect is not None
    title_high = model.model_data.assign(title="Ao. Prof.")
    title_low = model.model_data.assign(title="Full Prof.")

    def title_contrast(beta: np.ndarray) -> float:
        return mean_prediction(title_high, beta) - mean_prediction(title_low, beta)

    expected_title = title_contrast(parameter)
    assert title_effect.estimate == pytest.approx(expected_title, rel=1e-7)
    title_gradient = np.empty_like(parameter)
    for index in range(parameter.size):
        high, low = parameter.copy(), parameter.copy()
        high[index] += step
        low[index] -= step
        title_gradient[index] = (title_contrast(high) - title_contrast(low)) / (2 * step)
    title_se = float(np.sqrt(title_gradient @ covariance @ title_gradient))
    assert title_effect.std_error == pytest.approx(title_se, rel=1e-4)
    assert title_effect.p_value == pytest.approx(
        2 * stats.norm.sf(abs(expected_title / title_se)), rel=1e-4
    )


def test_fractional_logit_mwpr_marginal_effect_uses_percentage_scale() -> None:
    from patsy import build_design_matrices

    rng = np.random.default_rng(704)
    n = 120
    age = rng.normal(size=n)
    inbred = np.arange(n) % 2 == 0
    share = 1 / (1 + np.exp(-(-0.7 + 0.2 * age + 0.4 * inbred)))
    data = pd.DataFrame(
        {
            "hazen_perc_med": (share + rng.normal(scale=0.015, size=n)) * 100,
            "academic_age": age,
            "is_inbred": inbred,
        }
    )
    fitted = (
        RegressionSuite(
            data=data,
            config=RegressionSuiteConfig(
                dependent_variables=("hazen_perc_med",),
                model_specs=(RegressionModelSpec("m1", ("academic_age", "C(is_inbred)")),),
                model_types={"hazen_perc_med": "frac"},
                exclude_titles=(),
            ),
        )
        .fit()
        .models[("hazen_perc_med", "m1")]
    )
    ame = fitted.coefficients.query("term == 'AME[is_inbred]'").iloc[0]
    scenarios = []
    for value in (False, True):
        frame = fitted.model_data.copy()
        frame["is_inbred"] = value
        scenarios.append(
            np.asarray(
                build_design_matrices([fitted.fit.model._acib_design_info], frame)[0],
                dtype=float,
            )
        )
    predictions = [fitted.fit.model.predict(fitted.fit.params, exog=x) for x in scenarios]
    expected_percentage_point_change = 100 * float(np.mean(predictions[1] - predictions[0]))
    assert ame.estimate == pytest.approx(expected_percentage_point_change)
    assert 0 < abs(ame.estimate) < 100


def test_model7_fractional_ame_differentiates_quadratic_age_and_hides_square_row() -> None:
    from patsy import build_design_matrices

    from acib.regression.latex import LatexModelColumn

    rng = np.random.default_rng(619)
    count = 120
    age = rng.uniform(1, 40, size=count)
    inbred = np.arange(count) % 2 == 0
    probability = 1 / (1 + np.exp(-(-1 + 0.03 * age - 0.0005 * age**2 + 0.3 * inbred)))
    data = pd.DataFrame(
        {
            "share": np.clip(probability + rng.normal(0, 0.02, count), 0, 1),
            "academic_age": age,
            "is_inbred": inbred,
        }
    )
    spec = RegressionModelSpec("m7", ("academic_age", "I(academic_age ** 2)", "C(is_inbred)"))
    model = (
        RegressionSuite(
            data=data,
            config=RegressionSuiteConfig(
                dependent_variables=("share",),
                model_specs=(spec,),
                model_types={"share": "frac"},
                exclude_titles=(),
            ),
        )
        .fit()
        .models[("share", "m7")]
    )

    step = 1e-4
    low = model.model_data.copy()
    high = model.model_data.copy()
    low["academic_age"] -= step
    high["academic_age"] += step
    high_design = build_design_matrices([model.fit.model._acib_design_info], high)[0]
    low_design = build_design_matrices([model.fit.model._acib_design_info], low)[0]
    expected = np.mean(
        model.fit.model.predict(model.fit.params, exog=high_design)
        - model.fit.model.predict(model.fit.params, exog=low_design)
    ) / (2 * step)
    effects = model.coefficients
    age_effect = effects.query("term == 'AME[academic_age]'").iloc[0]
    assert age_effect.estimate == pytest.approx(expected, rel=1e-5)
    assert "I(academic_age ** 2)" in model.fit.params.index
    assert LatexModelColumn("share", model).term("I(academic_age ** 2)") is None
    assert not effects.query("term == 'AME[I(academic_age ** 2)]'").shape[0]


def test_ols_hc3_and_alternative_results_write_to_separate_files(
    tmp_path: Path,
) -> None:
    from acib.regression.config import RegressionOutputs

    rng = np.random.default_rng(143)
    n = 100
    x = rng.normal(size=n)
    inbred = np.arange(n) % 2 == 0
    data = pd.DataFrame(
        {
            "share": 1.0 / (1.0 + np.exp(-(-0.6 + 0.4 * x + 0.3 * inbred)))
            + rng.normal(scale=0.01, size=n),
            "count": rng.poisson(np.exp(0.3 + 0.2 * x + 0.25 * inbred)),
            "x": x,
            "is_inbred": inbred,
        }
    )
    result = RegressionSuite(
        data=data,
        config=RegressionSuiteConfig(
            dependent_variables=("share", "count"),
            model_specs=(RegressionModelSpec("m1", ("x", "C(is_inbred)")),),
            model_types={"share": "olshc3", "count": "ppml"},
            relative_performance_outcomes=("share",),
            absolute_performance_outcomes=("count",),
            outcome_aliases={"share": "Share", "count": "Count"},
            exclude_titles=(),
        ),
    ).fit()
    expected_hc3 = result.models[("share", "m1")].fit.bse["x"]
    outputs = RegressionOutputs(directory=tmp_path)

    result.write(outputs, data=data)

    ols_coefficients = pd.read_csv(outputs.model_coefficients)
    robustness_coefficients = pd.read_csv(outputs.robustness_model_coefficients)
    ols_fit_statistics = pd.read_csv(outputs.model_fit_statistics)
    robustness_fit_statistics = pd.read_csv(outputs.robustness_model_fit_statistics)
    assert set(ols_coefficients["model_type"]) == {"olshc3"}
    assert set(robustness_coefficients["model_type"]) == {"ppml"}
    assert set(ols_fit_statistics["model_type"]) == {"olshc3"}
    assert set(robustness_fit_statistics["model_type"]) == {"ppml"}
    assert robustness_coefficients["term"].str.startswith("AME[").any()
    assert ols_coefficients.loc[ols_coefficients.term == "x", "std_error"].iloc[0] == pytest.approx(
        expected_hc3
    )


def test_olshc3_uses_hc3_standard_errors_without_changing_ols_coefficients() -> None:
    rng = np.random.default_rng(341)
    x = np.linspace(-2.0, 2.0, 300)
    noise = rng.normal(scale=0.15 + np.abs(x), size=len(x))
    data = pd.DataFrame({"outcome": 0.5 + 1.2 * x + noise, "x": x})

    ordinary = fit_regression_model(model_type="ols", formula="outcome ~ x", data=data)
    robust = fit_regression_model(
        model_type="olshc3",
        formula="outcome ~ x",
        data=data,
        covariance_type="HC0",
    )

    assert robust.cov_type == "HC3"
    pd.testing.assert_series_equal(ordinary.params, robust.params)
    assert not np.allclose(ordinary.bse, robust.bse)


def test_latex_significance_and_model_inference_keep_model_family_semantics() -> None:
    assert format_p_value_cell(0.00001) == r"0.00\textsuperscript{***}"
    assert format_p_value_cell(0.032519) == r"0.03\textsuperscript{*}"
    assert format_p_value_cell(0.069892) == "0.07"
    assert model_inference_note({"n_public": "nb", "P_top10": "ols"}) == (
        "OLS uses conventional standard errors and $t$-test inference; "
        "negative-binomial inference uses "
        "$z$-tests and HC3 robust standard errors; negative-binomial tables report average "
        "marginal effects."
    )
    assert "OLS-HC3 uses HC3 heteroskedasticity-robust standard errors" in model_inference_note(
        {"pp_top_10": "olshc3"}
    )
    assert "fractional-logit AMEs" in model_inference_note({"pp_top_10": "frac"})
    assert "HuberT with the H1 covariance estimator" in model_inference_note(
        {"hazen_perc_med": "huber"}
    )
    assert "negative-binomial inference uses $z$-tests and HC3" in model_inference_note(
        {"n_public": "nb"}
    )


def test_latex_performance_template_uses_requested_captions_and_robust_se_header() -> None:
    relative = table_preamble(
        model_id="m9",
        outcomes=("hazen_perc_med", "pp_top_10", "pp_top_5", "pp_top_1"),
    )
    absolute = table_preamble(
        model_id="m9",
        outcomes=("n_public", "P_top10", "P_top5", "P_top1"),
        performance_type="absolute",
    )
    header = outcome_header(
        ("hazen_perc_med",),
        models=cast(
            Any,
            {("hazen_perc_med", "m9"): SimpleNamespace(model_type="olshc3")},
        ),
        model_id="m9",
    )

    assert (
        r"\caption{Association between academic inbreeding and relative citation impact}"
        in "\n".join(relative)
    )
    assert (
        r"\caption{Association between academic inbreeding and absolute research performance}"
        in "\n".join(absolute)
    )
    assert "S[table-format=-2.2]" in absolute
    assert "S[table-format=2.2]" in absolute
    assert r"\textbf{Robust}" in "\n".join(header)


def test_absolute_performance_note_distinguishes_nb_ames_from_ols_coefficients() -> None:
    columns = [
        LatexModelColumn(
            outcome,
            cast(
                Any,
                SimpleNamespace(
                    model_type="nb" if outcome == "n_public" else "olshc3",
                    formula="outcome ~ title + academic_age",
                    share_control_type="detailed_asjc_shares",
                    fit=SimpleNamespace(cov_type="HC3"),
                ),
            ),
        )
        for outcome in ("n_public", "P_top10", "P_top5", "P_top1")
    ]
    note = performance_table_note(
        columns=columns,
        outcomes=("n_public", "P_top10", "P_top5", "P_top1"),
        performance_type="absolute",
        model_id="m9",
    )

    assert note.startswith(r"\textit{Notes.} Reported are")
    assert "negative binomial regression (NB)" in note
    assert (
        "average marginal effects (AMEs) from negative binomial regression (NB) for Publications "
        "and ordinary least squares (OLS) regression coefficients (Coef.) for "
        r"P(top 10\%), P(top 5\%), and P(top 1\%)"
    ) in note
    assert "Publications AMEs are expressed in publication-count units" in note
    assert "number of publications per professor (Publications)" in note
    assert "OLS coefficients for the P indicators represent changes in fractionally counted" in note
    assert "OLS $p$-values are based on $t$-tests" in note
    assert "negative binomial AME $p$-values are based on $z$-tests" in note
    assert "Full Prof. (Full Professor) is the reference category" in note
    assert "Adjusted $R^2$" not in note


def test_regression_suite_sorts_printed_models_in_specification_order(
    capsys: CaptureFixture[str],
) -> None:
    data = pd.DataFrame({"outcome": np.linspace(0, 1, 40), "predictor": np.arange(40.0)})
    result = RegressionSuite(
        data=data,
        config=RegressionSuiteConfig(
            dependent_variables=("outcome",),
            model_specs=(
                RegressionModelSpec("m2", ("predictor",)),
                RegressionModelSpec("m1", ("predictor",)),
            ),
            exclude_titles=(),
        ),
    ).fit()
    result.print_model_summaries()
    output = capsys.readouterr().out
    assert "Outcome: outcome" in output
    assert output.index("M1") < output.index("M2")


def test_suite_aic_bic_rankings_retain_model_ids() -> None:
    data = pd.DataFrame(
        {
            "y": [0.1, 0.14, 0.18, 0.21, 0.24, 0.3, 0.34, 0.39],
            "age": np.arange(1, 9),
            "is_inbred": [False, True] * 4,
        }
    )
    result = RegressionSuite(
        data=data,
        config=RegressionSuiteConfig(
            dependent_variables=("y",),
            model_specs=(
                RegressionModelSpec("m1", ("age",)),
                RegressionModelSpec("m2", ("age", "C(is_inbred)")),
            ),
            exclude_titles=(),
        ),
    ).fit()
    assert list(result.aic_bic_ranked.columns) == [
        "dependent_variable",
        "model",
        "AIC",
        "BIC",
        "AIC_rank",
        "BIC_rank",
    ]
    assert set(result.best_by_aic.model) <= {"m1", "m2"}
    assert set(result.best_by_bic.model) <= {"m1", "m2"}


def test_model_comparison_table_reports_specification_and_fit_information() -> None:
    n = 40
    data = pd.DataFrame(
        {
            "pp_top_1": np.linspace(0.01, 0.07, n),
            "academic_age": np.arange(n) + 1,
            "is_inbred": np.arange(n) % 2 == 0,
        }
    )
    result = RegressionSuite(
        data=data,
        config=RegressionSuiteConfig(
            dependent_variables=("pp_top_1",),
            model_specs=(
                RegressionModelSpec("m1", ("academic_age",)),
                RegressionModelSpec("m2", ("academic_age", "C(is_inbred)")),
            ),
            exclude_titles=(),
        ),
    ).fit()
    table = model_comparison_latex_table(
        result.models, dependent_variable="pp_top_1", model_ids=("m1", "m2")
    )
    assert r"\textbf{M1}" in table and r"\textbf{M2}" in table
    assert "AIC" in table and "BIC" in table
    assert "Observations" not in table


def test_pipeline_settings_conversion_keeps_model_families_and_inflation_formula(
    tmp_path: Path,
) -> None:
    from acib.regression.analysis_config import (
        regression_suite_config_from_pipeline_config,
    )
    from acib.regression.config import RegressionConfig, RegressionOutputs

    config = RegressionConfig(
        article_author_file_path=tmp_path / "articles.csv",
        mobility_author_proxy=tmp_path / "authors.csv",
        outputs=RegressionOutputs(directory=tmp_path),
        dependent_variables={"n_public": "nb", "P_top10": "zinb"},
        zinb_inflation_formulas={"P_top10": "academic_age + C(title)"},
    )
    result = regression_suite_config_from_pipeline_config(config)
    assert result.dependent_variables == ("n_public", "P_top10")
    assert result.model_types == {"n_public": "nb", "P_top10": "zinb"}
    assert result.zinb_inflation_formulas == {"P_top10": "academic_age + C(title)"}
    assert result.zinb_regularization == 3.0


def test_negative_binomial_centers_first_year_and_reports_transform() -> None:
    from patsy import build_design_matrices

    rng = np.random.default_rng(702)
    count = 300
    age = rng.integers(1, 45, size=count).astype(float)
    year = rng.integers(1980, 2024, size=count).astype(float)
    mean = np.exp(0.8 + 0.01 * age + 0.015 * (year - year.mean()))
    outcome = rng.negative_binomial(2, 2 / (2 + mean))
    result = RegressionSuite(
        data=pd.DataFrame({"count": outcome, "academic_age": age, "first_year": year}),
        config=RegressionSuiteConfig(
            dependent_variables=("count",),
            model_specs=(RegressionModelSpec("m1", ("academic_age", "first_year")),),
            model_types={"count": "nb"},
            exclude_titles=(),
        ),
    ).fit()
    fitted = result.models[("count", "m1")]
    assert fitted.mle_diagnostics is not None
    assert fitted.mle_diagnostics["mle_failure_reasons"] == ""
    assert fitted.mle_diagnostics["mle_centered_first_year"] == pytest.approx(year.mean())
    assert result.formulas["count_m1_centering"].startswith("first_year = first_year - ")
    assert fitted._prediction_data()["first_year"].mean() == pytest.approx(0.0, abs=1e-12)

    effect = fitted.coefficients.query("term == 'AME[first_year]'").iloc[0]
    assert np.isfinite(effect[["estimate", "std_error", "p_value"]].to_numpy(dtype=float)).all()
    prediction_data = fitted.model_data.copy()
    prediction_data["first_year"] -= year.mean()
    step = 1e-5 * max(1.0, float(np.std(year)))
    high, low = prediction_data.copy(), prediction_data.copy()
    high["first_year"] += step
    low["first_year"] -= step
    design_info = fitted.fit.model._acib_design_info
    high_design = build_design_matrices([design_info], high)[0]
    low_design = build_design_matrices([design_info], low)[0]
    estimate = float(
        (
            fitted.fit.model.predict(fitted.fit.params, exog=high_design).mean()
            - fitted.fit.model.predict(fitted.fit.params, exog=low_design).mean()
        )
        / (2 * step)
    )
    assert effect.estimate == pytest.approx(estimate, rel=1e-6)

    params = np.asarray(fitted.fit.params, dtype=float)
    steps = 1e-5 * np.maximum(1.0, np.abs(params))
    gradient = np.empty_like(params)
    for index, parameter_step in enumerate(steps):
        upper, lower = params.copy(), params.copy()
        upper[index] += parameter_step
        lower[index] -= parameter_step
        upper_effect = (
            fitted.fit.model.predict(upper, exog=high_design).mean()
            - fitted.fit.model.predict(upper, exog=low_design).mean()
        ) / (2 * step)
        lower_effect = (
            fitted.fit.model.predict(lower, exog=high_design).mean()
            - fitted.fit.model.predict(lower, exog=low_design).mean()
        ) / (2 * step)
        gradient[index] = (upper_effect - lower_effect) / (2 * parameter_step)
    covariance = np.asarray(fitted.fit.cov_params(), dtype=float)
    expected_se = float(np.sqrt(gradient @ covariance @ gradient))
    assert effect.std_error == pytest.approx(expected_se, rel=1e-4)


def test_descriptive_statistics_keeps_group_sample_and_scaled_mwpr_summary() -> None:
    from acib.regression.descriptives import descriptive_statistics_latex_table

    data = pd.DataFrame(
        {
            "hazen_perc_med": [50.0, 60.0, 70.0],
            "pp_top_10": [0.1, 0.2, 0.3],
            "pp_top_5": [0.0, 0.1, 0.2],
            "pp_top_1": [0.0, 0.0, 0.03],
            "n_public": [4, 6, 8],
            "P_top10": [0.25, 0.5, 1.0],
            "P_top5": [0.0, 0.25, 0.5],
            "P_top1": [0.0, 0.0, 0.25],
            "academic_age": [10, 20, 30],
            "title": ["Full Prof.", "Ao. Prof.", "Full Prof."],
            "is_inbred": [False, True, True],
            "coaffiliation_publication_share": [0.0, np.nan, 0.2],
            "mean_coauthors_per_publication": [2.0, 4.0, 6.0],
            "n_away_years_domestic": [0, 1, 2],
            "n_away_years_abroad": [0, 0, 3],
            "first_year": [2000, 2005, 2010],
        }
    )
    table = descriptive_statistics_latex_table(data)
    assert (
        r"\caption{Descriptive statistics of the professor-level analytical sample by "
        r"academic-inbreeding status (\(n=3\))}" in table
    )
    assert r"\multicolumn{4}{c}{\textbf{Non-inbred}}" in table
    assert r"\multicolumn{4}{c}{\textbf{Inbred}}" in table
    assert r"Professors, \(n\) & \multicolumn{4}{l}{1} & \multicolumn{4}{l}{2}" in table
    assert (r"mwPR [\%] & 50.0 & -- & 50.0 & " "50.0 & 65.0 & 7.1 & 60.0 & 70.0") in table
    assert (r"PP(top 10\%) [\%] & 10.0 & -- & 10.0 & 10.0 & 25.0 & 7.1 & " r"20.0 & 30.0") in table
    assert r"Publications [count] & 4 & -- & 4 & 4 & 7 & 1 & 6 & 8" in table
    assert r"P(top 10\%) [count]" in table
    assert r"P(top 5\%) [count]" in table
    assert r"P(top 1\%) [count]" in table
    assert r"Co-affiliation publication share [\%]" in table
    assert "Mean coauthors per publication [-]" in table
    assert "Domestic mobility [years]" in table
    assert "International mobility [years]" in table
    assert "First publication year [year]" in table
    assert (
        r"\quad Full Prof. & \multicolumn{4}{l}{1 (100)} & " r"\multicolumn{4}{l}{1 (50)}" in table
    )
    assert r"\quad Ao. Prof. & \multicolumn{4}{l}{0 (0)} & " r"\multicolumn{4}{l}{1 (50)}" in table
    assert r"\textit{Notes.}" in table
    assert (
        "Continuous variables are summarized by mean, standard deviation (SD), minimum, and maximum"
        in table
    )
    assert "mean weighted percentile rank (mwPR) is reported in percentile points" in table
    # Target format puts absolute outcomes in the main variable sequence, with no source heading
    # or extra prose about fractional sums.
    assert r"\textbf{Absolute performance variables}" not in table
    assert (
        "fractional counts obtained by summing the corresponding "
        "publication-level fractional indicators" in table
    )
    assert (
        "Academic-title entries are counts with percentages within each "
        "academic-inbreeding group" not in table
    )
    assert "ASJC field composition is controlled" not in table
    assert "The external co-affiliation publication rate is available" not in table


def test_fit_regression_models_uses_configured_model_specs() -> None:
    data = pd.DataFrame(
        {
            "pp_top_10": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "is_inbred": [False, True, False, True, False, True],
            "academic_age": [1, 2, 3, 4, 5, 6],
        }
    )
    _, fit_statistics, formulas = fit_regression_models(
        data=data,
        dependent_variables=["pp_top_10"],
        model_specs=[RegressionModelSpec("custom", ("C(is_inbred)", "academic_age"))],
    )
    assert list(fit_statistics["model_id"]) == ["custom"]
    assert formulas == {"pp_top_10_custom": "pp_top_10 ~ C(is_inbred) + academic_age"}


def test_ols_estimates_match_original_implementation() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 4.0, 5.0],
            "age": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            "is_inbred": [False, False, False, False, True, True, True, True],
        }
    )

    coefficients, fit_statistics, formulas = fit_regression_models(
        data=data,
        dependent_variables=["y"],
        model_specs=[RegressionModelSpec("custom", ("C(is_inbred)", "age"))],
    )

    estimates = coefficients.set_index("term")["estimate"]
    assert estimates["Intercept"] == pytest.approx(-0.125, abs=1e-12)
    assert estimates["C(is_inbred)[T.True]"] == pytest.approx(0.75, abs=1e-12)
    assert estimates["age"] == pytest.approx(1.05, abs=1e-12)
    assert fit_statistics.loc[0, "n_obs"] == 8
    assert fit_statistics.loc[0, "r_squared"] == pytest.approx(0.9436893203883495)
    assert fit_statistics.loc[0, "aic"] == pytest.approx(11.494815204816376)
    assert fit_statistics.loc[0, "bic"] == pytest.approx(11.81258137153572)
    assert formulas == {"y_custom": "y ~ C(is_inbred) + age"}


def test_model_comparison_outcome_families_have_distinct_inference_notes() -> None:
    assert "negative-binomial inference uses $z$-tests and HC3" in model_inference_note(
        {"n_public": "nb"}
    )
    assert "OLS uses conventional standard errors" in model_inference_note({"pp_top_10": "ols"})


def test_configured_model_terms_preserve_model_10_omissions() -> None:
    assert "C(primary_field)" not in PAPER_REGRESSION_MODEL_TERMS["m8"]
    assert "C(primary_field)" not in PAPER_REGRESSION_MODEL_TERMS["m9"]
    assert "n_away_years_abroad" not in PAPER_REGRESSION_MODEL_TERMS["m10"]


def test_model9_latex_table_preserves_full_professor_and_field_reference_assertions() -> None:
    from acib.regression.constants import FIELD_SHARE_COLUMNS

    count = 80
    data = pd.DataFrame(
        {
            "hazen_perc_med": np.linspace(40, 70, count),
            "pp_top_10": np.linspace(0.05, 0.25, count),
            "pp_top_5": np.linspace(0.02, 0.15, count),
            "pp_top_1": np.linspace(0, 0.04, count),
            "is_inbred": [index % 3 == 0 for index in range(count)],
            "academic_age": np.arange(1, count + 1),
            "title": [
                ["Ao. Prof.", "Ass. Prof.", "FH-Prof.", "Full Prof."][i % 4] for i in range(count)
            ],
            "coaffiliation_publication_share": np.linspace(0, 1, count),
            "mean_coauthors_per_publication": np.linspace(1, 5, count),
            "n_away_years_domestic": [i % 5 for i in range(count)],
            "n_away_years_abroad": [i % 7 for i in range(count)],
            "first_year": np.arange(1980, 1980 + count),
        }
    )
    for share_column in FIELD_SHARE_COLUMNS:
        data[share_column] = 1.0 if share_column == "med_share" else 0.0
    result = RegressionSuite(
        data=data,
        config=RegressionSuiteConfig(
            dependent_variables=("hazen_perc_med", "pp_top_10", "pp_top_5", "pp_top_1"),
            model_specs=(RegressionModelSpec("m9", PAPER_REGRESSION_MODEL_TERMS["m9"]),),
            model_types=dict.fromkeys(
                ("hazen_perc_med", "pp_top_10", "pp_top_5", "pp_top_1"), "olshc3"
            ),
            exclude_titles=(),
        ),
    ).fit()
    table = result.model_9_latex_table
    assert table.startswith(r"\begin{table}[htbp]")
    assert (
        r"\caption{Association between academic inbreeding and relative citation impact (\(n="
        in table
    )
    assert r"\textbf{Robust}\\\textbf{SE}" in table
    assert r"Academic title (ref.: Full Prof.)" in table
    assert r"\quad Ao.\ Prof." in table
    assert r"ASJC field shares" in table
    assert r"27 major fields; Medicine as reference" in table
    assert table.count(r"27 major fields; Medicine as reference") == 1
    assert r"with Medicine as the reference field" in table
    assert r"Medicine is omitted" not in table
    assert r"Primary field" not in table


def test_each_outcome_selects_only_its_configured_estimator(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(827)
    count = 50
    inbred = np.arange(count) % 2 == 0
    age = rng.uniform(1, 40, size=count)
    data = pd.DataFrame(
        {
            "is_inbred": inbred,
            "academic_age": age,
            "hazen_perc_med": np.clip(48 + inbred * 3 + rng.normal(0, 8, count), 1, 99),
            "pp_top_10": rng.beta(2 + inbred, 6, size=count),
            "pp_top_5": rng.beta(1.5 + inbred, 8, size=count),
            "pp_top_1": rng.beta(1 + inbred, 12, size=count),
            "n_public": rng.negative_binomial(3, 3 / (3 + np.exp(1.4 + 0.2 * inbred))),
            "P_top10": rng.poisson(np.exp(-0.2 + 0.25 * inbred), size=count),
            "P_top5": rng.poisson(np.exp(-0.8 + 0.3 * inbred), size=count),
            "P_top1": rng.poisson(np.exp(-1.7 + 0.4 * inbred), size=count),
        }
    )
    outcomes = (
        "hazen_perc_med",
        "pp_top_10",
        "pp_top_5",
        "pp_top_1",
        "n_public",
        "P_top10",
        "P_top5",
        "P_top1",
    )
    model_types = {
        **dict.fromkeys(outcomes[:4], "frac"),
        "n_public": "nb",
        **dict.fromkeys(outcomes[5:], "ppml"),
    }
    model_specs = (RegressionModelSpec("m1", ("C(is_inbred)", "academic_age")),)
    result = RegressionSuite(
        data=data,
        config=RegressionSuiteConfig(
            dependent_variables=outcomes,
            model_specs=model_specs,
            model_types=model_types,
            exclude_titles=(),
        ),
    ).fit()
    assert set(result.models) == {(outcome, "m1") for outcome in outcomes}
    for outcome in outcomes:
        model = result.models[(outcome, "m1")]
        assert model.model_type == model_types[outcome]
        reference_ols = fit_regression_model(
            model_type="ols", formula=model.formula, data=model.model_data
        )
        np.testing.assert_allclose(model.fit.model.exog, reference_ols.model.exog)
        if model.model_type in {"frac", "ppml", "nb"}:
            assert model.fit.cov_type == "HC3"
            assert model.coefficients.effect_type.eq("AME").any()
    assert len(result.fit_statistics) == len(outcomes)
    assert set(result.aic_bic["model"]) == {model.model_id for model in result.models.values()}
    from acib.regression.config import RegressionOutputs

    outputs = RegressionOutputs(directory=tmp_path)
    result.write(outputs, data=data)
    robustness_coefficients = pd.read_csv(outputs.robustness_model_coefficients)
    robustness_statistics = pd.read_csv(outputs.robustness_model_fit_statistics)
    primary_coefficients = pd.read_csv(outputs.model_coefficients)
    generated_ids = {model.model_id for model in result.models.values()}
    assert generated_ids.issubset(set(robustness_coefficients.model_id))
    assert generated_ids.issubset(set(robustness_statistics.model_id))
    assert primary_coefficients.empty
    assert {"effect_type", "effect_units", "covariance_estimator"}.issubset(
        robustness_coefficients.columns
    )
    relative_table = (outputs.latex_dir / "research_performance_m1.tex").read_text()
    absolute_table = (outputs.latex_dir / "absolute_performance_m1.tex").read_text()
    assert r"\textbf{AME}" in relative_table
    assert r"\textbf{AME}" in absolute_table
    assert "fractional logit" in relative_table
    assert "fractional publication-count units" in absolute_table
    assert not list(outputs.latex_dir.glob("*frac*.tex"))
    assert not list(outputs.latex_dir.glob("*ppml*.tex"))


def test_mixed_estimator_performance_table_labels_each_outcome_correctly() -> None:
    from acib.regression.latex import research_performance_latex_table

    rng = np.random.default_rng(529)
    count = 100
    x = rng.normal(size=count)
    data = pd.DataFrame(
        {
            "linear_outcome": 1.0 + 0.5 * x + rng.normal(size=count),
            "fractional_outcome": np.clip(
                1 / (1 + np.exp(-(-0.5 + 0.3 * x))) + rng.normal(0, 0.03, count), 0, 1
            ),
            "academic_age": x,
            "is_inbred": np.arange(count) % 2 == 0,
            "title": ["Full Prof."] * count,
        }
    )
    outcomes = ("linear_outcome", "fractional_outcome")
    result = RegressionSuite(
        data=data,
        config=RegressionSuiteConfig(
            dependent_variables=outcomes,
            model_specs=(RegressionModelSpec("m1", ("academic_age",)),),
            model_types={"linear_outcome": "olshc3", "fractional_outcome": "frac"},
            relative_performance_outcomes=outcomes,
            absolute_performance_outcomes=(),
            outcome_aliases={
                "linear_outcome": "Linear",
                "fractional_outcome": "Fraction",
            },
            exclude_titles=(),
        ),
    ).fit()

    table = research_performance_latex_table(
        result.models,
        model_id="m1",
        outcomes=outcomes,
        outcome_aliases=result.outcome_aliases,
    )

    assert r"\textbf{Coef.}" in table
    assert r"\textbf{AME}" in table
    assert "Linear uses ordinary least squares (OLS) (coefficients)" in table
    assert "Fraction uses fractional logit (AMEs)" in table


def test_source_model10_full_specification_table_and_note_are_renderable() -> None:
    count = 400
    age = np.arange(count, dtype=float) + 1
    inbred = np.arange(count) % 2 == 0
    expected_publications = np.exp(3 + 0.01 * age - 0.1 * inbred)
    publication_rng = np.random.default_rng(127)
    outcomes = (
        "hazen_perc_med",
        "pp_top_10",
        "pp_top_5",
        "pp_top_1",
        "n_public",
        "P_top10",
        "P_top5",
        "P_top1",
    )
    data = pd.DataFrame(
        {
            "hazen_perc_med": 50 + age * 0.1 - inbred * 2,
            "pp_top_10": 0.1 + age * 0.001 - inbred * 0.02,
            "pp_top_5": 0.05 + age * 0.0005 - inbred * 0.01,
            "pp_top_1": 0.01 + age * 0.0001 - inbred * 0.002,
            "n_public": publication_rng.negative_binomial(2, 2 / (2 + expected_publications)),
            "P_top10": 4 + age * 0.04 - inbred * 0.4,
            "P_top5": 2 + age * 0.02 - inbred * 0.2,
            "P_top1": 0.4 + age * 0.004 - inbred * 0.04,
            "is_inbred": inbred,
            "academic_age": age,
            "title": ["Ao. Prof.", "Full Prof."] * (count // 2),
        }
    )
    result = RegressionSuite(
        data=data,
        config=RegressionSuiteConfig(
            dependent_variables=outcomes,
            model_specs=(RegressionModelSpec("m10", ("C(is_inbred)", "academic_age")),),
            model_types={
                **dict.fromkeys(outcomes[:4], "olshc3"),
                "n_public": "nb",
                **dict.fromkeys(outcomes[5:], "olshc3"),
            },
            relative_performance_outcomes=outcomes[:4],
            absolute_performance_outcomes=outcomes[4:],
            exclude_titles=(),
        ),
    ).fit()
    table = result.latex_tables["performance_without_international_mobility.tex"]
    assert r"\label{tab:without_international_mobility}" in table
    assert "research performance excluding international mobility" in table
    assert r"\textbf{Relative citation impact}" in table
    assert r"\textbf{Absolute publication output}" in table
    assert "All models include the explanatory variables of the full specification" in table
    assert "Inbreeding" in table
    assert "Robust SE" in table
    assert "Estimator" in table
    assert "Effect" in table
    assert table.count(r"\multicolumn{1}{c}{OLS}") == len(outcomes) - 1
    assert table.count(r"\multicolumn{1}{c}{NB}") == 1
    assert table.count(r"\multicolumn{1}{c}{Coef.}") == len(outcomes) - 1
    assert table.count(r"\multicolumn{1}{c}{AME}") == 1
    assert r"\(n=400\)" in table
    assert "AMEs for Publications represent changes in the expected" in table
    assert "AMEs are average derivatives for continuous predictors" in table
    assert "average discrete differences for categorical predictors" in table
    assert "HC3 heteroskedasticity-robust standard errors (Robust SE)" in table
    assert r"Table~\ref{tab:analytical_variables}" in table


def test_source_model_comparison_tables_keep_all_specification_assertions(
    tmp_path: Path,
) -> None:
    count = 160
    data = pd.DataFrame(
        {
            "hazen_perc_med": np.linspace(35, 75, count),
            "pp_top_10": np.linspace(0.02, 0.22, count),
            "pp_top_5": np.linspace(0.01, 0.14, count),
            "pp_top_1": np.linspace(0, 0.06, count) + (np.arange(count) % 5) * 0.001,
            "n_public": 10 + np.arange(count) % 17,
            "P_top10": 0.4 + (np.arange(count) % 9) * 0.3,
            "P_top5": 0.2 + (np.arange(count) % 7) * 0.15,
            "P_top1": 0.05 + (np.arange(count) % 5) * 0.04,
            "is_inbred": np.arange(count) % 3 == 0,
            "academic_age": 1 + (np.arange(count) * 7) % 45,
            "title": [
                ["Ao. Prof.", "Ass. Prof.", "FH-Prof.", "Full Prof."][i % 4] for i in range(count)
            ],
            "coaffiliation_publication_share": np.linspace(0, 0.8, count),
            "mean_coauthors_per_publication": 1 + (np.arange(count) % 11) / 2,
            "n_away_years": np.arange(count) % 12,
            "n_away_years_domestic": np.arange(count) % 6,
            "n_away_years_abroad": np.arange(count) * 2 % 9,
            "first_year": 1985 + (np.arange(count) * 3) % 35,
        }
    )
    active = ("med_share", "eng_share", "comp_sci_share", "chem_share", "physics_share")
    for column in FIELD_SHARE_COLUMNS:
        data[column] = 0.0
    for index, column in enumerate(active):
        data.loc[np.arange(count) % len(active) == index, column] = 1.0
    result = RegressionSuite(
        data=data,
        config=RegressionSuiteConfig(
            dependent_variables=(
                "hazen_perc_med",
                "pp_top_10",
                "pp_top_5",
                "pp_top_1",
                "n_public",
                "P_top10",
                "P_top5",
                "P_top1",
            ),
            model_specs=tuple(
                RegressionModelSpec(model_id, terms)
                for model_id, terms in PAPER_REGRESSION_MODEL_TERMS.items()
            ),
            exclude_titles=(),
        ),
    ).fit()
    table = model_comparison_latex_table(result.models, dependent_variable="pp_top_1")
    assert (
        r"\caption{Association between academic inbreeding and PP(top 1\%) "
        r"across regression model specifications (\(n=160\))}" in table
    )
    assert r"\providecommand{\coef}[3]{#1 & .#2\textsuperscript{#3}}" in table
    assert r"\providecommand{\fit}[1]{\multicolumn{2}{c}{\scriptsize #1}}" in table
    assert r"*{10}{r@{}l}" in table
    assert r"\textbf{M1}" in table and r"\textbf{M9}" in table and r"\textbf{M10}" in table
    assert r"\se{" in table
    assert "Academic age$^2$" in table
    assert "Total mobility" in table
    assert "Co-affiliation publication share" in table
    assert r"\providecommand{\na}{\multicolumn{2}{c}{--}}" in table
    assert table.count(r"\txt{Yes}") == 3
    first = result.models[("pp_top_1", "m1")]
    assert f"AIC & \\fit{{{float(first.fit_statistics['aic']):,.0f}}}" in table
    assert f"BIC & \\fit{{{float(first.fit_statistics['bic']):,.0f}}}" in table
    assert "\nObservations" not in table
    assert table.index("AIC") > table.index(r"Adjusted $R^2$")
    assert table.index("BIC") > table.index("AIC")
    assert "effects are expressed in percentage points" in table
    assert "M1–M10 denote different regression specifications" in table
    assert "AIC denotes the Akaike information criterion" in table
    assert "BIC denotes the Bayesian information criterion" in table
    assert "Lower AIC and BIC indicate a preferable balance" in table
    assert r"Significance levels: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$." in table
    assert '\\"' not in table
    assert "model_comparison_pp_top_1.tex" in result.latex_tables
    count_table = result.latex_tables["model_comparison_n_public.tex"]
    assert r"\textbf{M10}" in count_table
    assert "Inbreeding" in count_table
    assert "Domestic mobility" in count_table
    assert "First publication year" in count_table
    assert r"Adjusted $R^2$" in count_table
    robustness = result.latex_tables["model9_absolute_top_percentile_robustness.tex"]
    assert "fractional counts of highly cited publications (Model 9)" in robustness
    assert r"P(top 10\%)" in robustness
    assert r"P(top 5\%)" in robustness
    assert r"P(top 1\%)" in robustness
    assert "fractionally counted numbers of highly cited publications" in robustness
    for outcome, label in (
        ("P_top10", r"P(top 10\%)"),
        ("P_top5", r"P(top 5\%)"),
        ("P_top1", r"P(top 1\%)"),
    ):
        comparison = result.latex_tables[f"model_comparison_{outcome}.tex"]
        assert label in comparison
        assert r"\textbf{M1}" in comparison and r"\textbf{M9}" in comparison
        assert "fractional publication-count units" in comparison
        assert "reported in percentage points" not in comparison
    from acib.regression.config import RegressionOutputs

    outputs = RegressionOutputs(directory=tmp_path / "regression")
    result.write(outputs, data=data)
    assert (outputs.latex_dir / "model9_absolute_top_percentile_robustness.tex").is_file()
    assert (outputs.latex_dir / "performance_without_international_mobility.tex").is_file()
    assert (outputs.latex_dir / "model_comparison_pp_top_1.tex").is_file()


def test_n_public_negative_binomial_comparison_uses_full_rows_without_adjusted_r_squared() -> None:
    rng = np.random.default_rng(2026)
    count = 500
    age = rng.normal(25, 8, size=count)
    mean = np.exp(1 + 0.015 * age)
    publications = rng.negative_binomial(2, 2 / (2 + mean))
    specs = tuple(RegressionModelSpec(f"m{index}", ("academic_age",)) for index in range(1, 11))
    result = RegressionSuite(
        data=pd.DataFrame({"n_public": publications, "academic_age": age}),
        config=RegressionSuiteConfig(
            dependent_variables=("n_public",),
            model_specs=specs,
            model_types={"n_public": "nb"},
            exclude_titles=(),
        ),
    ).fit()
    table = result.latex_tables["model_comparison_n_public.tex"]
    assert all(model.fit_statistics["n_obs"] == count for model in result.models.values())
    assert r"\textbf{M10}" in table
    assert table.count(r"\multicolumn{2}{c}{\textit{AME}}") == len(specs)
    assert "Inbreeding" in table
    assert "Academic age" in table
    assert "AIC" in table and "BIC" in table
    assert r"Adjusted $R^2$" not in table
    assert "Publications AMEs are expressed in publication-count units" in table
    assert "average marginal effects (AMEs)" in table
    assert "HC3 heteroskedasticity-robust standard errors (Robust SE)" in table
    assert "p-values are based on $z$-tests" in table
