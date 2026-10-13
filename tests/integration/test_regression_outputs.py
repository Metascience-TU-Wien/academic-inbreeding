# Copyright © 2026 TU Wien

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from pytest import MonkeyPatch

from acib.regression import cli as regression_cli
from acib.regression.analysis_config import RegressionSuiteConfig
from acib.regression.config import (
    RegressionConfig,
    RegressionModelConfig,
    RegressionOutputs,
)
from acib.regression.specs import RegressionModelSpec
from acib.regression.suite import RegressionSuite, RegressionSuiteResult
from acib.regression.workflow import run_regression_suite

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "regression_analysis.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("target_analyze_regression", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
regression_script = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(regression_script)


def regression_frame() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    count = 240
    age = np.arange(count, dtype=float) % 35 + 1
    inbred = np.arange(count) % 2 == 0
    title_values = np.array(["Full Prof.", "Ao. Prof.", "Ass. Prof.", "FH-Prof.", "FH-Ass. Prof."])
    med_share = (np.arange(count) % 7) / 7
    return pd.DataFrame(
        {
            "hazen_perc_med": 55 + 0.04 * age - inbred * 1.5 + rng.normal(0, 0.3, count),
            "pp_top_10": 0.13 + 0.001 * age - inbred * 0.02 + rng.normal(0, 0.003, count),
            "pp_top_5": 0.07 + 0.0005 * age - inbred * 0.01 + rng.normal(0, 0.002, count),
            "pp_top_1": 0.015 + 0.0001 * age - inbred * 0.002 + rng.normal(0, 0.001, count),
            "n_public": rng.negative_binomial(4, 4 / (4 + np.exp(1.5 + 0.025 * age))),
            "is_inbred": inbred,
            "academic_age": age,
            "title": title_values[np.arange(count) % len(title_values)],
            "coaffiliation_publication_share": (np.arange(count) % 10) / 10,
            "mean_coauthors_per_publication": 1 + (np.arange(count) % 9) / 3,
            "n_away_years_domestic": np.arange(count) % 6,
            "n_away_years_abroad": np.arange(count) % 8,
            "first_year": 1980 + np.arange(count) % 40,
            "med_share": med_share,
        }
    )


def outputs_under(path: Path) -> RegressionOutputs:
    return RegressionOutputs(directory=path)


def pipeline_config(
    root: Path,
    *,
    rebuild_input: bool,
    outcome: str = "pp_top_10",
) -> RegressionConfig:
    return RegressionConfig(
        article_author_file_path=root / "article_author.csv",
        mobility_author_proxy=root / "mobility_authors.csv",
        outputs=outputs_under(root / "outputs"),
        dependent_variables={outcome: "ols"},
        model_specs=(RegressionModelConfig("custom", ("academic_age",)),),
        exclude_titles=(),
        rebuild_input=rebuild_input,
    )


def write_build_inputs(root: Path) -> None:
    authors = pd.DataFrame(
        {
            "sc_id": list(range(1, 7)),
            "author_ids": list(range(1, 7)),
            "is_inbreeding_proxy": [bool(i % 2) for i in range(1, 7)],
            "person_title": ["Univ.-Prof. Dr."] * 6,
            "n_unique": [4] * 6,
            "academic_age_years": [8, 10, 12, 14, 16, 18],
            "first_date": [f"{year}-01-01" for year in range(2000, 2006)],
        }
    )
    article_rows = []
    impact_rows = []
    for author_index in range(1, 7):
        for publication_index in range(4):
            eid = f"e{author_index}-{publication_index}"
            article_rows.append(
                {
                    "eid": eid,
                    "author_ids": str(author_index),
                    "author_afids": "",
                    "cover_date": f"{2010 + publication_index}-01-01",
                }
            )
            impact_rows.append(
                {
                    "eid": eid,
                    "hazen_perc_med": 45 + author_index + publication_index,
                    "pp_top_10": (author_index + publication_index) / 10,
                    "pp_top_5": (author_index + publication_index) / 20,
                    "pp_top_1": (author_index + publication_index) / 100,
                }
            )
    authors.to_csv(root / "mobility_authors.csv", index=False)
    pd.DataFrame(article_rows).to_csv(root / "article_author.csv", index=False)
    pd.DataFrame(impact_rows).to_csv(root / "impact.csv", index=False)


def test_configured_workflow_builds_inputs_and_writes_outputs(tmp_path: Path) -> None:
    write_build_inputs(tmp_path)
    config = pipeline_config(tmp_path, rebuild_input=True)
    config = replace(config, article_impact_file_path=tmp_path / "impact.csv")

    result = run_regression_suite(config)

    outputs = config.outputs
    assert outputs.regression_input.is_file()
    assert outputs.professor_article_input.is_file()
    assert (outputs.latex_dir / "descriptive_statistics.tex").is_file()
    assert result.fit_statistics.loc[0, "n_obs"] == 6
    assert pd.read_csv(outputs.regression_input)["title"].tolist() == ["Full Prof."] * 6


def test_configured_workflow_reuses_existing_regression_input(tmp_path: Path) -> None:
    config = pipeline_config(tmp_path, rebuild_input=False)
    config.outputs.regression_input.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "sc_id": range(1, 7),
            "is_inbred": [True, False] * 3,
            "title": ["Full Prof."] * 6,
            "pp_top_10": np.linspace(0.1, 0.6, 6),
            "academic_age": [10, 11, 12, 13, 14, 15],
        }
    ).to_csv(config.outputs.regression_input, index=False)

    result = run_regression_suite(config)

    assert result.fit_statistics.loc[0, "n_obs"] == 6
    assert (config.outputs.latex_dir / "descriptive_statistics.tex").is_file()


def test_configured_performance_latex_outputs_are_written_to_configured_directory(
    tmp_path: Path,
) -> None:
    count = 60
    data = pd.DataFrame(
        {
            "hazen_perc_med": np.linspace(35, 80, count),
            "pp_top_10": np.linspace(0.05, 0.35, count),
            "n_public": np.arange(count) % 12 + 2,
            "P_top10": np.arange(count) % 8 + 0.5,
            "is_inbred": np.arange(count) % 2 == 0,
            "academic_age": np.arange(count, dtype=float) + 1,
            "title": ["Ao. Prof.", "Ass. Prof.", "FH-Prof.", "Full Prof."] * 15,
        }
    )
    outcome_models = {
        "hazen_perc_med": "ols",
        "pp_top_10": "ols",
        "n_public": "ols",
        "P_top10": "ols",
    }
    result = RegressionSuite(
        data=data,
        config=RegressionSuiteConfig(
            dependent_variables=tuple(outcome_models),
            model_specs=(RegressionModelSpec("m1", ("C(is_inbred)", "academic_age", "C(title)")),),
            model_types=outcome_models,
            relative_performance_outcomes=("hazen_perc_med", "pp_top_10"),
            absolute_performance_outcomes=("n_public", "P_top10"),
            exclude_titles=(),
        ),
    ).fit()
    tables = result.latex_tables
    absolute = tables["absolute_performance_m1.tex"]
    assert "research_performance_m1.tex" in tables
    assert "absolute_performance_m1.tex" in tables
    assert "Publications" in absolute and r"P(top 10\%)" in absolute
    assert r"\textbf{$P$}" in absolute
    assert r"P{>}|t|" not in absolute
    assert "OLS $p$-values are based on $t$-tests" in absolute

    outputs = outputs_under(tmp_path)
    result.write(outputs, data=data)
    assert (outputs.latex_dir / "research_performance_m1.tex").is_file()
    assert (outputs.latex_dir / "absolute_performance_m1.tex").is_file()


def test_model10_full_specification_latex_is_written_to_configured_directory(
    tmp_path: Path,
) -> None:
    count = 80
    age = np.arange(count, dtype=float) + 1
    inbred = np.arange(count) % 2 == 0
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
            "n_public": 20 + age * 0.2 - inbred * 2,
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
            model_types=dict.fromkeys(outcomes, "ols"),
            relative_performance_outcomes=outcomes[:4],
            absolute_performance_outcomes=outcomes[4:],
            exclude_titles=(),
        ),
    ).fit()
    table = result.latex_tables["model10_without_international_mobility.tex"]
    assert r"\label{tab:without_international_mobility}" in table
    assert r"\textbf{Relative citation impact}" in table
    assert r"\textbf{Absolute publication output}" in table
    assert "full specification excluding international mobility" in table

    outputs = outputs_under(tmp_path)
    result.write(outputs, data=data)
    assert (outputs.latex_dir / "model10_without_international_mobility.tex").is_file()


def test_model9_writer_ignores_unrelated_nb_fit_and_writes_plot(tmp_path: Path) -> None:
    outcomes = ("hazen_perc_med", "pp_top_10", "pp_top_5", "pp_top_1", "n_public")
    model_types = {outcome: "ols" for outcome in outcomes[:-1]}
    model_types["n_public"] = "nb"
    suite = RegressionSuite(
        data=regression_frame(),
        config=RegressionSuiteConfig(
            dependent_variables=outcomes,
            model_specs=(RegressionModelSpec("m9", ("C(is_inbred)", "academic_age", "C(title)")),),
            model_types=model_types,
            exclude_titles=(),
        ),
    )
    result = suite.fit()

    assert all(result.models[(outcome, "m9")].model_type == "ols" for outcome in outcomes[:-1])
    assert result.models[("n_public", "m9")].model_type == "nb"
    professor_articles = suite.data.assign(
        eid=[f"paper-{index}" for index in range(len(suite.data))],
        sc_id=np.arange(len(suite.data)) % 10,
    )
    result.write(outputs_under(tmp_path), data=suite.data, professor_articles=professor_articles)

    assert (tmp_path / "plots" / "model9_inbreeding_predictions.png").is_file()
    assert (tmp_path / "plots" / "model9_inbreeding_predictions_by_title.png").is_file()
    assert not (tmp_path / "plots" / "model9_inbreeding_predictions_full_prof.png").exists()
    assert not (tmp_path / "plots" / "model9_inbreeding_predictions.csv").exists()
    assert (tmp_path / "plots" / "professor_title_inbreeding_violin.png").is_file()


def test_script_entrypoint_forwards_summary_setting(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    config = pipeline_config(tmp_path, rebuild_input=False)
    calls: dict[str, object] = {}

    def fake_run(config_arg: RegressionConfig, *, print_model_summaries: bool = False) -> None:
        calls["config"] = config_arg
        calls["print_model_summaries"] = print_model_summaries

    monkeypatch.setattr(regression_script, "run_regression_suite", fake_run)
    regression_script.run_regression(config, print_model_summaries=True)
    assert calls == {"config": config, "print_model_summaries": True}


def test_cli_passes_summary_flag_and_config_path(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_run_from_config(**kwargs: object) -> None:
        calls.update(kwargs)

    monkeypatch.setattr(regression_cli, "run_regression_suite_from_config", fake_run_from_config)
    config_path = tmp_path / "pipeline.toml"
    regression_cli.main(["--config", str(config_path), "--print-model-summaries"])
    assert calls == {
        "config_path": config_path,
        "validate_paths": False,
        "print_model_summaries": True,
    }


def test_workflow_prints_summaries_before_writing_outputs(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    config = pipeline_config(tmp_path, rebuild_input=False)
    config.outputs.regression_input.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "sc_id": range(1, 7),
            "is_inbred": [True, False] * 3,
            "title": ["Full Prof."] * 6,
            "pp_top_10": np.linspace(0.1, 0.6, 6),
            "academic_age": np.arange(10, 16),
        }
    ).to_csv(config.outputs.regression_input, index=False)
    config = replace(config, print_model_summaries=True)
    events: list[str] = []

    def record_print(self: RegressionSuiteResult) -> None:
        events.append("print")

    def record_write(
        self: RegressionSuiteResult,
        outputs: RegressionOutputs,
        *,
        data: pd.DataFrame | None = None,
        professor_articles: pd.DataFrame | None = None,
    ) -> None:
        events.append("write")

    monkeypatch.setattr(RegressionSuiteResult, "print_model_summaries", record_print)
    monkeypatch.setattr(RegressionSuiteResult, "write", record_write)
    run_regression_suite(config)
    assert events == ["print", "write"]
