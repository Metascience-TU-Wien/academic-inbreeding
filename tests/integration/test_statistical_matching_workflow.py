# Copyright © 2026 TU Wien

"""Canonical configuration and independent matching workflow tests."""

from pathlib import Path

import pytest

from acib.pipeline.config import PROJECT_ROOT, load_statistical_matching_config
from acib.statistical_matching.workflow import (
    run_statistical_matching_suite_from_config,
)


def _write_config(path: Path, *, absolute_input: str | None = None) -> None:
    outputs = [
        "balance_summary",
        "covariate_balance",
        "outcome_effects_long",
        "adjusted_outcome_means",
        "outcome_effects_wide",
        "outcome_standard_errors_wide",
        "method_diagnostics",
        "propensity_scores",
        "propensity_scores_by_method",
        "matching_formulas",
        "latex_dir",
        "plots_dir",
    ]
    text = "\n".join(
        [
            "[data]",
            'researcher_file_path = "unused/researchers.csv"',
            'candidate_authors_file = "unused/authors.csv"',
            "[name_matching]",
            'resolved_matches_output_file_path = "unused/matches.csv"',
            "[regression.outputs]",
            'regression_input = "data/generated/regression_input.csv"',
            "[regression.outcome_metadata.hazen_perc_med]",
            'outcome_type = "relative"',
            'alias = "mwPR"',
            "[regression.outcome_metadata.pp_top_10]",
            'outcome_type = "relative"',
            'alias = "PP(top 10%)"',
            "[statistical_matching]",
            'outcomes = ["hazen_perc_med", "pp_top_10"]',
            "[statistical_matching.outputs]",
            *[
                f'{name} = "tmp/matching/{name}' + ("" if name.endswith("_dir") else ".csv") + '"'
                for name in outputs
            ],
        ]
    )
    if absolute_input:
        text = text.replace("data/generated/regression_input.csv", absolute_input, 1)
    path.write_text(text, encoding="utf-8")


def test_matching_config_uses_producer_input_and_root_resolved_paths(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pipeline.toml"
    _write_config(config_path)
    config = load_statistical_matching_config(config_path)
    assert config.input_path == (PROJECT_ROOT / "data/generated/regression_input.csv").resolve()
    assert (
        config.outputs.balance_summary
        == (PROJECT_ROOT / "tmp/matching/balance_summary.csv").resolve()
    )
    assert config.outcomes == ("hazen_perc_med", "pp_top_10")
    assert config.outcome_types["hazen_perc_med"] == "relative"
    assert not hasattr(config, "dependent_variables")
    assert not hasattr(config, "output_dir")


def test_absolute_regression_output_path_remains_absolute(tmp_path: Path) -> None:
    external = (tmp_path / "external.csv").resolve()
    config_path = tmp_path / "pipeline.toml"
    _write_config(config_path)
    text = config_path.read_text().replace(
        '"data/generated/regression_input.csv"', f'"{external}"', 1
    )
    config_path.write_text(text)
    assert load_statistical_matching_config(config_path).input_path == external


@pytest.mark.parametrize(
    ("duplicate", "error_message"),
    [
        ('input_path = "duplicate.csv"', "must not duplicate"),
        ('output_dir = "duplicate"', "must not duplicate"),
        ('dependent_variables = ["P_top10"]', "Unrecognized"),
        ("balance_threshold = 0", "must be positive"),
    ],
)
def test_matching_config_rejects_duplicate_local_paths_and_outcome_metadata(
    tmp_path: Path, duplicate: str, error_message: str
) -> None:
    config_path = tmp_path / "pipeline.toml"
    _write_config(config_path)
    text = config_path.read_text().replace(
        "[statistical_matching]\n", f"[statistical_matching]\n{duplicate}\n"
    )
    config_path.write_text(text)
    with pytest.raises(ValueError, match=error_message):
        load_statistical_matching_config(config_path)


def test_workflow_runs_from_toml_without_regression_upstream_execution(
    tmp_path: Path,
) -> None:
    import pandas as pd

    config_path = tmp_path / "pipeline.toml"
    input_path = tmp_path / "regression_input.csv"
    pd.DataFrame(
        {
            "sc_id": range(80),
            "is_inbred": [False, True] * 40,
            "title": ["Full Prof."] * 80,
            "academic_age": list(range(1, 81)),
            "primary_field": ["Engineering"] * 80,
            "first_year": [2000] * 80,
            "hazen_perc_med": [float(i % 10) for i in range(80)],
            "pp_top_10": [i / 100 for i in range(80)],
        }
    ).to_csv(input_path, index=False)
    _write_config(config_path)
    text = config_path.read_text().replace(
        '"data/generated/regression_input.csv"', f'"{input_path}"', 1
    )
    config_path.write_text(text)
    result = run_statistical_matching_suite_from_config(
        config_path=config_path, write_outputs=False
    )
    assert set(result.balance_summary.method) == {"PSM", "IPTW", "CEM", "EB"}
