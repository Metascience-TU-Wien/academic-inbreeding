# Copyright © 2026 TU Wien

"""Canonical configuration and independent matching workflow tests."""

from pathlib import Path

import pytest

from acib.pipeline.config import PROJECT_ROOT, load_statistical_matching_config
from acib.statistical_matching.workflow import (
    run_statistical_matching_suite_from_config,
)


def _write_config(path: Path) -> None:
    path.write_text(
        """[data]
researcher_file_path = "unused/researchers.csv"
candidate_authors_file = "unused/authors.csv"

[name_matching.outputs]
directory = "unused/matches"

[regression.outputs]
directory = "data/generated"

[regression.outcome_metadata.hazen_perc_med]
outcome_type = "relative"
alias = "mwPR"

[regression.outcome_metadata.pp_top_10]
outcome_type = "relative"
alias = "PP(top 10%)"

[statistical_matching]
outcomes = ["hazen_perc_med", "pp_top_10"]

[statistical_matching.outputs]
directory = "tmp/matching"
""",
        encoding="utf-8",
    )


def test_matching_config_uses_producer_input_and_root_resolved_paths(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pipeline.toml"
    _write_config(config_path)
    config = load_statistical_matching_config(config_path)
    assert (
        config.input_path == (PROJECT_ROOT / "data/generated/author_regression_input.csv").resolve()
    )
    assert (
        config.outputs.balance_summary
        == (PROJECT_ROOT / "tmp/matching/balance_summary.csv").resolve()
    )
    assert config.outcomes == ("hazen_perc_med", "pp_top_10")
    assert config.outcome_types["hazen_perc_med"] == "relative"
    assert not hasattr(config, "dependent_variables")
    assert not hasattr(config, "output_dir")


def test_absolute_regression_output_path_remains_absolute(tmp_path: Path) -> None:
    external = (tmp_path / "absolute" / "author_regression_input.csv").resolve()
    config_path = tmp_path / "pipeline.toml"
    _write_config(config_path)
    text = config_path.read_text().replace(
        'directory = "data/generated"', f'directory = "{external.parent}"', 1
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
    input_path = tmp_path / "author_regression_input.csv"
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
        'directory = "data/generated"', f'directory = "{input_path.parent}"', 1
    )
    config_path.write_text(text)
    result = run_statistical_matching_suite_from_config(
        config_path=config_path, write_outputs=False
    )
    assert set(result.balance_summary.method) == {"PSM", "IPTW", "CEM", "EB"}
