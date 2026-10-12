# Copyright © 2026 TU Wien

"""Dependency-aware, single-directory artifact resolution tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from acib.pipeline.config import PROJECT_ROOT, load_pipeline_config


def _pipeline_toml(
    directories: dict[str, str],
) -> str:
    """Build a minimal all-stage TOML document with the selected output roots."""
    return f"""[data]
researcher_file_path = "/external/researchers.csv"
candidate_authors_file = "/external/authors.csv"
article_author_file_path = "/external/articles.parquet"
article_reference_file_path = "/external/references.parquet"
author_affiliations_file_path = "/external/affiliations.gpkg"
article_impact_file_path = "/external/impact.csv"

[name_matching]
[name_matching.outputs]
directory = "{directories["name_matching"]}"

[name_disambiguation]
[name_disambiguation.outputs]
directory = "{directories["name_disambiguation"]}"

[mobility]
early_affiliation_method = "first_n_years"
[mobility.outputs]
directory = "{directories["mobility"]}"

[regression]
[regression.outputs]
directory = "{directories["regression"]}"

[statistical_matching]
[statistical_matching.outputs]
directory = "{directories["statistical_matching"]}"
"""


@pytest.mark.parametrize("absolute", [False, True], ids=("project-relative", "absolute"))
def test_one_output_directory_resolves_artifacts_and_dependencies_for_every_stage(
    tmp_path: Path,
    absolute: bool,
) -> None:
    stage_names = (
        "name_matching",
        "name_disambiguation",
        "mobility",
        "regression",
        "statistical_matching",
    )
    directory_values = {
        stage: (
            str((tmp_path / stage).resolve()) if absolute else f"tmp/config-output-tests/{stage}"
        )
        for stage in stage_names
    }
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(_pipeline_toml(directory_values), encoding="utf-8")

    config = load_pipeline_config(config_path)
    expected = {
        stage: (
            Path(directory_values[stage]).resolve()
            if absolute
            else (PROJECT_ROOT / directory_values[stage]).resolve()
        )
        for stage in stage_names
    }

    assert config.name_matching.outputs.directory == expected["name_matching"]
    assert config.name_matching.outputs.resolved_researchers == (
        expected["name_matching"] / "resolved_researchers.csv"
    )
    assert config.name_disambiguation is not None
    assert config.name_disambiguation.outputs.directory == expected["name_disambiguation"]
    assert config.name_disambiguation.outputs.final_author_selection == (
        expected["name_disambiguation"] / "final_author_selection.csv"
    )
    assert config.name_disambiguation.matching_output == (
        config.name_matching.outputs.resolved_researchers
    )

    assert config.mobility is not None
    assert config.mobility.outputs.directory == expected["mobility"]
    assert config.mobility.outputs.author_inbreeding_proxy == (
        expected["mobility"] / "author_inbreeding_proxy.csv"
    )
    assert config.mobility.researcher_metadata == config.name_matching.outputs.resolved_researchers
    assert config.mobility.authors == config.name_disambiguation.outputs.final_author_selection

    assert config.regression is not None
    assert config.regression.outputs.directory == expected["regression"]
    assert config.regression.outputs.regression_input == (
        expected["regression"] / "author_regression_input.csv"
    )
    assert config.regression.outputs.latex_dir == expected["regression"] / "latex"
    assert config.regression.outputs.plots_dir == expected["regression"] / "plots"
    assert config.regression.mobility_author_proxy == (
        config.mobility.outputs.author_inbreeding_proxy
    )

    assert config.statistical_matching is not None
    assert config.statistical_matching.outputs.directory == expected["statistical_matching"]
    assert config.statistical_matching.outputs.balance_summary == (
        expected["statistical_matching"] / "balance_summary.csv"
    )
    assert config.statistical_matching.outputs.latex_dir == (
        expected["statistical_matching"] / "latex"
    )
    assert config.statistical_matching.outputs.plots_dir == (
        expected["statistical_matching"] / "plots"
    )
    assert config.statistical_matching.input_path == config.regression.outputs.regression_input


def test_directory_changes_recompute_downstream_artifact_paths(tmp_path: Path) -> None:
    mobility_directory = (tmp_path / "custom-mobility").resolve()
    regression_directory = (tmp_path / "custom-regression").resolve()
    directories = {
        "name_matching": "tmp/default-matching",
        "name_disambiguation": "tmp/default-disambiguation",
        "mobility": str(mobility_directory),
        "regression": str(regression_directory),
        "statistical_matching": "tmp/default-statistical-matching",
    }
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(_pipeline_toml(directories), encoding="utf-8")

    config = load_pipeline_config(config_path)

    assert config.mobility is not None and config.regression is not None
    assert config.statistical_matching is not None
    assert config.mobility.outputs.author_inbreeding_proxy == (
        mobility_directory / "author_inbreeding_proxy.csv"
    )
    assert config.regression.mobility_author_proxy == (
        mobility_directory / "author_inbreeding_proxy.csv"
    )
    assert config.regression.outputs.regression_input == (
        regression_directory / "author_regression_input.csv"
    )
    assert config.statistical_matching.input_path == (
        regression_directory / "author_regression_input.csv"
    )
