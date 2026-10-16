# Copyright © 2026 TU Wien

from pathlib import Path

import pytest

from acib.mobility.config import MobilityOutputs
from acib.pipeline.config import (
    PROJECT_ROOT,
    load_mobility_config,
    load_pipeline_config,
)


def write_pipeline_config(path: Path, *, mobility_options: str = "") -> None:
    """Write a complete canonical pipeline configuration for mobility tests.

    :param path: Temporary TOML destination.
    :param mobility_options: Optional extra options under ``[mobility]``.
    """
    path.write_text(
        f"""[data]
researcher_file_path = "external/researchers.csv"
candidate_authors_file = "external/authors.csv"
article_author_file_path = "data/input/article.parquet"
article_reference_file_path = "data/input/article_reference.parquet"
author_affiliations_file_path = "data/input/author_affiliation.gpkg"

[name_matching]
[name_matching.outputs]
directory = "data/slurm/output/name_matching"

[name_disambiguation]
max_workers = 3

[name_disambiguation.outputs]
directory = "data/slurm/output/name_disambiguation"

[mobility]
enable_university_affiliation_inbreeding_detection = true
enable_coaffiliated_professor_merge = true
professor_title_pattern = "prof"
early_affiliation_method = "first_n_years"
first_affiliation_publication_count = 3
first_affiliation_year_count = 2
min_publication_history_years = 0
max_workers = 5
{mobility_options}

[mobility.outputs]
directory = "data/slurm/output/mobility"
""",
        encoding="utf-8",
    )


def test_mobility_resolves_inputs_from_canonical_owners(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    write_pipeline_config(config_path)

    config = load_mobility_config(config_path)

    assert config.authors == (
        PROJECT_ROOT / "data/slurm/output/name_disambiguation/final_author_selection.csv"
    )
    assert config.researcher_metadata == (
        PROJECT_ROOT / "data/slurm/output/name_matching/resolved_researchers.csv"
    )
    assert config.article_author == PROJECT_ROOT / "data/input/article.parquet"
    assert config.affiliations == PROJECT_ROOT / "data/input/author_affiliation.gpkg"
    assert config.outputs.all_paths() == (
        PROJECT_ROOT / "data/slurm/output/mobility/author_affiliations.csv",
        PROJECT_ROOT / "data/slurm/output/mobility/author_affiliations_unique.csv",
        PROJECT_ROOT / "data/slurm/output/mobility/author_inbreeding_proxy.csv",
        PROJECT_ROOT / "data/slurm/output/mobility/author_articles.csv",
    )
    assert config.enable_university_affiliation_inbreeding_detection is True
    assert config.enable_coaffiliated_professor_merge is True
    assert config.professor_title_pattern == "prof"
    assert config.early_affiliation_method == "first_n_years"
    assert config.first_affiliation_publication_count == 3
    assert config.first_affiliation_year_count == 2
    assert config.min_publication_history_years == 0
    assert config.max_workers == 5


def test_mobility_relative_paths_ignore_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "pipeline.toml"
    write_pipeline_config(config_path)
    monkeypatch.chdir(tmp_path)

    config = load_mobility_config(config_path)

    assert config.authors.is_relative_to(PROJECT_ROOT)
    assert config.outputs.author_articles == (
        PROJECT_ROOT / "data/slurm/output/mobility/author_articles.csv"
    )


def test_mobility_missing_inputs_report_producing_stage_and_data_owner(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pipeline.toml"
    write_pipeline_config(config_path)
    text = (
        config_path.read_text(encoding="utf-8")
        .replace(
            'directory = "data/slurm/output/name_disambiguation"',
            f'directory = "{tmp_path / "disambiguation"}"',
        )
        .replace(
            'directory = "data/slurm/output/name_matching"',
            f'directory = "{tmp_path / "matching"}"',
        )
        .replace(
            'article_author_file_path = "data/input/article.parquet"',
            'article_author_file_path = "external-missing/article.parquet"',
        )
        .replace(
            'author_affiliations_file_path = "data/input/author_affiliation.gpkg"',
            'author_affiliations_file_path = "external-missing/author_affiliation.gpkg"',
        )
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(FileNotFoundError) as error:
        load_mobility_config(config_path).check_input_paths()

    message = str(error.value)
    assert "[name_disambiguation.outputs].final_author_selection" in message
    assert "producer: name_disambiguation" in message
    assert "[name_matching.outputs].resolved_researchers" in message
    assert "producer: name_matching" in message
    assert "[data].article_author_file_path" in message
    assert "[data].author_affiliations_file_path" in message


def test_mobility_rejects_redundant_stage_input_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    write_pipeline_config(config_path, mobility_options='author_file_path = "duplicate.csv"')

    with pytest.raises(ValueError, match=r"\[mobility\]\.author_file_path"):
        load_mobility_config(config_path)


def test_mobility_pipeline_config_records_external_geo_input(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    write_pipeline_config(config_path)

    config = load_pipeline_config(config_path)

    assert config.data.author_affiliations_file_path == (
        PROJECT_ROOT / "data/input/author_affiliation.gpkg"
    )
    assert config.mobility is not None
    assert config.name_disambiguation is not None
    assert config.name_disambiguation.outputs.final_author_selection == config.mobility.authors


def test_mobility_outputs_must_be_distinct(tmp_path: Path) -> None:
    outputs = MobilityOutputs(directory=tmp_path / "mobility")
    assert len(set(outputs.all_paths())) == 4
