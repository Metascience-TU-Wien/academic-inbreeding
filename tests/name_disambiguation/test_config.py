# Copyright © 2026 TU Wien

from pathlib import Path

import pytest

from acib.pipeline.config import load_name_disambiguation_config


def write_pipeline_config(path: Path, *, section_options: list[str] | None = None) -> None:
    """Write a dependency-aware pipeline configuration for loader tests.

    :param path: Temporary TOML destination.
    :param section_options: Optional extra disambiguation settings.
    """
    options = section_options or []
    path.write_text(
        "\n".join(
            [
                "[data]",
                'researcher_file_path = "data/researchers.csv"',
                'candidate_authors_file = "data/authors.csv"',
                'article_author_file_path = "tmp/article_author.csv"',
                'article_reference_file_path = "tmp/article_reference.parquet"',
                "",
                "[name_matching]",
                "[name_matching.outputs]",
                'directory = "tmp/name-matching"',
                "",
                "[name_disambiguation]",
                *options,
                "",
                "[name_disambiguation.outputs]",
                'directory = "tmp/disambiguation"',
            ]
        ),
        encoding="utf-8",
    )


def test_config_resolves_shared_inputs_and_six_output_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    write_pipeline_config(
        config_path,
        section_options=[
            "enable_reporting = false",
            "same_author_score_threshold = 5.0",
            "possible_author_score_threshold = 2.5",
        ],
    )
    config = load_name_disambiguation_config(config_path)
    assert config.matching_output.name == "resolved_researchers.csv"
    assert config.article_author_file_path.name == "article_author.csv"
    assert config.article_reference_file_path.name == "article_reference.parquet"
    assert config.enable_reporting is False
    assert config.same_author_score_threshold == 5.0
    assert config.possible_author_score_threshold == 2.5
    assert [path.name for path in config.outputs.all_paths()] == [
        "pair_decisions.csv",
        "clusters.csv",
        "final_author_selection.csv",
        "final_author_selection_scores.csv",
        "report.txt",
        "report.jsonl",
    ]


def test_disambiguation_config_preserves_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    write_pipeline_config(config_path)
    config = load_name_disambiguation_config(config_path)
    assert config.enable_reporting is True
    assert config.same_author_score_threshold == 6.0
    assert config.possible_author_score_threshold == 3.0
    assert config.max_workers == 1


def test_disambiguation_config_rejects_duplicate_upstream_and_external_paths(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pipeline.toml"
    write_pipeline_config(
        config_path,
        section_options=['author_file_path = "tmp/duplicate-matches.csv"'],
    )

    with pytest.raises(ValueError, match="must not duplicate"):
        load_name_disambiguation_config(config_path)


def test_disambiguation_config_rejects_possible_threshold_above_same_threshold(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pipeline.toml"
    write_pipeline_config(
        config_path,
        section_options=[
            "same_author_score_threshold = 2.0",
            "possible_author_score_threshold = 3.0",
        ],
    )
    with pytest.raises(ValueError, match="possible_author_score_threshold"):
        load_name_disambiguation_config(config_path)


def test_output_directory_is_resolved_and_artifacts_are_derived(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    entries = [
        "[data]",
        'researcher_file_path = "/tmp/researchers.csv"',
        'candidate_authors_file = "/tmp/authors.csv"',
        'article_author_file_path = "/tmp/articles.csv"',
        'article_reference_file_path = "/tmp/references.parquet"',
        "",
        "[name_matching]",
        "[name_matching.outputs]",
        'directory = "/tmp/name-matching"',
        "",
        "[name_disambiguation]",
        "[name_disambiguation.outputs]",
        f'directory = "{tmp_path / "outputs"}"',
    ]
    config_path.write_text("\n".join(entries), encoding="utf-8")
    config = load_name_disambiguation_config(config_path)
    assert config.outputs.all_paths() == (
        tmp_path / "outputs/pair_decisions.csv",
        tmp_path / "outputs/clusters.csv",
        tmp_path / "outputs/final_author_selection.csv",
        tmp_path / "outputs/final_author_selection_scores.csv",
        tmp_path / "outputs/report.txt",
        tmp_path / "outputs/report.jsonl",
    )


def test_shared_data_inputs_resolve_matcher_and_article_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        """[data]
researcher_file_path = "data/researchers.csv"
candidate_authors_file = "data/authors.csv"
article_author_file_path = "data/article.csv"
article_reference_file_path = "data/references.parquet"

[name_matching]
[name_matching.outputs]
directory = "tmp/name-matching"

[name_disambiguation.outputs]
directory = "tmp/disambiguation"
""",
        encoding="utf-8",
    )

    config = load_name_disambiguation_config(config_path)

    assert config.matching_output.name == "resolved_researchers.csv"
    assert config.article_author_file_path.name == "article.csv"
    assert config.article_reference_file_path.name == "references.parquet"
    assert config.outputs.pair_decisions.name == "pair_decisions.csv"


def test_disambiguation_config_checks_external_inputs_and_output_overwrite(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        "\n".join(
            [
                "[data]",
                'researcher_file_path = "/tmp/researchers.csv"',
                'candidate_authors_file = "/tmp/authors.csv"',
                f'article_author_file_path = "{tmp_path / "missing-articles.csv"}"',
                f'article_reference_file_path = "{tmp_path / "missing-references.parquet"}"',
                "",
                "[name_matching]",
                "[name_matching.outputs]",
                'directory = "/tmp/name-matching"',
                "",
                "[name_disambiguation]",
                "",
                "[name_disambiguation.outputs]",
                f'directory = "{tmp_path / "outputs"}"',
            ]
        ),
        encoding="utf-8",
    )
    config = load_name_disambiguation_config(config_path)
    with pytest.raises(FileNotFoundError, match="Missing name-disambiguation external inputs"):
        config.check_external_inputs_exist()
    output_path = config.outputs.pair_decisions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch()
    with pytest.raises(FileExistsError, match="outputs already exist"):
        config.check_output_paths()
    config.check_output_paths(overwrite=True)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_disambiguation_config_rejects_invalid_worker_count(
    tmp_path: Path, workers: object
) -> None:
    config_path = tmp_path / "pipeline.toml"
    value = str(workers).lower() if isinstance(workers, bool) else str(workers)
    write_pipeline_config(config_path, section_options=[f"max_workers = {value}"])
    with pytest.raises((ValueError, TypeError), match="max_workers"):
        load_name_disambiguation_config(config_path)
