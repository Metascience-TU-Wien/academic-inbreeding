# Copyright © 2026 TU Wien

from pathlib import Path

import pytest

from acib.pipeline.config import load_name_disambiguation_config


def write_legacy_config(path: Path, *, section_options: list[str] | None = None) -> None:
    """Write a legacy-style pipeline configuration for loader tests.

    :param path: Temporary TOML destination.
    :param section_options: Optional extra disambiguation settings.
    """
    options = section_options or []
    path.write_text(
        "\n".join(
            [
                "[name_matching]",
                'researcher_file_path = "data/researchers.csv"',
                'candidate_authors_file = "data/authors.csv"',
                'resolved_matches_output_file_path = "tmp/resolved.csv"',
                "",
                "[name_disambiguation]",
                'author_file_path = "tmp/resolved.csv"',
                'article_author_file_path = "tmp/article_author.csv"',
                'article_reference_file_path = "tmp/article_reference.parquet"',
                'disambiguated_author_path = "tmp/disambiguation"',
                *options,
            ]
        ),
        encoding="utf-8",
    )


def test_legacy_config_resolves_inputs_and_six_output_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    write_legacy_config(
        config_path,
        section_options=[
            "enable_reporting = false",
            "same_author_score_threshold = 5.0",
            "possible_author_score_threshold = 2.5",
        ],
    )
    config = load_name_disambiguation_config(config_path)
    assert config.matching_output.name == "resolved.csv"
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
    write_legacy_config(config_path)
    config = load_name_disambiguation_config(config_path)
    assert config.enable_reporting is True
    assert config.same_author_score_threshold == 6.0
    assert config.possible_author_score_threshold == 3.0
    assert config.max_workers == 1


def test_disambiguation_config_rejects_possible_threshold_above_same_threshold(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pipeline.toml"
    write_legacy_config(
        config_path,
        section_options=[
            "same_author_score_threshold = 2.0",
            "possible_author_score_threshold = 3.0",
        ],
    )
    with pytest.raises(ValueError, match="possible_author_score_threshold"):
        load_name_disambiguation_config(config_path)


def test_explicit_output_paths_are_resolved_individually(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    entries = [
        "[name_matching]",
        'researcher_file_path = "/tmp/researchers.csv"',
        'candidate_authors_file = "/tmp/authors.csv"',
        'resolved_matches_output_file_path = "/tmp/resolved.csv"',
        "",
        "[name_disambiguation]",
        'article_author_file_path = "/tmp/articles.csv"',
        'article_reference_file_path = "/tmp/references.parquet"',
        "[name_disambiguation.outputs]",
        f'pair_decisions = "{tmp_path / "pairs.csv"}"',
        f'clusters = "{tmp_path / "clusters.csv"}"',
        f'final_author_selection = "{tmp_path / "selection.csv"}"',
        f'final_author_selection_scores = "{tmp_path / "scores.csv"}"',
        f'report_txt = "{tmp_path / "report.txt"}"',
        f'report_jsonl = "{tmp_path / "report.jsonl"}"',
    ]
    config_path.write_text("\n".join(entries), encoding="utf-8")
    config = load_name_disambiguation_config(config_path)
    assert config.outputs.all_paths() == (
        tmp_path / "pairs.csv",
        tmp_path / "clusters.csv",
        tmp_path / "selection.csv",
        tmp_path / "scores.csv",
        tmp_path / "report.txt",
        tmp_path / "report.jsonl",
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
resolved_matches_output_file_path = "tmp/resolved.csv"

[name_disambiguation.outputs]
pair_decisions = "tmp/pairs.csv"
clusters = "tmp/clusters.csv"
final_author_selection = "tmp/selection.csv"
final_author_selection_scores = "tmp/scores.csv"
report_txt = "tmp/report.txt"
report_jsonl = "tmp/report.jsonl"
""",
        encoding="utf-8",
    )

    config = load_name_disambiguation_config(config_path)

    assert config.matching_output.name == "resolved.csv"
    assert config.article_author_file_path.name == "article.csv"
    assert config.article_reference_file_path.name == "references.parquet"
    assert config.outputs.pair_decisions.name == "pairs.csv"


def test_disambiguation_config_checks_external_inputs_and_output_overwrite(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        "\n".join(
            [
                "[name_matching]",
                'researcher_file_path = "/tmp/researchers.csv"',
                'candidate_authors_file = "/tmp/authors.csv"',
                'resolved_matches_output_file_path = "/tmp/resolved.csv"',
                "",
                "[name_disambiguation]",
                f'article_author_file_path = "{tmp_path / "missing-articles.csv"}"',
                f'article_reference_file_path = "{tmp_path / "missing-references.parquet"}"',
                f'disambiguated_author_path = "{tmp_path / "outputs"}"',
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
    write_legacy_config(config_path, section_options=[f"max_workers = {value}"])
    with pytest.raises((ValueError, TypeError), match="max_workers"):
        load_name_disambiguation_config(config_path)
