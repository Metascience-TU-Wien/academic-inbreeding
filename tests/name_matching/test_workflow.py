# Copyright © 2026 TU Wien

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from acib.constants import AFFILIATION_ID, AFFILIATION_ID_PARENT, AUTHOR_ID
from acib.name_matching.config import (
    PROJECT_ROOT,
    load_name_matching_config,
)
from acib.name_matching.constants import SCOPUS_ID_MAPPING
from acib.name_matching.workflow import (
    prepare_scopus_author_affiliations,
    split_researchers_with_existing_scopus_ids,
)


def test_split_researchers_with_existing_ids_normalizes_ids_and_priority() -> None:
    researchers = pd.DataFrame(
        [
            {"sc_id": 1, "scopus_author_id": "12345"},
            {"sc_id": 2, "scopus_author_id": ""},
            {"sc_id": 3, "scopus_author_id": 67890.0},
            {"sc_id": 4, "scopus_author_id": pd.NA},
        ]
    )

    to_match, pre_resolved = split_researchers_with_existing_scopus_ids(
        researchers, "scopus_author_id"
    )

    assert to_match["sc_id"].tolist() == [2, 4]
    assert pre_resolved["sc_id"].tolist() == [1, 3]
    assert pre_resolved[AUTHOR_ID].tolist() == ["12345", "67890"]
    assert pre_resolved["priority"].tolist() == [0, 0]
    assert "scopus_author_id" not in pre_resolved


def test_split_without_existing_id_column_keeps_rows() -> None:
    researchers = pd.DataFrame({"sc_id": [1]})

    to_match, pre_resolved = split_researchers_with_existing_scopus_ids(researchers, None)

    pd.testing.assert_frame_equal(to_match, researchers)
    assert pre_resolved.empty


def test_split_rejects_missing_configured_id_column() -> None:
    with pytest.raises(KeyError, match="scopus_author_id"):
        split_researchers_with_existing_scopus_ids(pd.DataFrame({"sc_id": [1]}), "scopus_author_id")


def test_prepare_scopus_author_affiliations_prefers_parent_and_tracks_current_ids() -> None:
    authors = pd.DataFrame(
        [
            {
                "author_id": 10,
                "affiliation_id": 107244059,
                "affiliation_id_parent": 101928095,
                "current_affiliation": True,
            },
            {
                "author_id": 10,
                "affiliation_id": 60025988,
                "affiliation_id_parent": pd.NA,
                "current_affiliation": False,
            },
            {
                "author_id": 20,
                "affiliation_id": 60025988,
                "affiliation_id_parent": pd.NA,
                "current_affiliation": False,
            },
        ]
    )

    result = prepare_scopus_author_affiliations(authors)

    assert result.loc[0, AFFILIATION_ID_PARENT] == 101928095
    assert result.loc[0, AFFILIATION_ID] == 101928095
    assert result.loc[0, "affiliation_ids_current"] == [101928095]
    assert result.loc[2, "affiliation_ids_current"] == []
    assert result.loc[2, "affiliation_id_current"] == -9999


def test_prepare_scopus_author_affiliations_requires_affiliation_id() -> None:
    with pytest.raises(KeyError, match="affiliation_id"):
        prepare_scopus_author_affiliations(pd.DataFrame({AUTHOR_ID: [1]}))


def test_scopus_mapping_contains_expected_university_ids() -> None:
    assert SCOPUS_ID_MAPPING["TU Wien"] == [60018163]


def test_name_matching_config_resolves_paths_and_preserves_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    researcher_path = tmp_path / "researchers.csv"
    authors_path = tmp_path / "authors.csv"
    config_path.write_text(
        "\n".join(
            [
                "[data]",
                f'researcher_file_path = "{researcher_path}"',
                f'candidate_authors_file = "{authors_path}"',
                "",
                "[name_matching]",
                "[name_matching.outputs]",
                'directory = "results"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_name_matching_config(config_path)

    assert config.researcher_file_path == researcher_path
    assert config.candidate_authors_file == authors_path
    assert config.outputs.resolved_researchers == PROJECT_ROOT / "results/resolved_researchers.csv"
    assert config.enable_given_name_aliases is True
    assert config.enable_organisation_filter is True
    assert config.max_workers == 3
    assert config.encoding == "utf-8-sig"


def test_name_matching_output_directory_accepts_absolute_path_and_rejects_legacy_file(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        f"""[data]
researcher_file_path = "researchers.csv"
candidate_authors_file = "authors.csv"

[name_matching]
[name_matching.outputs]
directory = "{tmp_path / 'absolute-output'}"
""",
        encoding="utf-8",
    )

    config = load_name_matching_config(config_path)
    assert config.outputs.directory == tmp_path / "absolute-output"
    assert config.outputs.resolved_researchers == (
        tmp_path / "absolute-output" / "resolved_researchers.csv"
    )

    legacy_config = config_path.read_text(encoding="utf-8").replace(
        "[name_matching]\n", '[name_matching]\nresolved_matches_output_file_path = "old.csv"\n'
    )
    config_path.write_text(legacy_config, encoding="utf-8")
    with pytest.raises(ValueError, match="resolved_matches_output_file_path"):
        load_name_matching_config(config_path)


def test_name_matching_config_rejects_duplicate_stage_input_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        """[data]
researcher_file_path = "/tmp/researchers.csv"
candidate_authors_file = "/tmp/authors.csv"

[name_matching]
researcher_file_path = "/tmp/duplicate-researchers.csv"

[name_matching.outputs]
directory = "/tmp/matches"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not duplicate"):
        load_name_matching_config(config_path)


def test_name_matching_config_loads_explicit_scopus_options(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        "\n".join(
            [
                "[data]",
                f'researcher_file_path = "{tmp_path / "researchers.csv"}"',
                f'candidate_authors_file = "{tmp_path / "authors.csv"}"',
                "",
                "[name_matching]",
                "enable_given_name_aliases = false",
                "enable_organisation_filter = false",
                "max_workers = 5",
                'encoding = "utf-8"',
                "[name_matching.outputs]",
                f'directory = "{tmp_path / "matches"}"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_name_matching_config(config_path)

    assert config.enable_given_name_aliases is False
    assert config.enable_organisation_filter is False
    assert config.max_workers == 5
    assert config.encoding == "utf-8"


@pytest.mark.parametrize("max_workers", [0, -1, True, "2"])
def test_name_matching_config_rejects_invalid_worker_counts(
    tmp_path: Path, max_workers: int | bool | str
) -> None:
    value = f'"{max_workers}"' if isinstance(max_workers, str) else str(max_workers).lower()
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        "\n".join(
            [
                "[data]",
                'researcher_file_path = "/tmp/researchers.csv"',
                'candidate_authors_file = "/tmp/authors.csv"',
                "",
                "[name_matching]",
                f"max_workers = {value}",
                "[name_matching.outputs]",
                'directory = "/tmp/matches"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_workers"):
        load_name_matching_config(config_path)
