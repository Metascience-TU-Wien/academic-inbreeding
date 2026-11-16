# Copyright © 2026 TU Wien

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest

from acib.constants import AFFILIATION_ID, AUTHOR_ID
from acib.name_matching.config import load_name_matching_config
from acib.name_matching.input import (
    _read_openalex_candidates,
)
from acib.name_matching.openalex import openalex_candidate_rows
from acib.name_matching.workflow import match_researchers
from acib.schemas import OpenAlexAuthorAffiliationInputSchema


def _row(
    *,
    alternatives: list[str] | None = None,
    raw_names: list[str] | None = None,
) -> dict[str, object]:
    """Return one observed-schema author-affiliation row."""
    return {
        "author_id": "https://openalex.org/A123",
        "institution_id": "https://openalex.org/I145847075",
        "author_display_name": "Anna Smith",
        "author_name_alternatives": alternatives,
        "raw_author_names": raw_names,
        "observations": [
            {
                "work_id": "https://openalex.org/W42",
                "raw_author_name": "Smith, Anna",
            }
        ],
    }


def test_openalex_schema_requires_string_ids_and_allows_repeated_authors() -> None:
    frame = pd.DataFrame([_row(), _row()])

    validated = OpenAlexAuthorAffiliationInputSchema.validate(frame)

    assert validated[AUTHOR_ID].tolist() == [
        "https://openalex.org/A123",
        "https://openalex.org/A123",
    ]
    assert validated["institution_id"].tolist() == [
        "https://openalex.org/I145847075",
        "https://openalex.org/I145847075",
    ]

    invalid = frame.copy()
    invalid[AUTHOR_ID] = [123, 123]
    with pytest.raises(pa.errors.SchemaError):
        OpenAlexAuthorAffiliationInputSchema.validate(invalid)


def test_adapter_preserves_name_provenance_orientation_and_work_id() -> None:
    adapted = openalex_candidate_rows(
        [
            {
                **_row(alternatives=["Anna Smith"], raw_names=["A. Smith"]),
                "observations": [
                    {
                        "work_id": "https://openalex.org/W42",
                        "raw_author_name": "Smith, Anna",
                    }
                ],
            }
        ]
    )

    display = adapted.loc[adapted["name_source"].eq("author_display_name")]
    assert display[AUTHOR_ID].eq("https://openalex.org/A123").all()
    assert display[AFFILIATION_ID].eq("https://openalex.org/I145847075").all()
    assert set(display["name_orientation"]) == {"given_first", "family_first"}
    evidence = adapted.loc[adapted["name_source"].eq("observation_raw_author_name")]
    assert evidence["work_id"].tolist() == ["https://openalex.org/W42"]
    assert evidence["raw_name"].tolist() == ["Smith, Anna"]
    assert json.loads(evidence.iloc[0]["name_variants"])[0]["source"] == (
        "observation_raw_author_name"
    )


def test_adapter_retains_null_and_empty_list_states() -> None:
    rows = [
        _row(alternatives=None, raw_names=None),
        _row(alternatives=[], raw_names=[]),
    ]

    adapted = openalex_candidate_rows(rows)

    states = set(
        zip(
            adapted["author_name_alternatives_is_null"],
            adapted["raw_author_names_is_null"],
            strict=True,
        )
    )
    assert states == {(True, True), (False, False)}


def test_adapter_groups_repeated_publication_names_without_losing_work_ids() -> None:
    row = _row()
    row["observations"] = [
        {"work_id": f"https://openalex.org/W{index}", "raw_author_name": "Smith, Anna"}
        for index in range(12)
    ]

    adapted = openalex_candidate_rows([row])
    publication_evidence = adapted.loc[adapted["name_source"].eq("observation_raw_author_name")]

    assert len(publication_evidence) == 1
    assert publication_evidence.iloc[0]["work_ids"] == [
        f"https://openalex.org/W{index}" for index in range(12)
    ]
    provenance = json.loads(publication_evidence.iloc[0]["name_variants"])
    assert provenance[0]["work_ids"] == publication_evidence.iloc[0]["work_ids"]


def test_unsplittable_variant_does_not_remove_other_matchable_evidence() -> None:
    candidates = openalex_candidate_rows(
        [
            {
                "author_id": "https://openalex.org/A123",
                "institution_id": "https://openalex.org/I145847075",
                "author_display_name": "Anna Smith",
                "author_name_alternatives": ["Cher"],
                "raw_author_names": [],
            }
        ]
    )
    researchers = pd.DataFrame(
        [
            {
                "sc_id": 1,
                "person_name": "Anna",
                "person_surname": "Smith",
                "university_name": "TU Wien",
            }
        ]
    )

    result = match_researchers(
        researchers,
        candidates,
        enable_organisation_filter=False,
    )

    assert result.matched[AUTHOR_ID].tolist() == ["https://openalex.org/A123"]
    assert "Cher" in candidates["raw_name"].tolist()


def test_openalex_csv_structured_fields_are_strict_and_preserve_empty_lists(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {
                "author_id": "https://openalex.org/A1",
                "institution_id": "https://openalex.org/I1",
                "author_display_name": "Anna Smith",
                "author_name_alternatives": "null",
                "raw_author_names": "[]",
                "observations": "[]",
            },
            {
                "author_id": "https://openalex.org/A2",
                "institution_id": "https://openalex.org/I2",
                "author_display_name": "Bob Jones",
                "author_name_alternatives": "[]",
                "raw_author_names": "null",
                "observations": "[]",
            },
        ]
    ).to_csv(path, index=False)

    loaded = _read_openalex_candidates(path)

    assert loaded.loc[0, "author_name_alternatives"] is None
    assert loaded.loc[0, "raw_author_names"] == []
    assert loaded.loc[1, "author_name_alternatives"] == []
    assert loaded.loc[1, "raw_author_names"] is None

    pd.DataFrame(
        [
            {
                "author_id": "https://openalex.org/A3",
                "institution_id": "https://openalex.org/I3",
                "author_name_alternatives": "not-json",
            }
        ]
    ).to_csv(path, index=False)
    with pytest.raises(json.JSONDecodeError):
        _read_openalex_candidates(path)


def test_openalex_config_uses_source_and_single_canonical_path(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        f"""[data]
source = "openalex"
researcher_file_path = "{tmp_path / "researchers.csv"}"
path = "{tmp_path / "candidates.parquet"}"

[name_matching]
[name_matching.outputs]
directory = "{tmp_path / "results"}"
""",
        encoding="utf-8",
    )

    config = load_name_matching_config(config_path)

    assert config.source == "openalex"
    assert config.researcher_file_path == tmp_path / "researchers.csv"
    assert config.candidate_authors_file == tmp_path / "candidates.parquet"
    assert config.enable_organisation_filter is True
    assert config.existing_scopus_author_id_column is None

    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            f'path = "{tmp_path / "candidates.parquet"}"',
            f'path = "{tmp_path / "candidates.parquet"}"\n'
            f'candidate_authors_file = "{tmp_path / "other.csv"}"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only one candidate-author path"):
        load_name_matching_config(config_path)


def test_openalex_config_file_points_to_observed_candidate_input() -> None:
    config = load_name_matching_config(Path("config/name_matching_openalex.toml"))

    assert config.source == "openalex"
    assert config.candidate_authors_file.name == "austrian_author_affiliation_pairs.parquet"
    assert config.outputs.directory.name == "name_matching_openalex"
