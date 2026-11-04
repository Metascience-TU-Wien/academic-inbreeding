# Copyright © 2026 TU Wien

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest

from acib.constants import AUTHOR_ID, RESEARCHER_ID
from acib.name_matching.config import NameMatchingConfig, NameMatchingOutputs
from acib.name_matching.input import prepare_name_matching_inputs
from acib.name_matching.output import save_name_matching_results
from acib.name_matching.workflow import compute_name_matching, run_name_matching


def test_run_name_matching_restores_metadata_and_preserves_matches(tmp_path: Path) -> None:
    researchers = pd.DataFrame(
        [
            {
                "sc_id": 1,
                "person_name": "Anna",
                "person_surname": "Smith",
                "person_fullname": "Anna Smith",
                "university_name": "TU Wien",
                "scopus_author_id": "",
                "researcher_metadata": "preserve-me",
            },
            {
                "sc_id": 2,
                "person_name": "Known",
                "person_surname": "Researcher",
                "person_fullname": "Known Researcher",
                "university_name": "BOKU Wien",
                "scopus_author_id": "999",
                "researcher_metadata": "also-preserve",
            },
            {
                "sc_id": 3,
                "person_name": "Nobody",
                "person_surname": "Matches",
                "person_fullname": "Nobody Matches",
                "university_name": "TU Wien",
                "scopus_author_id": "",
                "researcher_metadata": "not-resolved",
            },
            {
                "sc_id": 4,
                "person_name": "Maria",
                "person_surname": "Jones",
                "person_fullname": "Maria Jones",
                "university_name": "TU Wien",
                "scopus_author_id": "",
                "researcher_metadata": "fallback-match",
            },
        ]
    )
    authors = pd.DataFrame(
        [
            {
                "author_id": 10,
                "eid": "E1",
                "author_name": "smith, anna",
                "indexed_name": "Smith A.",
                "initials": "A.",
                "given_name": "Anna",
                "surname": "Smith",
                "publication_range": "2020-2024",
                "name_variants": "[]",
                "subject_areas": "[]",
                "affiliation_id": 60018163,
                "affiliation_id_parent": "60018163.0",
            },
            {
                "author_id": 11,
                "eid": "E2",
                "author_name": "",
                "name": "Jones, Maria",
                "indexed_name": "Jones M.",
                "initials": "M.",
                "given_name": "",
                "surname": "",
                "publication_range": "2021-2024",
                "name_variants": "[]",
                "subject_areas": "[]",
                "affiliation_id": 60018163,
                "affiliation_id_parent": "60018163.0",
            },
        ]
    )
    researcher_path = tmp_path / "researchers.csv"
    author_path = tmp_path / "authors.csv"
    output_path = tmp_path / "resolved_researchers.csv"
    researchers.to_csv(researcher_path, index=False)
    authors.to_csv(author_path, index=False)
    config = NameMatchingConfig(
        researcher_file_path=researcher_path,
        candidate_authors_file=author_path,
        outputs=NameMatchingOutputs(directory=output_path.parent),
        existing_scopus_author_id_column="scopus_author_id",
    )

    output = run_name_matching(config)

    assert output_path.exists()
    assert len(output) == 3
    assert output[RESEARCHER_ID].notna().all()
    assert output[RESEARCHER_ID].tolist() == [1, 2, 4]
    matched = output.loc[output["priority"].eq(1)].iloc[0]
    assert matched[AUTHOR_ID] == 10
    assert matched["researcher_name"] == "smith, anna"
    assert matched["author_name"] == "smith, anna"
    assert matched["person_name"] == "Anna"
    assert matched["person_surname"] == "Smith"
    assert matched["person_fullname"] == "Anna Smith"
    assert matched["researcher_metadata"] == "preserve-me"
    assert not any(column.endswith("_researcher_metadata") for column in output.columns)
    assert matched["affiliation_id_parent"] == 60018163
    passthrough = output.loc[output["priority"].eq(0)].iloc[0]
    assert passthrough[AUTHOR_ID] == 999
    assert "scopus_author_id" not in output.columns
    assert passthrough["researcher_metadata"] == "also-preserve"
    fallback = output.loc[output[RESEARCHER_ID].eq(4)].iloc[0]
    assert fallback[AUTHOR_ID] == 11
    assert fallback["researcher_name"] == "jones, maria"
    assert fallback["author_name"] == "jones, maria"
    assert output_path.stat().st_size > 0


def test_compute_name_matching_is_in_memory_and_persistence_is_explicit(
    tmp_path: Path,
) -> None:
    researchers = pd.DataFrame(
        [
            {
                "sc_id": 11,
                "person_name": "Anna",
                "person_surname": "Smith",
                "person_fullname": "Anna Smith",
                "university_name": "TU Wien",
                "source_note": "preserved",
                "scopus_author_id": float("nan"),
            }
        ]
    )
    authors = pd.DataFrame(
        [
            {
                "author_id": 101,
                "eid": "E-11",
                "author_name": "smith, anna",
                "indexed_name": "Smith A.",
                "initials": "A.",
                "given_name": "Anna",
                "surname": "Smith",
                "publication_range": "2020-2024",
                "name_variants": "[]",
                "subject_areas": "[]",
                "affiliation_id": 60018163,
                "affiliation_id_parent": "60018163.0",
            }
        ]
    )
    inputs = prepare_name_matching_inputs(researchers, authors)
    output_directory = tmp_path / "not-created-by-computation"
    config = NameMatchingConfig(
        researcher_file_path=tmp_path / "unused-researchers.csv",
        candidate_authors_file=tmp_path / "unused-authors.csv",
        outputs=NameMatchingOutputs(directory=output_directory),
        enable_organisation_filter=False,
        existing_scopus_author_id_column="scopus_author_id",
    )

    result = compute_name_matching(inputs, config)

    assert result.resolved_researchers[[RESEARCHER_ID, AUTHOR_ID, "priority"]].to_dict(
        "records"
    ) == [{RESEARCHER_ID: 11, AUTHOR_ID: 101, "priority": 1}]
    assert result.resolved_researchers.loc[0, "source_note"] == "preserved"
    assert result.matching_details is not None
    assert not output_directory.exists()

    save_name_matching_results(result, config)
    persisted = pd.read_csv(config.outputs.resolved_researchers)
    pd.testing.assert_frame_equal(
        persisted[[RESEARCHER_ID, AUTHOR_ID, "priority", "source_note"]],
        result.resolved_researchers[[RESEARCHER_ID, AUTHOR_ID, "priority", "source_note"]],
        check_dtype=False,
    )
    with pytest.raises(FileExistsError, match="overwrite=True"):
        save_name_matching_results(result, config)


@pytest.mark.parametrize(
    "researchers",
    [
        pd.DataFrame({"person_name": ["Missing ID"]}),
        pd.DataFrame(
            {
                "sc_id": [1, 1],
                "person_name": ["First", "Duplicate"],
                "person_surname": ["Example", "Example"],
            }
        ),
        pd.DataFrame(
            {
                "sc_id": [pd.NA],
                "person_name": ["Missing"],
                "person_surname": ["ID"],
            }
        ),
    ],
    ids=["missing-column", "duplicate-id", "missing-id"],
)
def test_run_name_matching_rejects_invalid_researcher_ids(
    tmp_path: Path, researchers: pd.DataFrame
) -> None:
    researcher_path = tmp_path / "researchers.csv"
    author_path = tmp_path / "authors.csv"
    output_path = tmp_path / "resolved.csv"
    researchers.to_csv(researcher_path, index=False)
    pd.DataFrame(
        [{"author_id": 10, "author_name": "smith, anna", "affiliation_id": 60018163}]
    ).to_csv(author_path, index=False)
    config = NameMatchingConfig(
        researcher_file_path=researcher_path,
        candidate_authors_file=author_path,
        outputs=NameMatchingOutputs(directory=output_path.parent),
        existing_scopus_author_id_column=None,
    )

    with pytest.raises(pa.errors.SchemaErrors, match="sc_id"):
        run_name_matching(config)
    assert not output_path.exists()
