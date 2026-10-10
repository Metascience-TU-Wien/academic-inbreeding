# Copyright © 2026 TU Wien

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest

from acib.constants import AUTHOR_ID, RESEARCHER_ID
from acib.name_matching.config import NameMatchingConfig
from acib.name_matching.workflow import run_name_matching


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
        ]
    )
    researcher_path = tmp_path / "researchers.csv"
    author_path = tmp_path / "authors.csv"
    output_path = tmp_path / "resolved.csv"
    researchers.to_csv(researcher_path, index=False)
    authors.to_csv(author_path, index=False)
    config = NameMatchingConfig(
        researcher_file_path=researcher_path,
        candidate_authors_file=author_path,
        resolved_matches_output_file_path=output_path,
        existing_scopus_author_id_column="scopus_author_id",
    )

    output = run_name_matching(config)

    assert output_path.exists()
    assert len(output) == 2
    assert output[RESEARCHER_ID].notna().all()
    assert output[RESEARCHER_ID].tolist() == [1, 2]
    matched = output.loc[output["priority"].eq(1)].iloc[0]
    assert matched[AUTHOR_ID] == 10
    assert matched["researcher_name"] == "smith,  anna"
    assert matched["author_name"] == "smith,  anna"
    assert matched["person_name"] == "Anna"
    assert matched["person_surname"] == "Smith"
    assert matched["person_fullname"] == "Anna Smith"
    assert matched["affiliation_id_parent"] == 60018163
    assert matched["researcher_metadata"] == "preserve-me"
    passthrough = output.loc[output["priority"].eq(0)].iloc[0]
    assert passthrough[AUTHOR_ID] == 999
    assert passthrough["scopus_author_id"] == 999.0
    assert passthrough["researcher_metadata"] == "also-preserve"
    assert output_path.stat().st_size > 0


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
        resolved_matches_output_file_path=output_path,
        existing_scopus_author_id_column=None,
    )

    with pytest.raises(pa.errors.SchemaErrors, match="sc_id"):
        run_name_matching(config)
    assert not output_path.exists()
