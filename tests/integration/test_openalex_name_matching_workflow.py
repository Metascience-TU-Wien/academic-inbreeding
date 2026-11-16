# Copyright © 2026 TU Wien

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from acib.constants import AFFILIATION_ID, AUTHOR_ID, RESEARCHER_ID
from acib.name_matching.config import NameMatchingConfig, NameMatchingOutputs
from acib.name_matching.workflow import run_name_matching


def test_openalex_workflow_reads_arrow_json_and_preserves_stable_ids(
    tmp_path: Path,
) -> None:
    researcher_path = tmp_path / "researchers.csv"
    pd.DataFrame(
        [
            {
                "sc_id": 1,
                "person_name": "Anna",
                "person_surname": "Smith",
                "person_fullname": "Anna Smith",
                "university_name": "TU Wien",
                "organization_names_relevant": ["Fakultät Wirtschaft"],
                "scopus_author_id": 999.0,
                "researcher_note": "retained",
            }
        ]
    ).to_csv(researcher_path, index=False)

    candidate_path = tmp_path / "openalex.parquet"
    observations = pa.array(
        [
            [
                {
                    "work_id": "https://openalex.org/W42",
                    "raw_author_name": "Smith, Anna",
                }
            ]
        ],
        type=pa.list_(
            pa.struct(
                [
                    pa.field("work_id", pa.string()),
                    pa.field("raw_author_name", pa.string()),
                ]
            )
        ),
    )
    table = pa.table(
        {
            "author_id": ["https://openalex.org/A123"],
            "institution_id": ["https://openalex.org/I145847075"],
            "author_display_name": ["Anna Smith"],
            "author_name_alternatives": pa.array(['["Smith, Anna", "A. Smith"]'], type=pa.json_()),
            "raw_author_names": [["A. Smith"]],
            "observations": observations,
        }
    )
    pq.write_table(table, candidate_path)

    config = NameMatchingConfig(
        researcher_file_path=researcher_path,
        candidate_authors_file=candidate_path,
        outputs=NameMatchingOutputs(directory=tmp_path / "output"),
        source="openalex",
        existing_scopus_author_id_column="scopus_author_id",
    )

    output = run_name_matching(config)
    persisted = pd.read_csv(
        config.outputs.resolved_researchers,
        dtype={AUTHOR_ID: "string", AFFILIATION_ID: "string"},
    )

    assert output[RESEARCHER_ID].tolist() == [1]
    assert output[AUTHOR_ID].tolist() == ["https://openalex.org/A123"]
    assert output[AFFILIATION_ID].tolist() == ["https://openalex.org/I145847075"]
    assert output["researcher_note"].tolist() == ["retained"]
    assert json.loads(output.loc[0, "inferred_openalex_field_ids"]) == [
        "https://openalex.org/fields/14"
    ]
    assert json.loads(output.loc[0, "inferred_openalex_field_labels"]) == [
        "Business, Management and Accounting"
    ]
    assert "scopus_author_id" not in output.columns
    assert persisted[AUTHOR_ID].tolist() == ["https://openalex.org/A123"]
    assert persisted[AFFILIATION_ID].tolist() == ["https://openalex.org/I145847075"]
    assert json.loads(persisted.loc[0, "inferred_openalex_field_ids"]) == [
        "https://openalex.org/fields/14"
    ]
    evidence = json.loads(persisted.loc[0, "name_variants"])
    assert evidence[0]["raw_name"] == "Anna Smith"
    assert evidence[0]["source"] == "author_display_name"
