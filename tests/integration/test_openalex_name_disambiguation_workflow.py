# Copyright © 2026 TU Wien

"""End-to-end OpenAlex Works evidence workflow test."""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from acib.constants import RESEARCHER_ID
from acib.name_disambiguation.config import (
    NameDisambiguationConfig,
    NameDisambiguationOutputs,
)
from acib.name_disambiguation.workflow import run_name_disambiguation


def test_openalex_workflow_scores_journal_grant_and_author_affiliation(
    tmp_path: Path,
) -> None:
    """Run OpenAlex input adaptation through persisted disambiguation scores."""
    matching_path = tmp_path / "matching.csv"
    works_path = tmp_path / "works.parquet"
    citations_path = tmp_path / "citations.parquet"
    affiliations_path = tmp_path / "institutions.parquet"
    output_directory = tmp_path / "outputs"

    pd.DataFrame(
        [
            {
                RESEARCHER_ID: 1,
                "author_id": author_id,
                "priority": 1,
                "eid": work_id,
                "researcher_name": "Doe, Jane",
                "author_name": "Doe, Jane",
                "name": "Doe, Jane",
                "surname": "Doe",
                "given_name": "Jane",
                "indexed_name": "Doe J",
                "initials": "J",
                "publication_range": "{2020,2022}",
                "name_variants": "[]",
                "subject_areas": "[]",
            }
            for author_id, work_id in (
                ("https://openalex.org/A1", "https://openalex.org/W1"),
                ("https://openalex.org/A2", "https://openalex.org/W2"),
            )
        ]
    ).to_csv(matching_path, index=False)

    source = {
        "id": "https://openalex.org/S1",
        "display_name": "Journal of Testing",
        "issn_l": "1234-5678",
        "issn": ["1234-5678"],
        "type": "journal",
    }
    works = []
    for author_id, work_id in (
        ("https://openalex.org/A1", "https://openalex.org/W1"),
        ("https://openalex.org/A2", "https://openalex.org/W2"),
    ):
        works.append(
            {
                "id": work_id,
                "topics": [],
                "primary_topic": None,
                "authorships": [
                    {
                        "author": {
                            "id": author_id,
                            "orcid": "0000-0002-1825-0097",
                        },
                        "raw_author_name": "Jane Doe",
                        "countries": ["AT"],
                        "institutions": [
                            {
                                "id": "https://openalex.org/I1",
                                "country_code": "AT",
                            }
                        ],
                        "affiliations": [{"institution_ids": ["https://openalex.org/I1"]}],
                    }
                ],
                "primary_location": {
                    "source": source,
                    "raw_source_name": "Journal of Testing",
                },
                "locations": [],
                "awards": [
                    {
                        "id": "https://openalex.org/AW1",
                        "funder_award_id": "PROJECT-123",
                        "funder_id": "https://openalex.org/F1",
                    }
                ],
            }
        )
    pq.write_table(pa.Table.from_pylist(works), works_path)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "affiliation_id": "https://openalex.org/I1",
                    "city": "Vienna",
                    "country": "Austria",
                    "country_code": "AT",
                }
            ]
        ),
        affiliations_path,
    )
    pd.DataFrame(columns=["citing_id", "cited_id"]).astype("string").to_parquet(
        citations_path, index=False
    )

    config = NameDisambiguationConfig(
        matching_output=matching_path,
        article_author_file_path=works_path,
        article_reference_file_path=citations_path,
        source="openalex",
        author_affiliations_file_path=affiliations_path,
        outputs=NameDisambiguationOutputs(directory=output_directory),
        enable_reporting=False,
    )

    selections = run_name_disambiguation(config)
    decisions = pd.read_csv(config.outputs.pair_decisions)
    components = literal_eval(decisions.iloc[0]["components"])
    labels = {component["label"] for component in components}

    assert selections["status"].tolist() == ["assigned_single_cluster"]
    assert {
        "orcid_match",
        "journal_match",
        "grant_number_match",
        "author_address_match",
    }.issubset(labels)
    assert "publication_address_match" not in labels
