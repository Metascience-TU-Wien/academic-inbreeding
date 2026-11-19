# Copyright © 2026 TU Wien

"""OpenAlex input-adapter tests for name disambiguation."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from structlog.testing import capture_logs

from acib.name_disambiguation.constants import (
    ARTICLE_AUTHOR_NAMES,
    ARTICLE_GRANT_NUMBERS,
    ARTICLE_JOURNAL,
    OPENALEX_AUTHOR_AFFILIATION_EVIDENCE,
    OPENALEX_AUTHOR_ORCID_EVIDENCE,
)
from acib.name_disambiguation.input import (
    _ambiguous_openalex_author_ids,
    validate_matching_artifact,
)
from acib.name_disambiguation.openalex import (
    OpenAlexCitationIndex,
    load_openalex_citation_edges,
    load_openalex_work_evidence,
    openalex_author_orcid_map,
)
from acib.name_disambiguation.scoring_context import (
    build_publication_evidence_index,
    build_scoring_context,
)
from acib.schemas.researchers import ResearcherDisambiguationSelectionSchema


def _numeric_work_id(work_id: str | int) -> int:
    """Normalize a reference-index ID to the compact numeric representation."""
    if isinstance(work_id, int):
        return work_id
    return OpenAlexCitationIndex.encode_work_id(work_id)


def test_openalex_evidence_ids_use_current_researcher_row_count_rule() -> None:
    """Select all author IDs from multi-row sc_id groups without deduplicating."""
    matching = pd.DataFrame(
        {
            "sc_id": [1, 1, 2, 3, 3],
            "author_id": [
                "https://openalex.org/A1",
                "https://openalex.org/A2",
                "https://openalex.org/A3",
                "https://openalex.org/A4",
                "https://openalex.org/A4",
            ],
        }
    )

    assert _ambiguous_openalex_author_ids(matching) == {
        "https://openalex.org/A1",
        "https://openalex.org/A2",
        "https://openalex.org/A4",
    }


def test_openalex_works_are_filtered_and_adapted_without_cross_author_cities(
    tmp_path: Path,
) -> None:
    """Adapt matching evidence while keeping affiliation places author-specific."""
    work_path = tmp_path / "works.parquet"
    institution_path = tmp_path / "institutions.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "https://openalex.org/W1",
                    "topics": [],
                    "primary_topic": None,
                    "authorships": [
                        {
                            "author": {
                                "id": "https://openalex.org/A1",
                                "orcid": "https://orcid.org/0000-0002-1825-0097",
                            },
                            "raw_author_name": "Ada Author",
                            "countries": ["AT"],
                            "institutions": [
                                {"id": "https://openalex.org/I1", "country_code": "AT"}
                            ],
                            "affiliations": [
                                {
                                    "institution_ids": ["https://openalex.org/I1"],
                                    "raw_affiliation_string": "Vienna University, Vienna",
                                }
                            ],
                        },
                        {
                            "author": {
                                "id": "https://openalex.org/A2",
                                "orcid": "0000-0002-1825-0097",
                            },
                            "raw_author_name": "Co Author",
                            "countries": ["US"],
                            "institutions": [
                                {"id": "https://openalex.org/I2", "country_code": "US"}
                            ],
                            "affiliations": [
                                {
                                    "institution_ids": ["https://openalex.org/I2"],
                                    "raw_affiliation_string": "Boston Institute",
                                }
                            ],
                        },
                    ],
                    "primary_location": {
                        "source": {
                            "id": "https://openalex.org/S1",
                            "display_name": "A Journal",
                            "issn_l": "1234-5678",
                            "issn": ["1234-5678"],
                            "type": "journal",
                        },
                        "raw_source_name": "A Journal",
                    },
                    "locations": [
                        {
                            "source": {
                                "id": "https://openalex.org/S2",
                                "display_name": "Repository",
                                "type": "repository",
                            },
                            "raw_source_name": "Institutional Repository",
                        }
                    ],
                    "awards": [
                        {
                            "id": "https://openalex.org/AW1",
                            "funder_award_id": "GRANT-42",
                            "funder_id": "https://openalex.org/F1",
                        }
                    ],
                },
                {
                    "id": "https://openalex.org/W2",
                    "topics": [],
                    "primary_topic": None,
                    "authorships": [{"author": {"id": "https://openalex.org/A2"}}],
                    "primary_location": None,
                    "locations": [],
                    "awards": [],
                },
                {
                    "id": "https://openalex.org/W3",
                    "topics": [],
                    "primary_topic": None,
                    "authorships": [{"author": {"id": "https://openalex.org/A3"}}],
                    "primary_location": None,
                    "locations": [],
                    "awards": [],
                },
            ]
        ),
        work_path,
    )
    pd.DataFrame(
        [
            {
                "affiliation_id": "https://openalex.org/I1",
                "city": "Vienna",
                "country": "Austria",
                "country_code": "AT",
            },
            {
                "affiliation_id": "https://openalex.org/I2",
                "city": "Boston",
                "country": "United States",
                "country_code": "US",
            },
        ]
    ).to_parquet(institution_path, index=False)

    with capture_logs() as logs:
        evidence = load_openalex_work_evidence(
            work_path,
            candidate_author_ids={"https://openalex.org/A1", "https://openalex.org/A2"},
            affiliations_path=institution_path,
            batch_size=1,
            max_workers=2,
        )

    assert evidence["eid"].tolist() == [
        "https://openalex.org/W1",
        "https://openalex.org/W2",
    ]
    assert evidence.iloc[0]["author_ids"] == [
        "https://openalex.org/A1",
        "https://openalex.org/A2",
    ]
    assert evidence.iloc[0][ARTICLE_AUTHOR_NAMES] == ["Ada Author", "Co Author"]
    assert evidence.iloc[0][ARTICLE_JOURNAL] == [
        "1234-5678",
        "A Journal",
        "https://openalex.org/S1",
    ]
    assert evidence.iloc[0][ARTICLE_GRANT_NUMBERS] == [
        "GRANT-42",
        "https://openalex.org/AW1",
    ]
    assert "city" not in evidence.columns
    assert "country" not in evidence.columns
    assert "country_code" not in evidence.columns
    author_places = evidence.iloc[0][OPENALEX_AUTHOR_AFFILIATION_EVIDENCE]
    assert author_places == [
        {
            "author_id": "https://openalex.org/A1",
            "cities": ["Vienna"],
            "countries": ["AT", "Austria"],
        },
        {
            "author_id": "https://openalex.org/A2",
            "cities": ["Boston"],
            "countries": ["US", "United States"],
        },
    ]
    index = build_publication_evidence_index(
        evidence,
        candidate_author_ids={"https://openalex.org/A1", "https://openalex.org/A2"},
    )
    assert index is not None
    assert index.author_linked_cities["https://openalex.org/A1"] == {"vienna"}
    assert index.author_linked_cities["https://openalex.org/A2"] == {"boston"}
    assert index.author_linked_countries["https://openalex.org/A1"] == {
        "AT",
        "austria",
    }
    assert index.author_linked_countries["https://openalex.org/A2"] == {"US", "united states"}
    assert index.author_publication_cities["https://openalex.org/A1"] == set()
    assert index.author_publication_countries["https://openalex.org/A1"] == set()
    assert openalex_author_orcid_map(evidence) == {
        "https://openalex.org/A1": "0000-0002-1825-0097",
        "https://openalex.org/A2": "0000-0002-1825-0097",
    }
    events = {entry["event"] for entry in logs}
    assert "Scanning OpenAlex Works" in events
    assert "Finished OpenAlex Works scan" in events


def test_openalex_citation_loader_returns_only_edges_touching_candidate_works(
    tmp_path: Path,
) -> None:
    """Filter both citing and cited sides while retaining string Work IDs."""
    citation_path = tmp_path / "citations.parquet"
    pd.DataFrame(
        [
            ("https://openalex.org/W1", "https://openalex.org/W9"),
            ("https://openalex.org/W8", "https://openalex.org/W1"),
            ("https://openalex.org/W1", "https://openalex.org/W9"),
            ("https://openalex.org/W2", "https://openalex.org/W3"),
        ],
        columns=["citing_id", "cited_id"],
        dtype="string",
    ).to_parquet(citation_path, index=False)

    with capture_logs() as logs:
        edges = load_openalex_citation_edges(
            citation_path,
            candidate_work_ids={"https://openalex.org/W1"},
            batch_size=1,
        )

    assert edges.cited_by_work == {1: {9}}
    assert edges.citing_by_work == {1: {8}}
    events = {entry["event"] for entry in logs}
    assert "Filtering OpenAlex citation edges" in events
    assert "Finished OpenAlex citation-edge indexing" in events


def test_compact_openalex_citations_match_dataframe_reference_index() -> None:
    """Keep citation scoring evidence equivalent without a citation-edge frame."""
    authors = pd.DataFrame(
        {
            "eid": ["https://openalex.org/W1", "https://openalex.org/W2"],
            "author_id": ["https://openalex.org/A1", "https://openalex.org/A2"],
        }
    )
    references = pd.DataFrame(
        {
            "citing_eid": [
                "https://openalex.org/W1",
                "https://openalex.org/W8",
            ],
            "cited_eid": [
                "https://openalex.org/W9",
                "https://openalex.org/W1",
            ],
        }
    )
    expected = build_publication_evidence_index(
        authors,
        article_reference_df=references,
        candidate_author_ids={"https://openalex.org/A1", "https://openalex.org/A2"},
    )
    actual = build_publication_evidence_index(
        authors,
        openalex_citation_index=OpenAlexCitationIndex(
            cited_by_work={1: {9}},
            citing_by_work={1: {8}},
        ),
        candidate_author_ids={"https://openalex.org/A1", "https://openalex.org/A2"},
    )

    assert actual is not None and expected is not None
    assert actual.author_cited_references == {
        author_id: {_numeric_work_id(work_id) for work_id in work_ids}
        for author_id, work_ids in expected.author_cited_references.items()
    }
    assert actual.citing_papers_by_cited_article == {
        _numeric_work_id(cited_id): {_numeric_work_id(citing_id) for citing_id in citing_ids}
        for cited_id, citing_ids in expected.citing_papers_by_cited_article.items()
    }


def test_scoring_context_releases_consumed_openalex_citation_maps() -> None:
    """Free intermediate directional maps after building scoring indexes."""
    authors = pd.DataFrame(
        {
            "eid": ["https://openalex.org/W1"],
            "author_id": ["https://openalex.org/A1"],
        }
    )
    citation_index = OpenAlexCitationIndex(
        cited_by_work={1: {9}},
        citing_by_work={1: {8}},
    )

    context = build_scoring_context(
        authors=None,
        article_author_df=authors,
        openalex_citation_index=citation_index,
    )

    assert citation_index.cited_by_work == {}
    assert citation_index.citing_by_work == {}
    assert context.publication_index is not None
    assert context.publication_index.author_cited_references == {"https://openalex.org/A1": {9}}
    assert context.publication_index.citing_papers_by_cited_article == {
        1: {8},
        9: {1},
    }


def test_openalex_matching_artifact_preserves_author_ids_as_strings(tmp_path: Path) -> None:
    """Select the OpenAlex schema rather than numeric Scopus ID coercion."""
    artifact = tmp_path / "resolved.csv"
    pd.DataFrame(
        {
            "sc_id": [7],
            "author_id": ["https://openalex.org/A123"],
            "priority": [1],
        }
    ).to_csv(artifact, index=False)

    validated = validate_matching_artifact(artifact, source="openalex")

    assert validated.loc[0, "author_id"] == "https://openalex.org/A123"


def test_disambiguation_selection_schema_accepts_openalex_author_ids() -> None:
    """Keep stable URL IDs valid in the final selection artifact."""
    selection = pd.DataFrame(
        {
            "sc_id": [7],
            "status": ["assigned_single_cluster"],
            "selected_disambiguated_cluster_id": ["https://openalex.org/A123"],
            "selected_author_ids": [["https://openalex.org/A123"]],
            "possible_clusters": [[]],
            "inferred_asjc_codes": [[]],
            "scopus_asjc_codes": [[]],
            "inferred_asjc_labels": [[]],
            "reason": ["selected"],
        }
    )

    validated = ResearcherDisambiguationSelectionSchema.validate(selection, lazy=True)

    assert validated.loc[0, "selected_disambiguated_cluster_id"] == ("https://openalex.org/A123")


def test_openalex_orcid_map_ignores_conflicting_or_invalid_values() -> None:
    """Avoid arbitrary ORCID selection when one author has conflicting evidence."""
    article_authors = pd.DataFrame(
        {
            OPENALEX_AUTHOR_ORCID_EVIDENCE: [
                [{"author_id": "https://openalex.org/A1", "orcid": "0000-0002-1825-0097"}],
                [{"author_id": "https://openalex.org/A1", "orcid": "0000-0001-5109-3700"}],
                [{"author_id": "https://openalex.org/A2", "orcid": "not-an-orcid"}],
            ]
        }
    )

    assert openalex_author_orcid_map(article_authors) == {}
