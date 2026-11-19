# Copyright © 2026 TU Wien

"""Phase 4 checks for OpenAlex Work-Field scoring and selection."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from acib.name_disambiguation.constants import (
    OPENALEX_WORK_FIELD_IDS,
)
from acib.name_disambiguation.models import (
    BatchDisambiguationResult,
    FinalAuthorSelection,
    final_author_selection_record,
)
from acib.name_disambiguation.openalex import load_openalex_work_evidence
from acib.name_disambiguation.output import collect_final_selections
from acib.name_disambiguation.parsing import author_from_row
from acib.name_disambiguation.reporting import render_text_report
from acib.name_disambiguation.scoring import score_pair
from acib.name_disambiguation.scoring_context import build_scoring_context
from acib.name_disambiguation.service import select_final_author
from acib.name_matching.openalex_fields import (
    OPENALEX_FIELD_IDS_COLUMN,
    OPENALEX_FIELD_LABELS_COLUMN,
)
from acib.schemas.researchers import ResearcherDisambiguationSelectionSchema

_FIELD_11 = "https://openalex.org/fields/11"
_FIELD_12 = "https://openalex.org/fields/12"


def test_work_adapter_extracts_topic_fields_and_uses_primary_fallback(
    tmp_path: Path,
) -> None:
    """Adapt topic Fields, falling back to primary topic only when needed."""
    works_path = tmp_path / "works.parquet"
    affiliations_path = tmp_path / "institutions.parquet"
    rows = [
        {
            "id": f"https://openalex.org/W{index}",
            "authorships": [{"author": {"id": f"https://openalex.org/A{index}"}}],
            "primary_location": None,
            "locations": [],
            "awards": [],
            "topics": topics,
            "primary_topic": {"field": {"id": primary}},
        }
        for index, topics, primary in (
            (1, [{"field": {"id": _FIELD_11}}, {"field": {"id": _FIELD_12}}], _FIELD_12),
            (2, [], _FIELD_12),
        )
    ]
    pq.write_table(pa.Table.from_pylist(rows), works_path)
    pd.DataFrame(columns=["affiliation_id", "city", "country", "country_code"]).to_parquet(
        affiliations_path, index=False
    )

    evidence = load_openalex_work_evidence(
        works_path,
        candidate_author_ids={
            "https://openalex.org/A1",
            "https://openalex.org/A2",
        },
        affiliations_path=affiliations_path,
    )

    assert evidence[OPENALEX_WORK_FIELD_IDS].tolist() == [
        [_FIELD_11, _FIELD_12],
        [_FIELD_12],
    ]


def _openalex_author(author_id: str) -> dict[str, object]:
    return {
        "author_id": author_id,
        "name": "Doe, Jane",
        "surname": "Doe",
        "given_name": "Jane",
        "initials": "J",
        "indexed_name": "Doe J",
        "name_variants": "[]",
        "subject_areas": "[]",
    }


def test_pair_score_uses_article_field_overlap_at_three_points() -> None:
    """Score exact shared Work Fields without adding separate profile evidence."""
    left = author_from_row(_openalex_author("https://openalex.org/A1"))
    right = author_from_row(_openalex_author("https://openalex.org/A2"))
    article_authors = pd.DataFrame(
        [
            {
                "eid": "https://openalex.org/W1",
                "author_id": "https://openalex.org/A1",
                OPENALEX_WORK_FIELD_IDS: [_FIELD_11],
            },
            {
                "eid": "https://openalex.org/W2",
                "author_id": "https://openalex.org/A2",
                OPENALEX_WORK_FIELD_IDS: [_FIELD_11],
            },
        ]
    )

    context = build_scoring_context(
        authors=[left, right],
        article_author_df=article_authors,
    )
    decision = score_pair(left, right, context=context)

    field_components = [
        component for component in decision.components if component.label == "article_field_overlap"
    ]
    assert len(field_components) == 1
    assert field_components[0].weight == 3.0
    assert _FIELD_11 in field_components[0].reason
    assert "profile_subject_overlap" not in {component.label for component in decision.components}


def _openalex_selection_inputs(
    candidate_ids: list[str],
    work_fields: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    inferred = [_FIELD_11]
    authors = pd.DataFrame(
        [
            {
                **_openalex_author(author_id),
                "researcher_name": "Doe, Jane",
                "canonical_given_name": "jane",
                OPENALEX_FIELD_IDS_COLUMN: inferred,
                OPENALEX_FIELD_LABELS_COLUMN: ["Agricultural and Biological Sciences"],
            }
            for author_id in candidate_ids
        ]
    )
    clusters = pd.DataFrame(
        [
            {
                "cluster_id": index,
                "author_id": author_id,
                "cluster_size": 1,
                "disambiguated_cluster_id": author_id,
            }
            for index, author_id in enumerate(candidate_ids, start=1)
        ]
    )
    article_authors = pd.DataFrame(
        [
            {
                "eid": f"https://openalex.org/W{index}",
                "author_id": author_id,
                OPENALEX_WORK_FIELD_IDS: work_fields.get(author_id, []),
            }
            for index, author_id in enumerate(candidate_ids, start=1)
        ]
    )
    context = build_scoring_context(
        authors=authors,
        article_author_df=article_authors,
    )
    return authors, clusters, context


def test_openalex_final_selection_assigns_unique_best_article_field() -> None:
    """Assign the unique cluster matching inferred Fields and retain Field IDs."""
    authors, clusters, context = _openalex_selection_inputs(
        ["https://openalex.org/A1", "https://openalex.org/A2"],
        {
            "https://openalex.org/A1": [_FIELD_11],
            "https://openalex.org/A2": [_FIELD_12],
        },
    )

    selection, scores = select_final_author(authors, clusters, "researcher-1", context)

    assert selection.status == "assigned_by_article_field"
    assert selection.selected_author_ids == ["https://openalex.org/A1"]
    assert selection.inferred_openalex_field_ids == [_FIELD_11]
    assert selection.openalex_field_ids == [_FIELD_11]
    assert scores["openalex_field_overlap_count"].tolist() == [1, 0]
    report = render_text_report(
        pd.DataFrame(),
        pd.DataFrame(columns=["cluster_id"]),
        selection,
        scores,
        source="openalex",
    )
    assert "openalex_field_overlap_ids" in report
    assert "overlap_codes=" not in report


def test_openalex_final_selection_keeps_field_ties_manual() -> None:
    """Do not turn equal Field evidence into an arbitrary cluster choice."""
    authors, clusters, context = _openalex_selection_inputs(
        ["https://openalex.org/A1", "https://openalex.org/A2"],
        {
            "https://openalex.org/A1": [_FIELD_11],
            "https://openalex.org/A2": [_FIELD_11],
        },
    )

    selection, scores = select_final_author(authors, clusters, "researcher-1", context)

    assert selection.status == "manual_selection_required"
    assert selection.selected_author_ids == []
    assert selection.possible_clusters == [
        ["https://openalex.org/A1"],
        ["https://openalex.org/A2"],
    ]
    assert scores["openalex_field_overlap_count"].tolist() == [1, 1]


def test_openalex_selection_output_uses_field_columns_only() -> None:
    """Persist OpenAlex Field evidence without Scopus ASJC columns."""
    selection = FinalAuthorSelection(
        sc_id="1",
        status="assigned_by_article_field",
        selected_cluster_id=1,
        selected_disambiguated_cluster_id="https://openalex.org/A1",
        selected_author_ids=["https://openalex.org/A1"],
        possible_clusters=[["https://openalex.org/A1"]],
        inferred_asjc_codes=[],
        scopus_asjc_codes=[],
        inferred_asjc_labels=[],
        reason="unique article Field overlap",
        inferred_openalex_field_ids=[_FIELD_11],
        inferred_openalex_field_labels=["Agricultural and Biological Sciences"],
        openalex_field_ids=[_FIELD_11],
    )
    serialized = final_author_selection_record(selection, source="openalex")
    collected = collect_final_selections(
        [
            BatchDisambiguationResult(
                sc_id="1",
                authors=pd.DataFrame(
                    [
                        {
                            "author_id": "https://openalex.org/A1",
                            OPENALEX_FIELD_IDS_COLUMN: [_FIELD_11],
                            OPENALEX_FIELD_LABELS_COLUMN: ["Agricultural and Biological Sciences"],
                        }
                    ]
                ),
                decisions=pd.DataFrame(),
                clusters=pd.DataFrame(),
                final_selection=selection,
                final_selection_scores=pd.DataFrame(),
            )
        ]
    )
    validated = ResearcherDisambiguationSelectionSchema.validate(
        collected,
        lazy=True,
    )

    assert "scopus_asjc_codes" not in validated
    assert "inferred_asjc_codes" not in validated
    assert validated.loc[0, "openalex_field_ids"] == [_FIELD_11]
    assert validated.loc[0, "status"] == "assigned_by_article_field"
    assert "scopus_asjc_codes" not in serialized


def test_scopus_selection_serialization_keeps_asjc_columns() -> None:
    """Keep the Scopus output contract separate from OpenAlex Fields."""
    selection = FinalAuthorSelection(
        sc_id="researcher-1",
        status="assigned_by_article_asjc",
        selected_cluster_id=1,
        selected_disambiguated_cluster_id="123",
        selected_author_ids=["123"],
        possible_clusters=[["123"]],
        inferred_asjc_codes=[2700],
        scopus_asjc_codes=[2700],
        inferred_asjc_labels=["Medicine"],
        reason="unique article ASJC overlap",
    )

    serialized = final_author_selection_record(selection, source="scopus")

    assert serialized["scopus_asjc_codes"] == [2700]
    assert serialized["inferred_asjc_codes"] == [2700]
    assert "openalex_field_ids" not in serialized
    assert "inferred_openalex_field_ids" not in serialized


def test_scopus_article_subject_scoring_remains_asjc_based() -> None:
    """Guard Scopus article overlap label and weight after adding OpenAlex Fields."""
    left = author_from_row(
        {
            **_openalex_author("100"),
            "subject_areas": "[]",
        }
    )
    right = author_from_row(
        {
            **_openalex_author("200"),
            "subject_areas": "[]",
        }
    )
    article_authors = pd.DataFrame(
        [
            {"eid": "A1", "author_id": "100", "subject_areas": '[{"area":"Medicine","code":2700}]'},
            {"eid": "A2", "author_id": "200", "subject_areas": '[{"area":"Medicine","code":2700}]'},
        ]
    )

    decision = score_pair(
        left,
        right,
        context=build_scoring_context(
            authors=[left, right],
            article_author_df=article_authors,
        ),
    )

    assert any(
        component.label == "article_subject_overlap" and component.weight == 3.0
        for component in decision.components
    )
    assert all(component.label != "article_field_overlap" for component in decision.components)
