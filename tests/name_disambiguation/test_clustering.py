# Copyright © 2026 TU Wien

import pandas as pd

from acib.name_disambiguation.clustering import cluster_authors
from acib.name_disambiguation.models import (
    BatchDisambiguationResult,
    FinalAuthorSelection,
)
from acib.name_disambiguation.output import (
    collect_clusters,
    collect_final_selection_scores,
    collect_final_selections,
    collect_pair_decisions,
)
from acib.name_disambiguation.service import disambiguate_batches


def test_cluster_representative_is_profile_with_largest_publication_count() -> None:
    authors = pd.DataFrame(
        [
            {
                "author_id": "10",
                "_id": "r1",
                "name": "Müller, Thomas",
                "surname": "Müller",
                "given_name": "Thomas",
                "indexed_name": "Müller T",
                "initials": "T",
                "document_count": 25,
                "publication_range": "",
                "name_variants": "[]",
                "subject_areas": "[]",
            },
            {
                "author_id": "20",
                "_id": "r2",
                "name": "Müller, Thomas",
                "surname": "Müller",
                "given_name": "Thomas",
                "indexed_name": "Müller T",
                "initials": "T",
                "document_count": 3,
                "publication_range": "",
                "name_variants": "[]",
                "subject_areas": "[]",
            },
        ]
    )
    decisions = pd.DataFrame(
        [{"left_author_id": "10", "right_author_id": "20", "label": "same_author"}]
    )
    clusters = cluster_authors(authors, decisions)
    assert clusters["disambiguated_cluster_id"].nunique() == 1
    assert clusters["disambiguated_cluster_id"].iloc[0] == "10"


def test_clustering_does_not_transitively_bridge_hard_gender_conflict() -> None:
    authors = pd.DataFrame(
        [
            {
                "author_id": "sandra",
                "name": "Müller, Sandra",
                "surname": "Müller",
                "given_name": "Sandra",
                "indexed_name": "Müller S.",
            },
            {
                "author_id": "josef",
                "name": "Müller, Josef S.",
                "surname": "Müller",
                "given_name": "Josef S.",
                "indexed_name": "Müller J.",
            },
            {
                "author_id": "initial",
                "name": "Müller, S.",
                "surname": "Müller",
                "given_name": "S.",
                "indexed_name": "Müller S.",
            },
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "left_author_id": "sandra",
                "right_author_id": "josef",
                "label": "distinct",
                "components": [
                    {
                        "label": "given_gender_conflict",
                        "weight": -100.0,
                        "reason": "reliable gender conflict",
                    }
                ],
            },
            {
                "left_author_id": "sandra",
                "right_author_id": "initial",
                "label": "same_author",
                "components": [],
            },
            {
                "left_author_id": "josef",
                "right_author_id": "initial",
                "label": "same_author",
                "components": [],
            },
        ]
    )
    clusters = cluster_authors(authors, decisions)
    members = clusters.groupby("cluster_id")["author_id"].apply(set).tolist()
    assert not any({"sandra", "josef"}.issubset(group) for group in members)
    assert clusters["cluster_id"].nunique() == 2


def test_clustering_does_not_bridge_hard_publication_conflicts() -> None:
    authors = pd.DataFrame(
        [
            {"author_id": author, "name": "Mayer, Otto", "surname": "Mayer", "given_name": "Otto"}
            for author in ["old", "middle", "modern"]
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "left_author_id": "old",
                "right_author_id": "middle",
                "label": "same_author",
                "components": [],
            },
            {
                "left_author_id": "middle",
                "right_author_id": "modern",
                "label": "same_author",
                "components": [],
            },
            {
                "left_author_id": "old",
                "right_author_id": "modern",
                "label": "distinct",
                "components": [
                    {
                        "label": "publication_range_gap_conflict",
                        "weight": 0.0,
                        "reason": "non-overlapping publication ranges have a 69-year gap",
                    }
                ],
            },
        ]
    )
    clusters = cluster_authors(authors, decisions)
    members = clusters.groupby("cluster_id")["author_id"].apply(set).tolist()
    assert not any({"old", "modern"}.issubset(group) for group in members)
    assert clusters["cluster_id"].nunique() == 2


def test_single_candidate_long_publication_span_requires_manual_review() -> None:
    candidates = pd.DataFrame(
        [
            {
                "sc_id": "1",
                "author_id": "10",
                "eid": "P1",
                "name": "Scherzer, Otmar",
                "surname": "Scherzer",
                "given_name": "Otmar",
                "indexed_name": "Scherzer O",
                "initials": "O",
                "publication_range": "{1949,2026}",
                "name_variants": "[]",
                "subject_areas": "[]",
            }
        ]
    )
    selection = disambiguate_batches(candidates)[0].final_selection
    assert selection.status == "manual_selection_required"
    assert selection.selected_author_ids == []
    assert "implausible publication span" in selection.reason
    assert "1949-2026" in selection.reason


def test_output_history_maps_preserve_input_history_and_metadata() -> None:
    candidates = pd.DataFrame(
        [
            {
                "sc_id": "1",
                "author_id": "57191070826",
                "eid": "P1",
                "researcher_name": "Müller, Thomas",
                "name": "Müller, Thomas",
                "surname": "Müller",
                "given_name": "Thomas",
                "indexed_name": "Müller T",
                "initials": "T",
                "publication_range": "",
                "name_variants": "[]",
                "subject_areas": "[]",
                "historical_identifier": "[57218689896, 57941965100]",
            },
            {
                "sc_id": "1",
                "author_id": "20",
                "eid": "P2",
                "researcher_name": "Müller, Thomas",
                "name": "Müller, Thomas",
                "surname": "Müller",
                "given_name": "Thomas",
                "indexed_name": "Müller T",
                "initials": "T",
                "publication_range": "",
                "name_variants": "[]",
                "subject_areas": "[]",
            },
        ]
    )
    results = disambiguate_batches(candidates)
    clusters = collect_clusters(results)
    pairs = collect_pair_decisions(results)
    selections = collect_final_selections(results)
    expected = {"57191070826": [57218689896, 57941965100]}
    assert pairs["left_history"].iloc[0] == expected
    assert pairs["right_history"].iloc[0] == {}
    assert clusters.set_index("author_id")["history"].to_dict() == {
        "20": {},
        "57191070826": expected,
    }
    assert selections["selected_history"].iloc[0] == {}
    assert selections["possible_cluster_history"].iloc[0] == [{}, expected]
    assert selections["researcher_name"].iloc[0] == "Müller, Thomas"
    scores = collect_final_selection_scores(results)
    assert scores.set_index("disambiguated_cluster_id")["history"].to_dict() == {
        "20": {},
        "57191070826": expected,
    }


def test_output_selection_carries_university_and_selected_affiliation_union() -> None:
    candidates = pd.DataFrame(
        [
            {
                "sc_id": "1",
                "author_id": "1",
                "eid": "P1",
                "name": "Müller, Thomas",
                "surname": "Müller",
                "given_name": "Thomas",
                "indexed_name": "Müller T",
                "initials": "T",
                "publication_range": "",
                "name_variants": "[]",
                "subject_areas": "[]",
                "university_name": "TU Wien",
                "affiliation_ids_current": "['A1', 'A2']",
            }
        ]
    )
    selection = collect_final_selections(disambiguate_batches(candidates))[
        "current_affiliation_ids"
    ].iloc[0]
    assert selection == ["A1", "A2"]
    assert collect_final_selections(disambiguate_batches(candidates))[
        "university_name"
    ].tolist() == ["TU Wien"]


def test_selected_affiliation_union_excludes_unselected_possible_cluster() -> None:
    authors = pd.DataFrame(
        [
            {"author_id": "1", "affiliation_ids_current": "['A1', 'A2']"},
            {"author_id": "2", "affiliation_id_current": "A1,A3"},
            {"author_id": "3", "affiliation_id_current": "A4"},
        ]
    )
    selection = FinalAuthorSelection(
        sc_id="1",
        status="assigned_single_cluster",
        selected_cluster_id=1,
        selected_disambiguated_cluster_id="1",
        selected_author_ids=["1", "2"],
        possible_clusters=[["1", "2"], ["3"]],
        inferred_asjc_codes=[],
        scopus_asjc_codes=[],
        inferred_asjc_labels=[],
        reason="test",
    )
    result = BatchDisambiguationResult(
        sc_id="1",
        authors=authors,
        decisions=pd.DataFrame(),
        clusters=pd.DataFrame(),
        final_selection=selection,
        final_selection_scores=pd.DataFrame(),
    )
    output = collect_final_selections([result])
    assert output["current_affiliation_ids"].iloc[0] == ["A1", "A2", "A3"]
