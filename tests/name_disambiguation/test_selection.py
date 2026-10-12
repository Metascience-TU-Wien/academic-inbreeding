# Copyright © 2026 TU Wien

import pandas as pd

from acib.name_disambiguation.parsing import author_from_row, parse_subject_areas
from acib.name_disambiguation.scoring import build_scoring_context, score_pair
from acib.name_disambiguation.service import (
    build_final_selection_scores,
    select_final_author,
)


def selection_inputs(
    records: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build candidate and singleton-cluster tables for selection rules.

    :param records: Candidate profile dictionaries.
    :return: Prepared input frames for the selection service.
    """
    authors = pd.DataFrame(records)
    if "subject_areas_parsed" not in authors:
        authors["subject_areas_parsed"] = [[] for _ in range(len(authors))]
    clusters = pd.DataFrame(
        [
            {
                "cluster_id": index,
                "author_id": str(row["author_id"]),
                "cluster_size": 1,
                "disambiguated_cluster_id": str(row["author_id"]),
            }
            for index, row in enumerate(records, start=1)
        ]
    )
    return authors, clusters


def candidate(
    author_id: str,
    *,
    name: str = "example, eva",
    given: str = "eva",
    codes: str = "",
    subjects: str = "[]",
    **extra: object,
) -> dict[str, object]:
    """Build one concise researcher-author input record.

    :param author_id: Synthetic Scopus author ID.
    :param name: Researcher full name.
    :param given: Candidate author given name.
    :param codes: Inferred ASJC codes.
    :param subjects: Serialized candidate subject areas.
    :param extra: Additional evidence columns.
    :return: Input record consumed by selection tests.
    """
    return {
        "author_id": author_id,
        "researcher_name": name,
        "canonical_given_name": given,
        "inferred_asjc_codes": codes,
        "subject_areas_parsed": parse_subject_areas(subjects),
        **extra,
    }


def test_parent_asjc_matching_retains_parent_coverage_and_code_alignment() -> None:
    authors, clusters = selection_inputs(
        [
            candidate(
                "a1",
                codes="2700,3003",
                subjects=(
                    '[{"area":"Medicine","code":2700},'
                    '{"area":"Anesthesiology and Pain Medicine","code":2703}]'
                ),
            ),
            candidate("a2", codes="2700,3003", subjects='[{"area":"Chemistry","code":1600}]'),
        ]
    )
    selection, scores = select_final_author(authors, clusters, sc_id="1")
    assert selection.status == "assigned_by_profile_asjc"
    assert selection.selected_author_ids == ["a1"]
    assert selection.scopus_asjc_codes == [2700]
    assert scores["inferred_code_overlap_codes"].tolist() == [[2700], []]


def test_profile_parent_asjc_tie_remains_manual_and_preserves_possible_codes() -> None:
    authors, clusters = selection_inputs(
        [
            candidate(
                "a1",
                codes="2700,2705",
                subjects='[{"area":"Cardiology and Cardiovascular Medicine","code":2705}]',
            ),
            candidate(
                "a2",
                codes="2700,2705",
                subjects=(
                    '[{"area":"General Medicine","code":2700},'
                    '{"area":"Anesthesiology and Pain Medicine","code":2703},'
                    '{"area":"Cardiology and Cardiovascular Medicine","code":2705},'
                    '{"area":"Emergency Medicine","code":2711},'
                    '{"area":"Surgery","code":2746}]'
                ),
            ),
        ]
    )
    selection, scores = select_final_author(authors, clusters, sc_id="16849")
    assert selection.status == "manual_selection_required"
    assert selection.selected_author_ids == []
    assert scores["inferred_code_coverage"].tolist() == [1.0, 1.0]
    assert scores["cluster_subject_count"].tolist() == [1, 1]


def test_reliable_researcher_gender_filters_incompatible_asjc_candidate() -> None:
    authors, clusters = selection_inputs(
        [
            candidate(
                "16422030900",
                name="muller, sandra",
                given="josef s",
                codes="2600",
                subjects='[{"area":"General Mathematics","code":2600}]',
            ),
            candidate(
                "57211463559",
                name="muller, sandra",
                given="sandra",
                codes="2600",
                subjects='[{"area":"General Mathematics","code":2600}]',
            ),
            candidate(
                "57217827401",
                name="muller, sandra",
                given="sandra",
                codes="2600",
                subjects='[{"area":"General Medicine","code":2700}]',
            ),
        ]
    )
    selection, scores = select_final_author(authors, clusters, sc_id="68920")
    assert selection.status == "assigned_by_profile_asjc"
    assert selection.selected_author_ids == ["57211463559"]
    assert selection.scopus_asjc_codes == [2600]
    male_math = scores.loc[scores["author_ids"].map(lambda ids: ids == ["16422030900"])].iloc[0]
    assert not male_math["gender_compatible"]


def test_current_affiliation_breaks_candidate_cluster_tie() -> None:
    authors, clusters = selection_inputs(
        [
            candidate(
                "a1",
                name="zauner, gerald",
                given="gerald",
                subjects='[{"area":"Electrical Engineering","code":2208}]',
                university_name="FH Oberösterreich",
                affiliation_id_current="101928095.0",
            ),
            candidate(
                "a2",
                name="zauner, gerald",
                given="gerald",
                subjects='[{"area":"Electrical Engineering","code":2208}]',
                university_name="FH Oberösterreich",
                affiliation_id_current="60013500.0",
            ),
        ]
    )
    selection, scores = select_final_author(authors, clusters, sc_id="11213")
    assert selection.status == "assigned_by_current_affiliation"
    assert selection.selected_author_ids == ["a1"]
    selected = scores.loc[scores["author_ids"].map(lambda ids: ids == ["a1"])].iloc[0]
    assert selected["university_affiliation_ids"] == [
        "101928095",
        "114521147",
        "119146064",
        "60032193",
    ]
    assert selected["cluster_current_affiliation_ids"] == ["101928095"]
    assert selected["current_affiliation_overlap_ids"] == ["101928095"]
    assert selected["current_affiliation_overlap_count"] == 1
    other = scores.loc[scores["author_ids"].map(lambda ids: ids == ["a2"])].iloc[0]
    assert other["current_affiliation_overlap_count"] == 0


def test_researcher_given_name_selects_full_name_and_tie_stays_manual() -> None:
    authors, clusters = selection_inputs(
        [
            candidate(
                "a1",
                name="leitner, karl heinz",
                given="karl heinz",
                subjects='[{"area":"General Medicine","code":2700}]',
            ),
            candidate(
                "a2",
                name="leitner, karl heinz",
                given="h",
                subjects='[{"area":"General Medicine","code":2700}]',
            ),
        ]
    )
    selection, scores = select_final_author(authors, clusters, sc_id="27831")
    assert selection.status == "assigned_by_researcher_given_name"
    assert selection.selected_author_ids == ["a1"]
    assert scores["researcher_given_name_overlap_count"].tolist() == [2, 0]

    tied, tied_clusters = selection_inputs(
        [
            candidate(
                "b1",
                name="leitner, karl heinz",
                given="karl",
                subjects='[{"area":"General Medicine","code":2700}]',
            ),
            candidate(
                "b2",
                name="leitner, karl heinz",
                given="heinz",
                subjects='[{"area":"General Medicine","code":2700}]',
            ),
        ]
    )
    tie_selection, tie_scores = select_final_author(tied, tied_clusters, sc_id="27831")
    assert tie_selection.status == "manual_selection_required"
    assert tie_scores["researcher_given_name_overlap_count"].tolist() == [1, 1]


def test_asjc_beats_given_name_but_partial_parent_coverage_is_not_decisive() -> None:
    authors, clusters = selection_inputs(
        [
            candidate(
                "a1",
                name="leitner, karl heinz",
                given="karl heinz",
                codes="2700",
                subjects='[{"area":"Computer Science Applications","code":1706}]',
            ),
            candidate(
                "a2",
                name="leitner, karl heinz",
                given="h",
                codes="2700",
                subjects='[{"area":"General Medicine","code":2700}]',
            ),
        ]
    )
    selection, _ = select_final_author(authors, clusters, sc_id="27831")
    assert selection.status == "assigned_by_profile_asjc"
    assert selection.selected_author_ids == ["a2"]

    partial, partial_clusters = selection_inputs(
        [
            candidate(
                "p1",
                codes="2700,3003",
                subjects='[{"area":"Medicine (miscellaneous)","code":2701}]',
            ),
            candidate(
                "p2",
                codes="2700,3003",
                subjects=(
                    '[{"area":"General Medicine","code":2700},'
                    '{"area":"Anesthesiology and Pain Medicine","code":2703}]'
                ),
            ),
        ]
    )
    unresolved, partial_scores = select_final_author(partial, partial_clusters, sc_id="16811")
    assert unresolved.status == "manual_selection_required"
    assert partial_scores["inferred_code_coverage"].tolist() == [0.5, 0.5]


def test_exact_profile_asjc_breaks_parent_tie_but_sibling_child_does_not() -> None:
    authors, clusters = selection_inputs(
        [
            candidate(
                "a1",
                name="martinek, johannes",
                given="johannes",
                codes="2204,2700",
                subjects=(
                    '[{"area":"Biomedical Engineering","code":2204},'
                    '{"area":"General Medicine","code":2700}]'
                ),
            ),
            candidate(
                "a2",
                name="martinek, johannes",
                given="johannes",
                codes="2204,2700",
                subjects=(
                    '[{"area":"Civil and Structural Engineering","code":2207},'
                    '{"area":"Computational Mechanics","code":2208},'
                    '{"area":"Radiology","code":2742}]'
                ),
            ),
        ]
    )
    selection, scores = select_final_author(authors, clusters, sc_id="12690")
    assert selection.status == "assigned_by_profile_asjc"
    assert selection.selected_author_ids == ["a1"]
    selected = scores.loc[scores["author_ids"].map(lambda ids: ids == ["a1"])].iloc[0]
    other = scores.loc[scores["author_ids"].map(lambda ids: ids == ["a2"])].iloc[0]
    assert selected["overlap_codes"] == [2200, 2700]
    assert other["overlap_codes"] == [2200, 2700]
    assert selected["exact_overlap_codes"] == [2204, 2700]
    assert other["exact_overlap_codes"] == [2700]

    sibling, sibling_clusters = selection_inputs(
        [
            candidate(
                "s1",
                name="hormandinger, karl",
                given="karl",
                codes="2700,2706",
                subjects='[{"area":"Anesthesiology and Pain Medicine","code":2703}]',
            ),
            candidate(
                "s2",
                name="hormandinger, karl",
                given="karl",
                codes="2700,2706",
                subjects='[{"area":"Surgery","code":2746}]',
            ),
        ]
    )
    sibling_selection, sibling_scores = select_final_author(
        sibling, sibling_clusters, sc_id="17615"
    )
    assert sibling_selection.status == "manual_selection_required"
    assert sibling_scores["inferred_code_overlap_codes"].tolist() == [[2700], [2700]]


def test_parent_overlap_can_assign_unique_profile_and_maps_specific_child_to_parent() -> None:
    authors, clusters = selection_inputs(
        [
            candidate(
                "56614799600",
                name="zauner, michal",
                given="m",
                codes="2100,2207,2209",
                subjects='[{"area":"Modeling and Simulation","code":2611}]',
            ),
            candidate(
                "57224912910",
                name="zauner, michal",
                given="michal",
                codes="2100,2207,2209",
                subjects=(
                    '[{"area":"General Computer Science","code":1700},'
                    '{"area":"Electrical and Electronic Engineering","code":2208},'
                    '{"area":"Modeling and Simulation","code":2611}]'
                ),
            ),
        ]
    )
    selection, scores = select_final_author(authors, clusters, sc_id="10835")
    assert selection.status == "assigned_by_profile_asjc"
    assert selection.selected_author_ids == ["57224912910"]
    selected = scores.loc[scores["author_ids"].map(lambda ids: ids == ["57224912910"])].iloc[0]
    assert selected["inferred_code_overlap_count"] == 0
    assert selected["overlap_codes"] == [2200]

    parent, parent_clusters = selection_inputs(
        [candidate("a1", codes="2700", subjects='[{"area":"Radiology","code":2741}]')]
    )
    parent_scores = build_final_selection_scores(parent, parent_clusters, inferred_codes=[2700])
    assert parent_scores.iloc[0]["cluster_level_two_subject_codes"] == [2700]
    assert parent_scores.iloc[0]["inferred_code_overlap_codes"] == [2700]
    assert parent_scores.iloc[0]["inferred_code_overlap_count"] == 1
    assert parent_scores.iloc[0]["inferred_code_coverage"] == 1.0
    assert parent_scores.iloc[0]["exact_specific_overlap_count"] == 0


def test_manual_selection_preserves_scopus_codes_aligned_with_possible_clusters() -> None:
    authors, clusters = selection_inputs(
        [
            candidate(
                "a1",
                codes="2700,3003",
                subjects=(
                    '[{"area":"Medicine","code":2700},'
                    '{"area":"Anesthesiology and Pain Medicine","code":2703},'
                    '{"area":"Health Policy","code":2719}]'
                ),
            ),
            candidate(
                "a2",
                codes="2700,3003",
                subjects=(
                    '[{"area":"Medicine","code":2700},'
                    '{"area":"Anesthesiology and Pain Medicine","code":2703}]'
                ),
            ),
        ]
    )
    selection, _ = select_final_author(authors, clusters, sc_id="16811")
    assert selection.status == "manual_selection_required"
    assert selection.possible_clusters == [["a1"], ["a2"]]
    assert selection.scopus_asjc_codes == [[2700], [2700]]


def test_specific_child_asjc_breaks_parent_tie_and_partial_coverage_does_not() -> None:
    authors, clusters = selection_inputs(
        [
            candidate("a1", codes="2700,2741", subjects='[{"area":"Radiology","code":2741}]'),
            candidate(
                "a2", codes="2700,2741", subjects='[{"area":"General Medicine","code":2700}]'
            ),
        ]
    )
    selection, scores = select_final_author(authors, clusters, sc_id="1")
    assert selection.status == "assigned_by_profile_asjc"
    assert selection.selected_author_ids == ["a1"]
    selected = scores.loc[scores["author_ids"].map(lambda ids: ids == ["a1"])].iloc[0]
    other = scores.loc[scores["author_ids"].map(lambda ids: ids == ["a2"])].iloc[0]
    assert selected["inferred_code_overlap_codes"] == [2700, 2741]
    assert other["inferred_code_overlap_codes"] == [2700, 2741]
    assert selected["exact_specific_overlap_codes"] == [2741]
    assert other["exact_specific_overlap_codes"] == []


def test_single_cluster_assignment_and_possible_cluster_scopus_codes() -> None:
    one_author, one_cluster = selection_inputs(
        [
            candidate(
                "a1",
                codes="2700,3003",
                subjects=(
                    '[{"area":"Medicine","code":2700},'
                    '{"area":"Anesthesiology and Pain Medicine","code":2703}]'
                ),
            )
        ]
    )
    assigned, _ = select_final_author(one_author, one_cluster, sc_id="1")
    assert assigned.status == "assigned_single_cluster"
    assert assigned.selected_author_ids == ["a1"]
    assert assigned.scopus_asjc_codes == [2700]

    authors, clusters = selection_inputs(
        [
            candidate(
                "57082685500",
                name="selg, manul",
                given="manuel",
                subjects='[{"area":"Plant Science","code":1110}]',
            ),
            candidate(
                "57217943868",
                name="selg, manul",
                given="manuel",
                subjects='[{"area":"Arts and Humanities (all)","code":1200}]',
            ),
        ]
    )
    manual, _ = select_final_author(authors, clusters, sc_id="11210")
    assert manual.status == "manual_selection_required"
    assert manual.scopus_asjc_codes == [[1100], [1200]]


def test_final_selection_score_parent_categories_and_coverage_are_exact() -> None:
    authors = pd.DataFrame(
        [
            {
                "author_id": "a1",
                "subject_areas_parsed": parse_subject_areas(
                    '[{"area":"Applied Mathematics","code":2604}]'
                ),
            },
            {
                "author_id": "a2",
                "subject_areas_parsed": parse_subject_areas(
                    '[{"area":"Statistics and Probability","code":2613}]'
                ),
            },
            {
                "author_id": "b1",
                "subject_areas_parsed": parse_subject_areas('[{"area":"Virology","code":2406}]'),
            },
        ]
    )
    clusters = pd.DataFrame(
        [
            {
                "cluster_id": 1,
                "author_id": "a1",
                "cluster_size": 2,
                "disambiguated_cluster_id": "a1",
            },
            {
                "cluster_id": 1,
                "author_id": "a2",
                "cluster_size": 2,
                "disambiguated_cluster_id": "a1",
            },
            {
                "cluster_id": 2,
                "author_id": "b1",
                "cluster_size": 1,
                "disambiguated_cluster_id": "b1",
            },
        ]
    )
    scores = build_final_selection_scores(authors, clusters, inferred_codes=[2604, 2403])
    first = scores.loc[scores["cluster_id"] == 1].iloc[0]
    second = scores.loc[scores["cluster_id"] == 2].iloc[0]
    assert first["cluster_level_two_subject_codes"] == [2600]
    assert first["overlap_codes"] == [2600]
    assert first["overlap_count"] == 1
    assert first["inferred_coverage"] == 0.5
    assert second["cluster_level_two_subject_codes"] == [2400]
    assert second["overlap_codes"] == [2400]
    assert second["overlap_count"] == 1


def test_article_parent_subject_overlap_component_has_source_weight() -> None:
    left = author_from_row(
        {
            "author_id": "1",
            "name": "Müller, Thomas",
            "surname": "Müller",
            "given_name": "Thomas",
            "initials": "T",
            "indexed_name": "Müller T",
            "name_variants": "[]",
            "subject_areas": "[]",
        }
    )
    right = author_from_row(
        {
            "author_id": "2",
            "name": "Müller, Thomas",
            "surname": "Müller",
            "given_name": "Thomas",
            "initials": "T",
            "indexed_name": "Müller T",
            "name_variants": "[]",
            "subject_areas": "[]",
        }
    )
    articles = pd.DataFrame(
        [
            {"author_ids": "1", "eid": "A1", "subject_areas": '[{"area":"Marketing","code":1406}]'},
            {"author_ids": "2", "eid": "B1", "subject_areas": '[{"area":"Business","code":1400}]'},
        ]
    )
    decision = score_pair(left, right, context=build_scoring_context(articles))
    component = next(
        item for item in decision.components if item.label == "article_subject_overlap"
    )
    assert component.weight == 3.0
    assert "profile_subject_overlap" not in {item.label for item in decision.components}
    assert "subject_conflict" not in {item.label for item in decision.components}
