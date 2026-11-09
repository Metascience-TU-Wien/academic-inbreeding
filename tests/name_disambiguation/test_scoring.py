# Copyright © 2026 TU Wien

import pandas as pd

from acib.common.names import compound_match
from acib.name_disambiguation.models import (
    AuthorRecord,
    PairwiseDecision,
    PublicationRange,
)
from acib.name_disambiguation.parsing import author_from_row, parse_subject_areas
from acib.name_disambiguation.scoring import build_scoring_context, score_pair
from acib.name_disambiguation.scoring_common import surname_compatible


def make_author(
    author_id: str,
    given_name: str,
    initials: str,
    *,
    orcid: str = "",
    publication_range: PublicationRange | None = None,
    affiliation: str = "",
) -> AuthorRecord:
    """Create a compact parsed profile for pair-scoring tests.

    :param author_id: Synthetic Scopus author ID.
    :param given_name: Given-name form.
    :param initials: Initials form.
    :param orcid: Optional ORCID anchor.
    :param publication_range: Optional temporal evidence.
    :param affiliation: Optional current affiliation value.
    :return: Parsed author record.
    """
    return author_from_row(
        {
            "_id": author_id,
            "author_id": author_id,
            "name": f"Müller, {given_name}",
            "surname": "Müller",
            "given_name": given_name,
            "indexed_name": "Müller T",
            "initials": initials,
            "orcid": orcid,
            "publication_range": (
                f"{{{publication_range.start},{publication_range.end}}}"
                if publication_range and publication_range.start and publication_range.end
                else ""
            ),
            "affiliation_id_current": affiliation,
            "name_variants": "[]",
            "subject_areas": "[]",
        }
    )


def component_labels(decision: PairwiseDecision) -> set[str]:
    """Return evidence labels emitted for one scored pair.

    :param decision: Pairwise scoring result.
    :return: Set of evidence labels.
    """
    return {component.label for component in decision.components}


def test_orcid_is_a_strong_anchor_and_overrides_temporal_conflict() -> None:
    left = make_author(
        "1",
        "Otto",
        "O",
        orcid="0000-0001-2345-6789",
        publication_range=PublicationRange(1907, 1942),
    )
    right = make_author(
        "2",
        "Otto",
        "O",
        orcid="0000-0001-2345-6789",
        publication_range=PublicationRange(2011, 2026),
    )
    decision = score_pair(left, right)
    assert decision.label == "same_author"
    assert {"orcid_match", "publication_range_gap_conflict"} <= component_labels(decision)


def test_temporal_conflict_blocks_match_without_strong_anchor() -> None:
    left = make_author("1", "Otto", "O", publication_range=PublicationRange(1907, 1942))
    right = make_author("2", "Otto", "O", publication_range=PublicationRange(2011, 2026))
    decision = score_pair(left, right)
    assert decision.label == "distinct"
    assert "publication_range_gap_conflict" in component_labels(decision)


def test_publication_gap_and_span_thresholds_are_distinct_hard_conflicts() -> None:
    left = make_author("1", "Otto", "O", publication_range=PublicationRange(1907, 1942))
    right = make_author("2", "Otto", "O", publication_range=PublicationRange(2011, 2026))
    decision = score_pair(left, right)
    assert decision.label == "distinct"
    assert {
        "publication_range_gap_conflict",
        "publication_range_span_conflict",
    } <= component_labels(decision)

    left = make_author("3", "Otto", "O", publication_range=PublicationRange(1960, 1970))
    right = make_author("4", "Otto", "O", publication_range=PublicationRange(1996, 2023))
    decision = score_pair(left, right)
    assert decision.label == "distinct"
    assert "publication_range_gap_conflict" in component_labels(decision)
    assert "publication_range_span_conflict" not in component_labels(decision)

    left = make_author("5", "Otto", "O", publication_range=PublicationRange(1950, 2025))
    right = make_author("6", "O.", "O", publication_range=PublicationRange(1960, 1970))
    decision = score_pair(left, right, same_author_score_threshold=3.0)
    assert decision.label == "distinct"
    assert "publication_range_span_conflict" in component_labels(decision)
    assert "publication_range_gap_conflict" not in component_labels(decision)


def test_subject_parent_overlap_is_scientifically_scored() -> None:
    left = make_author("1", "Thomas", "T")
    right = make_author("2", "Thomas", "T")
    left.subject_areas = parse_subject_areas('[{"area":"Software","code":1712}]')
    right.subject_areas = parse_subject_areas('[{"area":"Artificial Intelligence","code":1702}]')
    decision = score_pair(left, right)
    overlap = next(item for item in decision.components if item.label == "profile_subject_overlap")
    assert overlap.weight == 1.5
    assert "article_subject_overlap" not in component_labels(decision)
    assert "subject_conflict" not in component_labels(decision)


def test_external_citations_count_only_third_party_shared_citers() -> None:
    left = make_author("1", "Thomas", "TJ")
    right = make_author("2", "Thomas", "TJ")
    articles = pd.DataFrame({"author_ids": ["1", "2"], "eid": ["A", "B"]})
    references = pd.DataFrame(
        {
            "citing_eid": ["X", "X", "Y", "Y", "A"],
            "cited_eid": ["A", "B", "A", "B", "B"],
        }
    )
    decision = score_pair(
        left,
        right,
        context=build_scoring_context(
            [left, right], article_author_df=articles, article_reference_df=references
        ),
    )
    cocitation = next(item for item in decision.components if item.label == "co_citation")
    assert cocitation.weight == 3.0
    assert cocitation.reason == "shared citing papers: 2"


def test_given_name_and_initial_conflicts_block_high_contextual_overlap() -> None:
    left = make_author("1", "Christian", "C")
    right = make_author("2", "Peter R.", "PR")
    context = build_scoring_context(
        article_author_df=pd.DataFrame(
            [
                {
                    "eid": "A",
                    "author_ids": "1;3;4",
                    "author_names": "Müller, Christian;A;B",
                    "source_title": "Nature",
                    "country": "Germany",
                    "city": "Vienna",
                },
                {
                    "eid": "B",
                    "author_ids": "2;3;4",
                    "author_names": "Mueller, Peter R.;A;B",
                    "source_title": "Nature",
                    "country": "Germany",
                    "city": "Vienna",
                },
            ]
        )
    )
    decision = score_pair(left, right, context=context)
    labels = component_labels(decision)
    assert decision.label == "distinct"
    assert "given_conflict_with_initials_conflict" in labels
    assert "shared_coauthors" in labels


def test_reliable_gender_conflict_with_shared_initial_uses_overlap_component() -> None:
    left = make_author("1", "Thomas", "T")
    right = make_author("2", "Theresa", "T")
    articles = pd.DataFrame(
        [
            {
                "author_ids": "1;3",
                "eid": "A1",
                "source_title": "Nature",
                "country": "AUT",
                "city": "Vienna",
            },
            {
                "author_ids": "2;3",
                "eid": "B1",
                "source_title": "Nature",
                "country": "AUT",
                "city": "Vienna",
            },
        ]
    )

    decision = score_pair(left, right, context=build_scoring_context(articles))
    labels = component_labels(decision)

    assert decision.label == "distinct"
    assert "given_gender_conflict" in labels
    assert "given_conflict_with_initial_overlap" in labels
    assert "given_conflict_with_initials_conflict" not in labels


def test_ernst_gudrun_and_manuela_wolfgang_gender_conflicts_block_same_author() -> None:
    for left_name, right_name in [("Ernst", "Gudrun E."), ("Manuela", "Wolfgang M.")]:
        decision = score_pair(
            make_author(
                "1", left_name, "E" if left_name == "Ernst" else "M", affiliation="60018163"
            ),
            make_author(
                "2", right_name, "GE" if left_name == "Ernst" else "WM", affiliation="60018163"
            ),
        )
        labels = component_labels(decision)
        assert decision.label == "distinct"
        assert "given_gender_conflict" in labels


def test_overlapping_initial_conflict_is_capped_but_orcid_overrides() -> None:
    left = make_author("1", "Christian", "C")
    right = make_author("2", "Dennis C.", "DC")
    context = build_scoring_context(
        article_author_df=pd.DataFrame(
            [
                {
                    "eid": "A",
                    "author_ids": "1",
                    "subject_areas": '[{"area":"Medicine","code":2700}]',
                    "publication_name": "Shared Journal",
                    "country": "Austria",
                    "city": "Vienna",
                },
                {
                    "eid": "B",
                    "author_ids": "2",
                    "subject_areas": '[{"area":"Medicine","code":2700}]',
                    "publication_name": "Shared Journal",
                    "country": "Austria",
                    "city": "Vienna",
                },
            ]
        )
    )
    decision = score_pair(left, right, context=context)
    assert decision.raw_score >= 5.0
    assert decision.label == "possible_same_author"
    conflict = next(
        item for item in decision.components if item.label == "given_conflict_with_initial_overlap"
    )
    assert conflict.weight == -3.0
    anchored = score_pair(
        make_author("3", "Christian", "C", orcid="0000-0001-2345-6789"),
        make_author("4", "Dennis C.", "DC", orcid="0000-0001-2345-6789"),
    )
    assert anchored.label == "same_author"
    assert "orcid_match" in component_labels(anchored)


def test_initial_only_later_and_first_name_cases_keep_source_thresholds() -> None:
    left = make_author("1", "Karl Heinz", "KH", affiliation="60000001")
    right = make_author("2", "H.", "H", affiliation="60000001")
    articles = pd.DataFrame(
        [
            {
                "eid": "A",
                "author_ids": "1;3",
                "subject_areas": '[{"area":"Medicine","code":2700}]',
                "country": "AUT",
                "city": "Vienna",
            },
            {
                "eid": "B",
                "author_ids": "2;4",
                "subject_areas": '[{"area":"Medicine","code":2700}]',
                "country": "AUT",
                "city": "Vienna",
            },
        ]
    )
    context = build_scoring_context(article_author_df=articles)
    later = score_pair(left, right, context=context)
    assert later.raw_score == 9.0
    assert later.label == "possible_same_author"
    assert "initial_only_later_given_initial" in component_labels(later)

    first_left = make_author("3", "Michael", "M", affiliation="60000001")
    first_right = make_author("4", "M.", "M", affiliation="60000001")
    first_context = build_scoring_context(article_author_df=articles)
    first = score_pair(first_left, first_right, context=first_context)
    assert first.raw_score >= 6.0
    assert first.label == "possible_same_author"
    assert "initial_only_given_name_match" in component_labels(first)

    anchored = score_pair(
        make_author("5", "Karl Heinz", "KH", orcid="0000-0001-2345-6789"),
        make_author("6", "H.", "H", orcid="0000-0001-2345-6789"),
    )
    assert anchored.label == "same_author"
    assert {"initial_only_later_given_initial", "orcid_match"} <= component_labels(anchored)


def test_full_given_name_initial_compatibility_and_exact_threshold() -> None:
    compatible = score_pair(make_author("1", "Christian", "C"), make_author("2", "C.", "C"))
    assert "given_compatible" in component_labels(compatible)
    assert not any(label.startswith("given_conflict") for label in component_labels(compatible))
    assert "initials_conflict" not in component_labels(compatible)

    left = make_author("3", "Rudolf", "R")
    right = make_author("4", "R.", "R")
    context = build_scoring_context(
        article_author_df=pd.DataFrame(
            [
                {
                    "eid": "A",
                    "author_ids": "3",
                    "subject_areas": '[{"area":"Physical and Theoretical Chemistry","code":1606}]',
                },
                {
                    "eid": "B",
                    "author_ids": "4",
                    "subject_areas": '[{"area":"Physical and Theoretical Chemistry","code":1606}]',
                },
            ]
        )
    )
    default = score_pair(left, right, context=context)
    lowered = score_pair(left, right, context=context, same_author_score_threshold=3.0)
    assert default.raw_score == 3.0
    assert default.label == "possible_same_author"
    # The source implementation's initial-only risk cap takes precedence over
    # a lowered same-author threshold for this pair.
    assert lowered.label == "possible_same_author"


def test_affiliation_match_normalization_and_missing_sentinels() -> None:
    decision = score_pair(
        make_author("1", "Gerald", "G", affiliation="101928095.0"),
        make_author("2", "G.", "G", affiliation="['101928095', '60013500']"),
    )
    match = next(
        item for item in decision.components if item.label == "current_affiliation_id_match"
    )
    assert match.weight == 4.0
    assert decision.raw_score == 4.0
    assert decision.label == "possible_same_author"
    no_match = score_pair(
        make_author("3", "Gerald", "G", affiliation="-9999"),
        make_author("4", "G.", "G", affiliation=""),
    )
    assert "current_affiliation_id_match" not in component_labels(no_match)
    missing = make_author("5", "Gerald", "G")
    missing.raw = {"affiliation_id_current": pd.NA}
    also_missing = make_author("6", "G.", "G")
    also_missing.raw = {"affiliation_id_current": pd.NA}
    assert "current_affiliation_id_match" not in component_labels(score_pair(missing, also_missing))


def test_transliteration_compound_surnames_and_transposed_fields() -> None:
    transliterated = score_pair(
        make_author("1", "Thomas", "T"),
        author_from_row(
            {
                "author_id": "2",
                "name": "Mueller, T.",
                "surname": "Mueller",
                "given_name": "T.",
                "initials": "T",
                "indexed_name": "Mueller T",
                "name_variants": "[]",
                "subject_areas": "[]",
            }
        ),
    )
    assert "surname_exact" in component_labels(transliterated)
    assert "surname_conflict" not in component_labels(transliterated)
    compound = score_pair(
        author_from_row(
            {
                "author_id": "7",
                "name": "Mueller-Lynch, Thomas",
                "surname": "Mueller-Lynch",
                "given_name": "Thomas",
                "initials": "T",
                "indexed_name": "Mueller-Lynch T",
                "name_variants": "[]",
                "subject_areas": "[]",
            }
        ),
        author_from_row(
            {
                "author_id": "8",
                "name": "Müller, Thomas",
                "surname": "Müller",
                "given_name": "Thomas",
                "initials": "T",
                "indexed_name": "Müller T",
                "name_variants": "[]",
                "subject_areas": "[]",
            }
        ),
    )
    assert "surname_compatible" in component_labels(compound)
    assert "surname_conflict" not in component_labels(compound)
    assert surname_compatible("windisch gratz", "windischgratz")
    assert surname_compatible("windisch-graetz", "windischgraetz")
    assert not surname_compatible("windisch gratz", "windischgarten")
    assert not surname_compatible("mayer", "mayr")
    assert compound_match("windisch gratz, michala", "windischgratz, michala")

    transposed = score_pair(
        author_from_row(
            {
                "author_id": "5",
                "name": "Bednar, Thomas",
                "surname": "Bednar",
                "given_name": "Thomas",
                "initials": "T",
                "indexed_name": "Bednar T",
                "name_variants": "[]",
                "subject_areas": "[]",
            }
        ),
        author_from_row(
            {
                "author_id": "6",
                "name": "Thomas, Bednar",
                "surname": "Thomas",
                "given_name": "Bednar",
                "initials": "B",
                "indexed_name": "Thomas B",
                "name_variants": "[]",
                "subject_areas": "[]",
            }
        ),
    )
    assert "transposed_name_fields" in component_labels(transposed)
    assert not any(label.startswith("given_conflict") for label in component_labels(transposed))
    assert "initials_conflict" not in component_labels(transposed)
    assert transposed.label == "distinct"

    compatible_transposed = score_pair(
        author_from_row(
            {
                "author_id": "9",
                "name": "Leutgeb-Schuster, Elisabeth",
                "surname": "Leutgeb-Schuster",
                "given_name": "Elisabeth",
                "initials": "E",
                "indexed_name": "Leutgeb-Schuster E",
                "name_variants": "[]",
                "subject_areas": "[]",
            }
        ),
        author_from_row(
            {
                "author_id": "10",
                "name": "Elisabeth, Leutgeb Schuster",
                "surname": "Elisabeth",
                "given_name": "Leutgeb Schuster",
                "initials": "L",
                "indexed_name": "Elisabeth L",
                "name_variants": "[]",
                "subject_areas": "[]",
            }
        ),
    )
    assert "transposed_name_fields" in component_labels(compatible_transposed)
    assert not any(
        label.startswith("given_conflict") for label in component_labels(compatible_transposed)
    )
    assert "initials_conflict" not in component_labels(compatible_transposed)
    assert compatible_transposed.label == "distinct"


def test_fused_compound_surname_matches_and_single_shared_initial_is_neutral() -> None:
    left = author_from_row(
        {
            "author_id": "1",
            "name": "Windisch Gratz, Michaela",
            "surname": "Windisch Gratz",
            "given_name": "Michaela",
            "initials": "M",
            "indexed_name": "Windisch Gratz M",
            "name_variants": "[]",
            "subject_areas": "[]",
        }
    )
    right = author_from_row(
        {
            "author_id": "2",
            "name": "Windischgraetz, Michaela",
            "surname": "Windischgraetz",
            "given_name": "Michaela",
            "initials": "M",
            "indexed_name": "Windischgraetz M",
            "name_variants": "[]",
            "subject_areas": "[]",
        }
    )
    labels = component_labels(score_pair(left, right))
    assert "surname_compatible" in labels
    assert "surname_conflict" not in labels

    josef = make_author("3", "Josef S.", "JS")
    sandra = make_author("4", "Sandra", "S")
    decision = score_pair(josef, sandra)
    assert "shared_initials_2" not in component_labels(decision)
    assert "shared_initials_gt2" not in component_labels(decision)
    assert "initials_conflict" not in component_labels(decision)
    assert "given_gender_conflict" in component_labels(decision)
    assert decision.label == "distinct"


def test_subject_code_missing_broad_and_parent_scoring_rules() -> None:
    left = make_author("1", "Thomas", "T")
    right = make_author("2", "Thomas", "T")
    left.subject_areas = parse_subject_areas('[{"area":"Unknown","code":0}]')
    right.subject_areas = parse_subject_areas('[{"area":"Unknown","code":3104}]')
    assert "subject_conflict" not in component_labels(score_pair(left, right))

    left.subject_areas = parse_subject_areas('[{"area":"Computer Science","code":17}]')
    right.subject_areas = parse_subject_areas('[{"area":"Software","code":1712}]')
    labels = component_labels(score_pair(left, right))
    assert "article_subject_overlap" not in labels
    assert "profile_subject_overlap" not in labels
    assert "subject_conflict" not in labels


def test_frequency_scoring_uses_corpus_counts() -> None:
    common_left = make_author("1", "Thomas", "T")
    common_right = make_author("2", "Thomas", "T")
    rare_left = make_author("3", "Xaver", "X")
    rare_right = make_author("4", "Xaver", "X")
    frequency = pd.DataFrame(
        {"given_name": ["Thomas"] * 20 + [f"Rare{i}" for i in range(20)] + ["Xaver"]}
    )
    common = score_pair(
        common_left,
        common_right,
        context=build_scoring_context([common_left, common_right], name_frequency_df=frequency),
    )
    rare = score_pair(
        rare_left,
        rare_right,
        context=build_scoring_context([rare_left, rare_right], name_frequency_df=frequency),
    )
    assert "shared_first_name_general" in component_labels(common)
    assert "shared_first_name_nongeneral" in component_labels(rare)


def test_citation_components_include_self_citation_coupling_and_cocitation() -> None:
    left = make_author("1", "Thomas", "TJ")
    right = make_author("2", "Thomas", "TJ")
    articles = pd.DataFrame(
        [
            {"author_ids": "1", "eid": "A1", "cited_eid": "B1"},
            {"author_ids": "1", "eid": "A1", "cited_eid": "R1"},
            {"author_ids": "2", "eid": "B1", "cited_eid": "R1"},
            {"author_ids": "3", "eid": "C1", "cited_eid": "A1"},
            {"author_ids": "3", "eid": "C1", "cited_eid": "B1"},
        ]
    )
    decision = score_pair(left, right, context=build_scoring_context(articles))
    assert {"self_citation", "bibliographic_coupling", "co_citation"} <= component_labels(decision)
    assert decision.label == "same_author"


def test_external_reference_edges_are_linked_after_candidate_filtering() -> None:
    left = make_author("1", "Thomas", "TJ")
    right = make_author("2", "Thomas", "TJ")
    articles = pd.DataFrame([{"author_ids": "1", "eid": "A"}, {"author_ids": "2", "eid": "B"}])
    references = pd.DataFrame(
        [{"citing_eid": "X", "cited_eid": "A"}, {"citing_eid": "X", "cited_eid": "B"}]
    )
    context = build_scoring_context(
        [left, right], article_author_df=articles, article_reference_df=references
    )
    decision = score_pair(left, right, context=context)
    evidence = next(item for item in decision.components if item.label == "co_citation")
    assert evidence.weight == 2.0
    assert evidence.reason == "shared citing papers: 1"


def test_article_journal_address_and_coauthor_name_evidence() -> None:
    left = make_author("1", "Thomas", "TJ")
    right = make_author("2", "Thomas", "TJ")
    articles = pd.DataFrame(
        [
            {
                "author_ids": "1;3",
                "eid": "A1",
                "source_title": "Nature",
                "country": "AUT",
                "city": "Vienna",
            },
            {
                "author_ids": "2;3",
                "eid": "B1",
                "source_title": "Nature",
                "country": "AUT",
                "city": "Vienna",
            },
        ]
    )
    decision = score_pair(left, right, context=build_scoring_context(articles))
    assert {"journal_match", "shared_coauthors", "publication_address_match"} <= component_labels(
        decision
    )

    names = pd.DataFrame(
        [
            {
                "author_ids": "1",
                "eid": "C1",
                "author_names": "Müller, Thomas;Smith, Alice;Jones, Bob",
            },
            {
                "author_ids": "2",
                "eid": "C2",
                "author_names": "Müller, Thomas;Smith, Alice;Miller, Carol",
            },
        ]
    )
    by_name = score_pair(left, right, context=build_scoring_context(names))
    assert "shared_coauthors" in component_labels(by_name)


def test_publication_name_word_order_does_not_count_author_as_shared_coauthor() -> None:
    left = author_from_row(
        {
            "author_id": "990000001",
            "name": "Mustermann, Max",
            "surname": "Mustermann",
            "given_name": "Max",
            "indexed_name": "Mustermann, Max",
            "initials": "M",
            "name_variants": "[]",
            "subject_areas": "[]",
        }
    )
    right = author_from_row(
        {
            "author_id": "990000003",
            "name": "Mustermann, Max",
            "surname": "Mustermann",
            "given_name": "Max",
            "indexed_name": "Mustermann, Max",
            "initials": "M",
            "name_variants": "[]",
            "subject_areas": "[]",
        }
    )
    articles = pd.DataFrame(
        [
            {
                "eid": "SYN-PUB-001",
                "author_ids": "990000001;990010001",
                "author_names": '["Max Mustermann", "Taylor Example"]',
            },
            {
                "eid": "SYN-PUB-003",
                "author_ids": "990000003;990010002",
                "author_names": '["Max Mustermann", "Morgan Sample"]',
            },
        ]
    )

    decision = score_pair(left, right, context=build_scoring_context(articles))

    assert "shared_coauthors" not in component_labels(decision)


def test_grouped_article_rows_and_id_name_overlap_are_not_double_counted() -> None:
    left = make_author("1", "Thomas", "T")
    right = make_author("2", "Thomas", "T")
    grouped = pd.DataFrame(
        [
            {
                "eid": "A1",
                "author_ids": 1,
                "author_names": "Müller, Thomas;Smith, Alice;Jones, Bob",
            },
            {
                "eid": "A1",
                "author_ids": 3,
                "author_names": "Müller, Thomas;Smith, Alice;Jones, Bob",
            },
            {
                "eid": "B1",
                "author_ids": 2,
                "author_names": "Müller, Thomas;Smith, Alice;Miller, Carol",
            },
            {
                "eid": "B1",
                "author_ids": 4,
                "author_names": "Müller, Thomas;Smith, Alice;Miller, Carol",
            },
        ]
    )
    decision = score_pair(left, right, context=build_scoring_context(grouped))
    assert "shared_coauthors" in component_labels(decision)
    duplicated_names = grouped.assign(author_names="Müller, Thomas;Smith, Alice")
    decision = score_pair(left, right, context=build_scoring_context(duplicated_names))
    coauthors = next(item for item in decision.components if item.label == "shared_coauthors")
    assert coauthors.weight == 4.0


def test_missing_and_malformed_publication_inputs_do_not_create_evidence() -> None:
    left = make_author("1", "Thomas", "T")
    right = make_author("2", "Thomas", "T")
    missing = build_scoring_context(article_author_df=pd.DataFrame())
    malformed = build_scoring_context(article_author_df=pd.DataFrame({"author_ids": ["1", "2"]}))
    assert missing.publication_index is None
    assert malformed.publication_index is None
    no_evidence = score_pair(left, right, context=missing)
    assert not (
        {"self_citation", "bibliographic_coupling", "co_citation"} & component_labels(no_evidence)
    )
