# Copyright © 2026 TU Wien

from pathlib import Path

import numpy as np
import pandas as pd

from acib.mobility.config import MobilityOutputs
from acib.mobility.constants import (
    AUTHOR_PROFILE_AMBIGUITY_REASON as AUTHOR_PROFILE_AMBIGUITY_REASON_COLUMN,
    AUTHOR_PROFILE_AMBIGUOUS as AUTHOR_PROFILE_AMBIGUOUS_COLUMN,
    MOBILITY_AUTHOR_PROFILE_ID as MOBILITY_AUTHOR_PROFILE_ID_COLUMN,
)
from acib.mobility.inbreeding import (
    InbreedingResult,
    analyze_authors,
    author_article_export,
    build_author_summary_row,
    detect_publication_mobility_spell,
    first_publication_affiliation_estimate,
    write_inbreeding_outputs,
)
from acib.mobility.input import (
    add_mobility_author_profile_ids,
    clean_affiliations,
    enrich_authors_with_researcher_metadata,
    general_asjc_subject_area_dict,
    parallelize_dataframe,
    process_input_data,
    selected_history_long,
)


def test_selected_history_long_combines_disambiguated_and_historical_ids() -> None:
    resolved_authors = pd.DataFrame(
        {
            "selected_disambiguated_cluster_id": [1001, 1002],
            "selected_history": ["{1001: [9001, 9002]}", "{}"],
            "historical_identifier": ["[8001, 9001]", "[8002]"],
        }
    )

    result = selected_history_long(resolved_authors=resolved_authors)

    assert result.to_dict(orient="records") == [
        {"current_author_id": 1001, "historical_author_id": 9001},
        {"current_author_id": 1001, "historical_author_id": 9002},
        {"current_author_id": 1001, "historical_author_id": 8001},
        {"current_author_id": 1002, "historical_author_id": 8002},
    ]


def test_metadata_enrichment_matches_string_and_integer_researcher_ids() -> None:
    authors = pd.DataFrame(
        {
            "sc_id": ["1"],
            "person_title": [None],
            "selected_author_ids": [[1001]],
        }
    )
    metadata = pd.DataFrame(
        {
            "sc_id": [1],
            "person_title": ["Prof."],
            "university_name": ["University of Vienna"],
        }
    )

    result = enrich_authors_with_researcher_metadata(
        authors=authors,
        researcher_metadata=metadata,
    )

    assert result.loc[0, "sc_id"] == "1"
    assert result.loc[0, "person_title"] == "Prof."
    assert result.loc[0, "university_name"] == "University of Vienna"
    assert result.loc[0, "selected_author_ids"] == [1001]


def test_mobility_author_profile_ids_mark_overlapping_nonidentical_profiles() -> None:
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "selected_author_ids": "[1001, 9001]",
                "sc_id": 1,
            },
            {
                "selected_disambiguated_cluster_id": 1001,
                "selected_author_ids": "[1001, 9002]",
                "sc_id": 2,
            },
            {
                "selected_disambiguated_cluster_id": 1003,
                "selected_author_ids": "[1003]",
                "sc_id": 3,
            },
        ]
    )

    result = add_mobility_author_profile_ids(resolved_authors)

    assert result[MOBILITY_AUTHOR_PROFILE_ID_COLUMN].nunique() == 3
    assert (
        result.loc[0, MOBILITY_AUTHOR_PROFILE_ID_COLUMN]
        != result.loc[1, MOBILITY_AUTHOR_PROFILE_ID_COLUMN]
    )
    assert result.loc[2, MOBILITY_AUTHOR_PROFILE_ID_COLUMN] == 1003
    assert result[AUTHOR_PROFILE_AMBIGUOUS_COLUMN].tolist() == [True, True, False]
    assert result.loc[0, AUTHOR_PROFILE_AMBIGUITY_REASON_COLUMN] == (
        "overlapping_nonidentical_selected_author_ids"
    )


def test_analyze_authors_marks_overlapping_nonidentical_profiles_for_manual_review() -> None:
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "selected_author_ids": "[1001, 9001]",
                "sc_id": 1,
                "researcher_name": "Muster, Max",
            },
            {
                "selected_disambiguated_cluster_id": 1001,
                "selected_author_ids": "[1001, 9002]",
                "sc_id": 2,
                "researcher_name": "Muster, Maria",
            },
        ]
    )
    prepared = add_mobility_author_profile_ids(resolved_authors)
    article_author_df = pd.DataFrame(
        [
            {
                "eid": "1",
                "author_ids": prepared.loc[0, MOBILITY_AUTHOR_PROFILE_ID_COLUMN],
                "author_afids": 10,
                "cover_date": "2020-03-01",
            },
            {
                "eid": "2",
                "author_ids": prepared.loc[1, MOBILITY_AUTHOR_PROFILE_ID_COLUMN],
                "author_afids": 20,
                "cover_date": "2021-03-01",
            },
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 10, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 20, "affiliation_id_parent": pd.NA},
        ]
    )

    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
    )

    assert author_summary["is_inbreeding_proxy"].isna().all()
    assert author_summary["inbreeding_manual_review_required"].tolist() == [True, True]
    assert author_summary["inbreeding_detection_status"].tolist() == [
        "manual_review_overlapping_selected_author_profile",
        "manual_review_overlapping_selected_author_profile",
    ]


def test_clean_affiliations_uses_splink_chatgpt_after_parent() -> None:
    affiliations = pd.DataFrame(
        [
            {
                "affiliation_id": 1,
                "affiliation_id_parent_splink_chatgpt": 50,
                "affiliation_id_parent_chatgpt": 40,
                "affiliation_id_parent": 30,
                "parent": 20,
                "umbrella_id": 10,
                "preferred_name": "Child",
            },
            {
                "affiliation_id": 2,
                "affiliation_id_parent_splink_chatgpt": 70,
                "affiliation_id_parent_chatgpt": pd.NA,
                "affiliation_id_parent": pd.NA,
                "parent": 60,
                "umbrella_id": 50,
                "preferred_name": "Parent fallback",
            },
            {
                "affiliation_id": 3,
                "affiliation_id_parent_splink_chatgpt": 80,
                "affiliation_id_parent_chatgpt": pd.NA,
                "affiliation_id_parent": pd.NA,
                "parent": pd.NA,
                "umbrella_id": 70,
                "preferred_name": "Splink fallback",
            },
            {
                "affiliation_id": 4,
                "affiliation_id_parent_splink_chatgpt": pd.NA,
                "affiliation_id_parent_chatgpt": pd.NA,
                "affiliation_id_parent": pd.NA,
                "parent": pd.NA,
                "umbrella_id": 60,
                "preferred_name": "Umbrella fallback",
            },
        ]
    )

    result = clean_affiliations(affiliations)

    assert result.loc[0, "affiliation_id_parent"] == 40
    assert result.loc[1, "affiliation_id_parent"] == 60
    assert result.loc[2, "affiliation_id_parent"] == 80
    assert result.loc[3, "affiliation_id_parent"] == 60


def test_analyze_authors_adds_academic_age_years_to_summary() -> None:
    article_author_df = pd.DataFrame(
        [
            {"eid": "1", "author_ids": 1001, "author_afids": 10, "cover_date": "2010-03-01"},
            {"eid": "middle", "author_ids": 1001, "author_afids": 30, "cover_date": "2015-03-01"},
            {"eid": "2", "author_ids": 1001, "author_afids": 20, "cover_date": "2024-09-01"},
        ]
    )
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 1,
                "researcher_name": "Muster, Max",
            }
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 10, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 20, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 30, "affiliation_id_parent": pd.NA},
        ]
    )

    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
    )

    assert author_summary.loc[0, "academic_age_years"] == 14
    assert not author_summary.loc[0, "is_inbreeding_proxy"]
    assert author_summary.loc[0, "n_unique"] == 3
    assert author_summary.loc[0, "author_profile_article_eids"] == ["1", "middle", "2"]


def test_analyze_authors_keeps_multiple_researchers_for_same_author_id() -> None:
    article_author_df = pd.DataFrame(
        [
            {"eid": "1", "author_ids": 1001, "author_afids": 10, "cover_date": "2010-03-01"},
            {"eid": "2", "author_ids": 1001, "author_afids": 20, "cover_date": "2024-09-01"},
        ]
    )
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 1,
                "researcher_name": "Muster, Max",
            },
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 2,
                "researcher_name": "Muster, Maria",
            },
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 10, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 20, "affiliation_id_parent": pd.NA},
        ]
    )

    author_affiliations, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
    )

    assert author_summary["sc_id"].tolist() == [1, 2]
    assert author_summary["author_ids"].tolist() == [1001, 1001]
    assert author_affiliations["sc_id"].tolist() == [1, 1, 2, 2]


def test_parallel_analyze_authors_matches_sequential_outputs() -> None:
    article_author_df = pd.DataFrame(
        [
            {"eid": "1", "author_ids": 1001, "author_afids": 10, "cover_date": "2010-03-01"},
            {"eid": "2", "author_ids": 1001, "author_afids": 20, "cover_date": "2024-09-01"},
            {"eid": "3", "author_ids": 1002, "author_afids": 30, "cover_date": "2012-05-01"},
            {"eid": "4", "author_ids": 1002, "author_afids": 40, "cover_date": "2023-05-01"},
        ]
    )
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "selected_author_ids": "['1001']",
                "selected_history": "{}",
                "sc_id": 1,
                "person_title": "Prof.",
                "researcher_name": "Muster, Max",
            },
            {
                "selected_disambiguated_cluster_id": 1002,
                "selected_author_ids": "['1002']",
                "selected_history": "{}",
                "sc_id": 2,
                "person_title": "Prof.",
                "researcher_name": "Muster, Maria",
            },
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 10, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 20, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 30, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 40, "affiliation_id_parent": pd.NA},
        ]
    )

    sequential_affiliations, sequential_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
        max_workers=1,
    )
    parallel_affiliations, parallel_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
        max_workers=2,
    )

    pd.testing.assert_frame_equal(sequential_affiliations, parallel_affiliations)
    pd.testing.assert_frame_equal(sequential_summary, parallel_summary)


def test_parallel_chunk_transform_without_temporary_index_matches_sequential() -> None:
    frame = pd.DataFrame({"value": ["a", "b", "c", "d"]}, index=[10, 20, 30, 40])

    sequential = parallelize_dataframe(
        _drop_temporary_parallel_index,
        frame,
        chunk_size=1,
        max_workers=1,
    )
    parallel = parallelize_dataframe(
        _drop_temporary_parallel_index,
        frame,
        chunk_size=1,
        max_workers=2,
    )

    assert sequential["value"].tolist() == ["a", "b", "c", "d"]
    assert sorted(parallel["value"]) == ["a", "b", "c", "d"]
    assert parallel.index.tolist() == list(range(len(frame)))


def test_parallel_input_preparation_matches_sequential_for_chunked_articles() -> None:
    articles = pd.DataFrame(
        {
            "eid": [f"paper-{index}" for index in range(5_001)],
            "author_ids": ["1001"] * 5_001,
            "author_afids": ["10"] * 5_001,
            "cover_date": ["2020-01-01"] * 5_001,
        }
    )
    resolved_authors = pd.DataFrame(
        {
            "selected_disambiguated_cluster_id": [1001],
            "selected_author_ids": ["[1001]"],
            "selected_history": ["{}"],
            "sc_id": [1],
        }
    )
    affiliations = pd.DataFrame([{"affiliation_id": 10, "affiliation_id_parent": pd.NA}])

    sequential, sequential_affiliations = process_input_data(
        author_affiliations=affiliations,
        articles=articles,
        resolved_authors=resolved_authors,
        max_workers=1,
    )
    parallel, parallel_affiliations = process_input_data(
        author_affiliations=affiliations,
        articles=articles,
        resolved_authors=resolved_authors,
        max_workers=2,
    )

    pd.testing.assert_frame_equal(sequential, parallel)
    pd.testing.assert_frame_equal(sequential_affiliations, parallel_affiliations)
    assert len(parallel) == 5_001
    assert parallel["eid"].iloc[[0, -1]].tolist() == ["paper-0", "paper-5000"]


def _drop_temporary_parallel_index(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["_parallel_gdf_index"], errors="ignore")


def test_analyze_authors_preserves_disambiguation_metadata_in_proxy() -> None:
    article_author_df = pd.DataFrame(
        [
            {"eid": "1", "author_ids": 1001, "author_afids": 10, "cover_date": "2010-03-01"},
            {"eid": "2", "author_ids": 1001, "author_afids": 20, "cover_date": "2024-09-01"},
        ]
    )
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "selected_author_ids": "['1001', '9001']",
                "selected_history": "{1001: [9001]}",
                "sc_id": 1,
                "person_title": "Prof.",
                "researcher_name": "Muster, Max",
            }
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 10, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 20, "affiliation_id_parent": pd.NA},
        ]
    )

    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
    )

    row = author_summary.loc[0]
    assert row["author_ids"] == 1001
    assert row["selected_disambiguated_cluster_id"] == 1001
    assert row["selected_author_ids"] == "['1001', '9001']"
    assert row["selected_history"] == "{1001: [9001]}"
    assert row["person_title_raw"] == "Prof."
    assert row["title"] == "Full Prof."


def test_analyze_authors_adds_university_detection_article_eids() -> None:
    article_author_df = pd.DataFrame(
        [
            {
                "eid": "first_1",
                "author_ids": 1001,
                "author_afids": 60025988,
                "cover_date": "2010-01-01",
            },
            {"eid": "first_2", "author_ids": 1001, "author_afids": 30, "cover_date": "2011-01-01"},
            {"eid": "middle", "author_ids": 1001, "author_afids": 40, "cover_date": "2012-01-01"},
            {
                "eid": "last",
                "author_ids": 1001,
                "author_afids": 60025988,
                "cover_date": "2020-01-01",
            },
        ]
    )
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 1,
                "researcher_name": "Muster, Max",
                "university_name": "University of Vienna",
            }
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 60025988, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 30, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 40, "affiliation_id_parent": pd.NA},
        ]
    )

    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
        enable_university_affiliation_inbreeding_detection=True,
        first_affiliation_publication_count=2,
    )

    row = author_summary.loc[0]
    assert row["is_inbreeding_proxy"]
    assert row["n_unique"] == 4
    assert row["author_profile_article_eids"] == ["first_1", "first_2", "middle", "last"]


def test_first_publication_affiliation_estimate_deduplicates_doi_events() -> None:
    group = pd.DataFrame(
        [
            {
                "eid": "first_graz",
                "doi": "10.1000/first",
                "cover_date": "2010-01-01",
                "author_afids": 60015267,
                "source_id": "journal-1",
                "publication_name": "First Article",
            },
            {
                "eid": "monash_chapter",
                "doi": "10.4324/book",
                "cover_date": "2012-01-01",
                "author_afids": 60019578,
                "source_id": "book-1",
                "publication_name": "Book Title",
            },
            {
                "eid": "monash_editorial_duplicate",
                "doi": "https://doi.org/10.4324/book",
                "cover_date": "2012-01-01",
                "author_afids": 60019578,
                "source_id": "book-1",
                "publication_name": "Book Title",
            },
            {
                "eid": "later_graz",
                "doi": "",
                "cover_date": "2016-01-01",
                "author_afids": 60015267,
                "source_id": "book-2",
                "publication_name": "Later Chapter",
            },
        ]
    )

    first_ids, counts, observed_count, status = first_publication_affiliation_estimate(
        group,
        publication_count=3,
    )

    assert first_ids == [60015267]
    assert counts == {60015267: 2, 60019578: 1}
    assert observed_count == 3
    assert status == "assigned"


def test_first_publication_affiliation_estimate_keeps_ambiguous_same_doi_eids() -> None:
    group = pd.DataFrame(
        [
            {
                "eid": "first_graz",
                "doi": "10.1000/first",
                "cover_date": "2010-01-01",
                "author_afids": 60015267,
                "source_id": "journal-1",
                "publication_name": "First Article",
            },
            {
                "eid": "monash_chapter",
                "doi": "10.4324/book",
                "cover_date": "2012-01-01",
                "author_afids": 60019578,
                "source_id": "book-1",
                "publication_name": "Book Title",
            },
            {
                "eid": "monash_related_record",
                "doi": "https://doi.org/10.4324/book",
                "cover_date": "2012-01-01",
                "author_afids": 60019578,
                "source_id": "book-2",
                "publication_name": "Book Title",
            },
            {
                "eid": "later_graz",
                "doi": "",
                "cover_date": "2016-01-01",
                "author_afids": 60015267,
                "source_id": "book-3",
                "publication_name": "Later Chapter",
            },
        ]
    )

    first_ids, counts, observed_count, status = first_publication_affiliation_estimate(
        group,
        publication_count=3,
    )

    assert first_ids == [60019578]
    assert counts == {60015267: 1, 60019578: 2}
    assert observed_count == 4
    assert status == "assigned"


def test_first_n_years_method_classifies_target_present_in_early_window() -> None:
    article_author_df = pd.DataFrame(
        [
            {
                "eid": "first",
                "author_ids": 1001,
                "author_afids": 60009999,
                "cover_date": "1998-01-01",
            },
            {
                "eid": "early",
                "author_ids": 1001,
                "author_afids": 60018163,
                "cover_date": "1999-01-01",
            },
            {
                "eid": "current",
                "author_ids": 1001,
                "author_afids": 60018163,
                "cover_date": "2003-01-01",
            },
        ]
    )
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 1,
                "researcher_name": "Muster, Max",
                "university_name": "TU Wien",
            }
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 60009999, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 60018163, "affiliation_id_parent": pd.NA},
        ]
    )

    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
        enable_university_affiliation_inbreeding_detection=True,
        early_affiliation_method="first_n_years",
        first_affiliation_year_count=2,
    )

    row = author_summary.loc[0]
    assert row["is_inbreeding_proxy"]
    assert row["early_affiliation_method"] == "first_n_years"
    assert row["first_publication_year"] == 1998
    assert row["early_window_end_year"] == 1999
    assert row["n_publications_in_early_window"] == 2
    assert row["target_in_early_window"]


def test_first_n_years_method_ignores_target_after_early_window() -> None:
    article_author_df = pd.DataFrame(
        [
            {
                "eid": "first",
                "author_ids": 1001,
                "author_afids": 60009999,
                "cover_date": "1998-01-01",
            },
            {
                "eid": "current",
                "author_ids": 1001,
                "author_afids": 60018163,
                "cover_date": "2003-01-01",
            },
        ]
    )
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 1,
                "researcher_name": "Muster, Max",
                "university_name": "TU Wien",
            }
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 60009999, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 60018163, "affiliation_id_parent": pd.NA},
        ]
    )

    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
        enable_university_affiliation_inbreeding_detection=True,
        early_affiliation_method="first_n_years",
        first_affiliation_year_count=2,
    )

    row = author_summary.loc[0]
    assert not row["is_inbreeding_proxy"]
    assert row["early_window_end_year"] == 1999
    assert row["n_publications_in_early_window"] == 1
    assert not row["target_in_early_window"]
    assert row["inbreeding_detection_status"] == "assigned_first_affiliation_not_current_university"


def test_first_n_years_method_counts_coaffiliation_as_target_presence() -> None:
    article_author_df = pd.DataFrame(
        [
            {
                "eid": "first",
                "author_ids": 1001,
                "author_afids": 60009999,
                "cover_date": "1998-01-01",
            },
            {
                "eid": "coaff",
                "author_ids": 1001,
                "author_afids": 60009999,
                "cover_date": "1999-01-01",
            },
            {
                "eid": "coaff",
                "author_ids": 1001,
                "author_afids": 60018163,
                "cover_date": "1999-01-01",
            },
            {
                "eid": "current",
                "author_ids": 1001,
                "author_afids": 60018163,
                "cover_date": "2003-01-01",
            },
        ]
    )
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 1,
                "researcher_name": "Muster, Max",
                "university_name": "TU Wien",
            }
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 60009999, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 60018163, "affiliation_id_parent": pd.NA},
        ]
    )

    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
        enable_university_affiliation_inbreeding_detection=True,
        early_affiliation_method="first_n_years",
        first_affiliation_year_count=2,
    )

    row = author_summary.loc[0]
    assert row["is_inbreeding_proxy"]
    assert row["n_publications_in_early_window"] == 2
    assert row["target_in_early_window"]
    assert row["estimated_first_affiliation_counts"] == {60009999: 2, 60018163: 1}


def test_first_n_years_method_deduplicates_publication_events() -> None:
    article_author_df = pd.DataFrame(
        [
            {
                "eid": "first",
                "author_ids": 1001,
                "author_afids": 60009999,
                "cover_date": "1998-01-01",
            },
            {
                "eid": "first",
                "author_ids": 1001,
                "author_afids": 60009999,
                "cover_date": "1998-01-01",
            },
            {
                "eid": "current",
                "author_ids": 1001,
                "author_afids": 60018163,
                "cover_date": "2003-01-01",
            },
        ]
    )
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 1,
                "researcher_name": "Muster, Max",
                "university_name": "TU Wien",
            }
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 60009999, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 60018163, "affiliation_id_parent": pd.NA},
        ]
    )

    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
        enable_university_affiliation_inbreeding_detection=True,
        early_affiliation_method="first_n_years",
        first_affiliation_year_count=2,
    )

    row = author_summary.loc[0]
    assert row["n_publications_in_early_window"] == 1
    assert row["estimated_first_affiliation_counts"] == {60009999: 1}


def test_min_publication_history_years_blocks_short_histories() -> None:
    article_author_df = pd.DataFrame(
        [
            {
                "eid": "first",
                "author_ids": 1001,
                "author_afids": 60018163,
                "cover_date": "1998-01-01",
            },
            {
                "eid": "current",
                "author_ids": 1001,
                "author_afids": 60018163,
                "cover_date": "2003-01-01",
            },
        ]
    )
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 1,
                "researcher_name": "Muster, Max",
                "university_name": "TU Wien",
            }
        ]
    )
    affiliations = pd.DataFrame([{"affiliation_id": 60018163, "affiliation_id_parent": pd.NA}])

    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
        enable_university_affiliation_inbreeding_detection=True,
        early_affiliation_method="first_n_years",
        first_affiliation_year_count=2,
        min_publication_history_years=6,
    )

    row = author_summary.loc[0]
    assert pd.isna(row["is_inbreeding_proxy"])
    assert row["publication_history_years"] == 5
    assert not row["min_publication_history_requirement_met"]
    assert row["inbreeding_detection_status"] == "not_classified_publication_history_too_short"


def test_implausibly_long_publication_history_requires_manual_review() -> None:
    article_author_df = pd.DataFrame(
        [
            {
                "eid": "first",
                "author_ids": 1001,
                "author_afids": 60018163,
                "cover_date": "1950-01-01",
            },
            {
                "eid": "current",
                "author_ids": 1001,
                "author_afids": 60018163,
                "cover_date": "2026-01-01",
            },
        ]
    )
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 1,
                "researcher_name": "Muster, Max",
                "university_name": "TU Wien",
            }
        ]
    )
    affiliations = pd.DataFrame([{"affiliation_id": 60018163, "affiliation_id_parent": pd.NA}])

    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
        enable_university_affiliation_inbreeding_detection=True,
        early_affiliation_method="first_n_years",
        first_affiliation_year_count=2,
    )

    row = author_summary.loc[0]
    assert pd.isna(row["is_inbreeding_proxy"])
    assert row["academic_age_years"] == 76
    assert row["publication_history_years"] == 76
    assert bool(row["inbreeding_manual_review_required"])
    assert row["inbreeding_detection_status"] == "manual_review_publication_history_too_long"


def test_first_n_publications_method_remains_selectable() -> None:
    article_author_df = pd.DataFrame(
        [
            {
                "eid": "first",
                "author_ids": 1001,
                "author_afids": 60018163,
                "cover_date": "1998-01-01",
            },
            {
                "eid": "second",
                "author_ids": 1001,
                "author_afids": 60009999,
                "cover_date": "1999-01-01",
            },
            {
                "eid": "current",
                "author_ids": 1001,
                "author_afids": 60018163,
                "cover_date": "2003-01-01",
            },
        ]
    )
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 1,
                "researcher_name": "Muster, Max",
                "university_name": "TU Wien",
            }
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 60009999, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 60018163, "affiliation_id_parent": pd.NA},
        ]
    )

    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
        enable_university_affiliation_inbreeding_detection=True,
        early_affiliation_method="first_n_publications",
        first_affiliation_publication_count=1,
    )

    row = author_summary.loc[0]
    assert row["is_inbreeding_proxy"]
    assert row["early_affiliation_method"] == "first_n_publications"
    assert row["first_affiliation_publication_count_observed"] == 3
    assert pd.isna(row["early_window_end_year"])
    assert pd.isna(row["n_publications_in_early_window"])


def test_author_article_export_groups_affiliations_per_author_and_article() -> None:
    article_author_df = pd.DataFrame(
        [
            {
                "eid": "paper",
                "author_ids": 1001,
                "author_afids": 10,
                "unmapped_id": "10-11",
                "cover_date": "2020-01-01",
                "doi": "10/example",
                "source_id": "journal-1",
            },
            {
                "eid": "paper",
                "author_ids": 1001,
                "author_afids": 20,
                "unmapped_id": "10-11",
                "cover_date": "2020-01-01",
                "doi": "10/example",
                "source_id": "journal-1",
            },
            {
                "eid": "paper",
                "author_ids": 1002,
                "author_afids": 30,
                "unmapped_id": 30,
                "cover_date": "2020-01-01",
                "doi": "10/example",
                "source_id": "journal-1",
            },
            {
                "eid": "unassigned-paper",
                "author_ids": None,
                "author_afids": 40,
                "unmapped_id": 40,
                "cover_date": "2021-01-01",
                "doi": "10/unassigned",
                "source_id": "journal-1",
            },
        ]
    )

    result = author_article_export(article_author_df)

    assert len(result) == 2
    assert result[["author_ids", "eid"]].to_records(index=False).tolist() == [
        (1001, "paper"),
        (1002, "paper"),
    ]
    assert result.loc[0, "author_afids"] == "10;20"
    assert result.loc[0, "unmapped_id"] == "10;11"
    assert result.loc[1, "author_afids"] == "30"
    assert result.loc[1, "unmapped_id"] == "30"
    assert result["doi"].tolist() == ["10/example", "10/example"]
    assert result["author_ids"].dtype == "int64"


def test_write_inbreeding_outputs_writes_three_artifacts(tmp_path: Path) -> None:
    result = InbreedingResult(
        author_affiliations=pd.DataFrame(
            {"author_ids": [1001], "affiliation_id": [10], "affiliation_id_parent": [pd.NA]}
        ),
        author_summary=pd.DataFrame({"author_ids": [1001], "is_inbreeding_proxy": [True]}),
        author_articles=pd.DataFrame(
            {
                "eid": ["paper"],
                "author_ids": [1001],
                "author_afids": ["10;20"],
                "unmapped_id": ["100;200"],
                "doi": ["10/example"],
            }
        ),
    )

    outputs = MobilityOutputs(directory=tmp_path)
    write_inbreeding_outputs(result=result, outputs=outputs)

    articles = pd.read_csv(tmp_path / "author_articles.csv")
    assert articles.to_dict(orient="records") == [
        {
            "eid": "paper",
            "author_ids": 1001,
            "author_afids": "10;20",
            "unmapped_id": "100;200",
            "doi": "10/example",
        }
    ]
    assert not (tmp_path / "author_affiliations_unique.csv").exists()
    assert {path.name for path in outputs.all_paths()} == {
        "author_affiliations.csv",
        "author_inbreeding_proxy.csv",
        "author_articles.csv",
    }


def test_detect_publication_mobility_counts_away_years_abroad() -> None:
    publications = pd.DataFrame(
        [
            {
                "eid": "home",
                "author_afids": 10,
                "inst_key": 10,
                "cover_date": "2018-01-01",
                "affiliation_country": "Austria",
            },
            {
                "eid": "away_austria",
                "author_afids": 20,
                "inst_key": 20,
                "cover_date": "2019-01-01",
                "affiliation_country": "Austria",
            },
            {
                "eid": "away_abroad_1",
                "author_afids": 30,
                "inst_key": 30,
                "cover_date": "2020-01-01",
                "affiliation_country": "Germany",
            },
            {
                "eid": "away_abroad_2",
                "author_afids": 40,
                "inst_key": 40,
                "cover_date": "2020-06-01",
                "affiliation_country": "Switzerland;Austria",
            },
        ]
    )

    result = detect_publication_mobility_spell(publications, home_inst_set={10})

    assert result["n_away_years"] == 2
    assert result["n_away_years_domestic"] == 1
    assert result["n_away_years_abroad"] == 1


def publication_rows_for_parent_sets(parent_sets: list[list[int]]) -> pd.DataFrame:
    rows = []
    raw_affiliation_id = 1
    for publication_index, parent_set in enumerate(parent_sets, start=1):
        for parent_id in parent_set:
            rows.append(
                {
                    "eid": f"paper_{publication_index}",
                    "author_afids": raw_affiliation_id,
                    "inst_key": parent_id,
                    "cover_date": f"2020-0{publication_index}-01",
                    "affiliation_country": "Austria",
                }
            )
            raw_affiliation_id += 1
    return pd.DataFrame(rows)


def test_coaffiliation_publication_indicator_counts_distinct_author_affiliations() -> None:
    cases = [
        ([100], 0, 0, 0.0),
        ([100, 200], 1, 1, 1.0),
        ([100, 200, 300], 1, 2, 1.0),
        ([100, 100], 0, 0, 0.0),
        ([100, 200, 200], 1, 1, 1.0),
        ([500, 1000], 1, 1, 1.0),
    ]

    for parent_set, expected_coaffiliated_pubs, expected_coaffiliations, expected_share in cases:
        result = detect_publication_mobility_spell(
            publication_rows_for_parent_sets([parent_set]),
            home_inst_set={100},
        )

        assert result["n_coaffiliation_publications"] == expected_coaffiliated_pubs
        assert result["n_coaffiliations"] == expected_coaffiliations
        assert result["coaffiliation_publication_share"] == expected_share


def test_author_affiliation_positions_prevent_coauthor_institutions_from_counting() -> None:
    resolved_authors = pd.DataFrame(
        [
            {
                "author_id": 1,
                "sc_id": 1,
                "person_title": "Prof. Dr.",
                "researcher_name": "Focal, Professor",
                "university_name": "University of Vienna",
            }
        ]
    )
    articles = pd.DataFrame(
        [
            {
                "eid": "shared_paper",
                "author_ids": "1,2",
                "author_afids": "10,20",
                "cover_date": "2020-01-01",
                "affiliation_country": "Austria",
            },
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 10, "affiliation_id_parent": 60025988},
            {"affiliation_id": 20, "affiliation_id_parent": 200},
        ]
    )
    article_author_df, normalized_affiliations = process_input_data(
        author_affiliations=affiliations,
        articles=articles,
        resolved_authors=resolved_authors,
        max_workers=1,
    )

    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=normalized_affiliations,
        enable_university_affiliation_inbreeding_detection=True,
        first_affiliation_publication_count=1,
    )

    row = author_summary.iloc[0]
    assert row["n_coaffiliation_publications"] == 0
    assert row["coaffiliation_publication_share"] == 0.0


def test_author_affiliation_positions_count_only_focal_author_group() -> None:
    resolved_authors = pd.DataFrame(
        [
            {
                "author_id": 1,
                "sc_id": 1,
                "person_title": "Prof. Dr.",
                "researcher_name": "Focal, Professor",
                "university_name": "University of Vienna",
            },
            {
                "author_id": 2,
                "sc_id": 2,
                "person_title": "Prof. Dr.",
                "researcher_name": "Other, Professor",
                "university_name": "TU Wien",
            },
        ]
    )
    articles = pd.DataFrame(
        [
            {
                "eid": "shared_paper",
                "author_ids": "1,2",
                "author_afids": "10-20,30",
                "cover_date": "2020-01-01",
                "affiliation_country": "Austria",
            },
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 10, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 20, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 30, "affiliation_id_parent": pd.NA},
        ]
    )

    article_author_df, normalized_affiliations = process_input_data(
        author_affiliations=affiliations,
        articles=articles,
        resolved_authors=resolved_authors,
        max_workers=1,
    )
    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=normalized_affiliations,
        enable_university_affiliation_inbreeding_detection=True,
        first_affiliation_publication_count=1,
    )

    values = author_summary.set_index("sc_id")["coaffiliation_publication_share"].to_dict()
    assert values == {1: 1.0, 2: 0.0}


def test_coaffiliation_publication_share_uses_total_known_publications_denominator() -> None:
    result = detect_publication_mobility_spell(
        publication_rows_for_parent_sets(
            [
                [100],
                [100, 200],
                [100, 300],
                [200, 300],
            ]
        ),
        home_inst_set={100},
    )

    assert result["n_coaffiliation_publications"] == 3
    assert result["coaffiliation_publication_share"] == 3 / 4


def test_coaffiliation_harmonization_collapses_raw_ids_to_same_parent() -> None:
    resolved_authors = pd.DataFrame(
        [
            {
                "author_id": 1,
                "sc_id": 1,
                "person_title": "Prof. Dr.",
                "researcher_name": "Focal, Professor",
                "university_name": "University of Vienna",
            }
        ]
    )
    articles = pd.DataFrame(
        [
            {
                "eid": "paper",
                "author_ids": "1",
                "author_afids": "10-20",
                "cover_date": "2020-01-01",
                "affiliation_country": "Austria",
            }
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 10, "affiliation_id_parent": 100},
            {"affiliation_id": 20, "affiliation_id_parent": 100},
        ]
    )
    article_author_df, normalized_affiliations = process_input_data(
        author_affiliations=affiliations,
        articles=articles,
        resolved_authors=resolved_authors,
        max_workers=1,
    )
    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=normalized_affiliations,
    )

    row = author_summary.iloc[0]
    assert row["n_coaffiliation_publications"] == 0
    assert row["coaffiliation_publication_share"] == 0.0


def test_coaffiliation_harmonization_keeps_genuinely_different_parents() -> None:
    resolved_authors = pd.DataFrame(
        [
            {
                "author_id": 1,
                "sc_id": 1,
                "person_title": "Prof. Dr.",
                "researcher_name": "Focal, Professor",
                "university_name": "University of Vienna",
            }
        ]
    )
    articles = pd.DataFrame(
        [
            {
                "eid": "paper",
                "author_ids": "1",
                "author_afids": "10-20",
                "cover_date": "2020-01-01",
                "affiliation_country": "Austria",
            }
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 10, "affiliation_id_parent": 100},
            {"affiliation_id": 20, "affiliation_id_parent": 200},
        ]
    )
    article_author_df, normalized_affiliations = process_input_data(
        author_affiliations=affiliations,
        articles=articles,
        resolved_authors=resolved_authors,
        max_workers=1,
    )
    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=normalized_affiliations,
    )

    row = author_summary.iloc[0]
    assert row["n_coaffiliation_publications"] == 1
    assert row["coaffiliation_publication_share"] == 1.0


def test_coaffiliation_harmonization_keeps_genuinely_different_parents_directly() -> None:
    result = detect_publication_mobility_spell(
        pd.DataFrame(
            [
                {
                    "eid": "paper",
                    "author_afids": 10,
                    "inst_key": 100,
                    "cover_date": "2020-01-01",
                    "affiliation_country": "Austria",
                },
                {
                    "eid": "paper",
                    "author_afids": 20,
                    "inst_key": 200,
                    "cover_date": "2020-01-01",
                    "affiliation_country": "Germany",
                },
            ]
        ),
        home_inst_set={999},
    )

    assert result["n_coaffiliation_publications"] == 1
    assert result["coaffiliation_publication_share"] == 1.0


def test_author_summary_counts_mobility_relative_to_professor_university_for_non_inbred() -> None:
    author_match = pd.DataFrame(
        {
            "sc_id": [1],
            "person_title": ["Prof. Dr."],
            "researcher_name_processed": ["prof, noninbred"],
            "university_name": ["University of Vienna"],
        }
    )
    publications = pd.DataFrame(
        [
            {
                "eid": "early",
                "author_afids": 100,
                "institution_parent_id": 100,
                "inst_key": 100,
                "cover_date": pd.Timestamp("2000-01-01"),
                "affiliation_country": "Austria",
            },
            {
                "eid": "current",
                "author_afids": 60025988,
                "institution_parent_id": 60025988,
                "inst_key": 60025988,
                "cover_date": pd.Timestamp("2010-01-01"),
                "affiliation_country": "Austria",
            },
            {
                "eid": "coaffiliated",
                "author_afids": 60025988,
                "institution_parent_id": 60025988,
                "inst_key": 60025988,
                "cover_date": pd.Timestamp("2011-01-01"),
                "affiliation_country": "Austria",
            },
            {
                "eid": "coaffiliated",
                "author_afids": 300,
                "institution_parent_id": 300,
                "inst_key": 300,
                "cover_date": pd.Timestamp("2011-01-01"),
                "affiliation_country": "Germany",
            },
        ]
    )

    result = build_author_summary_row(
        1,
        author_match,
        publications,
        publications,
        enable_university_affiliation_inbreeding_detection=True,
        first_affiliation_publication_count=1,
    )

    assert result["is_inbreeding_proxy"] is False
    assert result["university_affiliation_ids"] == [60025988]
    assert result["shared_inst_set"] == []
    assert result["n_away_years"] == 1
    assert result["n_coaffiliation_publications"] == 1
    assert result["coaffiliation_publication_share"] == 1 / 3


def test_general_asjc_subject_area_dict_handles_numpy_array_entries() -> None:
    value = np.array(
        [
            {"area": "Bioengineering", "code": 1502, "abbreviation": "CENG"},
            {"area": "Communication", "code": 3315, "abbreviation": "SOCI"},
        ],
        dtype=object,
    )

    assert general_asjc_subject_area_dict(value) == {
        1500: "General Chemical Engineering",
        3300: "General Social Sciences",
    }


def test_general_asjc_subject_area_dict_uses_official_parent_labels_from_codes() -> None:
    value = [
        {"area": "Ecology, Evolution, Behavior and Systematics", "code": 1105},
        {"area": "Endocrinology, Diabetes and Metabolism", "code": 2712},
        {"area": "Electronic, Optical and Magnetic Materials", "code": 2504},
    ]

    assert general_asjc_subject_area_dict(value) == {
        1100: "General Agricultural and Biological Sciences",
        2500: "General Materials Science",
        2700: "General Medicine",
    }


def test_analyze_authors_adds_mean_coauthors_from_full_publication_author_list() -> None:
    articles = pd.DataFrame(
        [
            {
                "eid": "paper_1",
                "author_ids": "1001;9001;2002",
                "author_afids": "10;10;20",
                "cover_date": "2020-01-01",
                "subject_areas": '[{"area": "Bioengineering", "code": 1502}]',
            },
            {
                "eid": "paper_2",
                "author_ids": "1001;4004",
                "author_afids": "10;40",
                "cover_date": "2021-01-01",
                "subject_areas": '[{"area": "Chemical Engineering (miscellaneous)", "code": 1501}]',
            },
            {
                "eid": "paper_3",
                "author_ids": "1001;5005",
                "author_afids": "20;40",
                "cover_date": "2022-01-01",
                "subject_areas": '[{"area": "Chemistry", "code": 1600}]',
            },
        ]
    )
    resolved_authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "selected_author_ids": "['1001', '9001']",
                "sc_id": 1,
                "researcher_name": "Muster, Max",
            }
        ]
    )
    affiliations = pd.DataFrame(
        [
            {"affiliation_id": 10, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 20, "affiliation_id_parent": pd.NA},
            {"affiliation_id": 40, "affiliation_id_parent": pd.NA},
        ]
    )

    article_author_df, normalized_affiliations = process_input_data(
        author_affiliations=affiliations,
        articles=articles,
        resolved_authors=resolved_authors,
        max_workers=1,
    )
    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=normalized_affiliations,
    )

    row = author_summary.loc[0]
    assert row["n_unique"] == 3
    assert row["total_coauthors_across_publications"] == 3
    assert row["mean_coauthors_per_publication"] == 1.0
    assert row["article_general_asjc_subject_areas"] == {
        1500: "General Chemical Engineering",
        1600: "General Chemistry",
    }
    assert row["article_general_asjc_subject_area_publication_counts"] == {
        1500: {"category": "General Chemical Engineering", "n": 2},
        1600: {"category": "General Chemistry", "n": 1},
    }
