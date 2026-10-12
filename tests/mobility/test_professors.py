# Copyright © 2026 TU Wien

import pandas as pd

from acib.mobility.inbreeding import analyze_authors
from acib.mobility.professors import prepare_professor_author_profiles


def test_prepare_professor_author_profiles_merges_same_person_universities() -> None:
    authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "selected_author_ids": "[1001, 9001]",
                "selected_history": "{1001: [8001]}",
                "current_affiliation_ids": "[60022457]",
                "sc_id": 1,
                "researcher_name": "Muster, Max",
                "person_title": "Ao. Univ.-Prof.",
                "university_name": "Karl Franzens Uni Graz",
            },
            {
                "selected_disambiguated_cluster_id": 1001,
                "selected_author_ids": "[1001, 9001]",
                "selected_history": "{1001: [8001, 8002]}",
                "current_affiliation_ids": "[60018163]",
                "sc_id": 2,
                "researcher_name": "Muster, Maximilian",
                "person_title": "Univ.-Prof. Dr.",
                "university_name": "TU Wien",
            },
            {
                "selected_disambiguated_cluster_id": 1002,
                "selected_author_ids": "[1002]",
                "selected_history": "{}",
                "sc_id": 3,
                "researcher_name": "Example, Eva",
                "person_title": "Dr.",
                "university_name": "TU Graz",
            },
        ]
    )

    result = prepare_professor_author_profiles(authors)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["title"] == "Full Prof."
    assert row["person_title_class"] == "Full Prof."
    assert row["sc_ids"] == [1, 2]
    assert row["person_titles_raw"] == ["Ao. Univ.-Prof.", "Univ.-Prof. Dr."]
    assert row["selected_author_ids"] == [1001, 9001]
    assert row["selected_history"] == {"1001": [8001, 8002]}
    assert row["current_affiliation_ids"] == [60022457, 60018163]
    assert row["is_coaffiliated_professor"]
    assert row["coaffiliated_professor_universities"] == ["Karl Franzens Uni Graz", "TU Wien"]
    assert row["n_coaffiliated_professor_universities"] == 2
    assert row["n_merged_professor_rows"] == 2


def test_prepare_professor_author_profiles_does_not_merge_different_author_sets() -> None:
    authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "selected_author_ids": "[1001, 9001]",
                "sc_id": 1,
                "researcher_name": "Muster, Max",
                "person_title": "Ao. Univ.-Prof.",
                "university_name": "Karl Franzens Uni Graz",
            },
            {
                "selected_disambiguated_cluster_id": 1001,
                "selected_author_ids": "[1001, 9002]",
                "sc_id": 2,
                "researcher_name": "Muster, Maximilian",
                "person_title": "Univ.-Prof. Dr.",
                "university_name": "TU Wien",
            },
        ]
    )

    result = prepare_professor_author_profiles(authors)

    assert len(result) == 2
    assert result["sc_ids"].tolist() == [[1], [2]]
    assert result["is_merged_professor_profile"].tolist() == [False, False]


def test_prepare_professor_author_profiles_recomputes_stale_title_classes() -> None:
    authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 1,
                "person_title": "Prof.in (FH)",
                "person_title_class": "Full Prof.",
                "title": "Full Prof.",
                "university_name": "FH Gesundheit Tirol",
            }
        ]
    )

    result = prepare_professor_author_profiles(authors)

    assert result.loc[0, "title"] == "FH-Prof."
    assert result.loc[0, "person_title_class"] == "FH-Prof."
    assert result.loc[0, "title_classes"] == ["FH-Prof."]


def test_prepare_professor_author_profiles_classifies_mci_only_professor_as_fh_professor() -> None:
    authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 1,
                "person_title": "Prof. Dr.",
                "university_name": "mci hochschule",
            }
        ]
    )

    result = prepare_professor_author_profiles(authors)

    assert result.loc[0, "title"] == "FH-Prof."
    assert result.loc[0, "person_title_class"] == "FH-Prof."
    assert result.loc[0, "title_classes"] == ["FH-Prof."]


def test_prepare_professor_author_profiles_keeps_real_full_prof_for_non_mci_coaffiliation() -> None:
    authors = pd.DataFrame(
        [
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 1,
                "person_title": "Prof. Dr.",
                "university_name": "mci hochschule",
            },
            {
                "selected_disambiguated_cluster_id": 1001,
                "sc_id": 2,
                "person_title": "Univ.-Prof. Dr.",
                "university_name": "University of Vienna",
            },
        ]
    )

    result = prepare_professor_author_profiles(authors)

    assert result.loc[0, "title"] == "Full Prof."
    assert result.loc[0, "title_classes"] == ["Full Prof.", "FH-Prof."]
    assert result.loc[0, "coaffiliated_professor_universities"] == [
        "mci hochschule",
        "University of Vienna",
    ]


def test_university_detection_uses_merged_professor_university_list() -> None:
    article_author_df = pd.DataFrame(
        [
            {
                "eid": "first",
                "author_ids": 1001,
                "author_afids": 60018163,
                "cover_date": "2010-01-01",
            },
            {
                "eid": "last",
                "author_ids": 1001,
                "author_afids": 60018163,
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
                "person_title": "Prof.",
                "university_name": "Karl Franzens Uni Graz",
                "coaffiliated_professor_universities": ["Karl Franzens Uni Graz", "TU Wien"],
            }
        ]
    )
    affiliations = pd.DataFrame([{"affiliation_id": 60018163, "affiliation_id_parent": pd.NA}])

    _, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=affiliations,
        enable_university_affiliation_inbreeding_detection=True,
        first_affiliation_publication_count=1,
    )

    row = author_summary.loc[0]
    assert row["is_inbreeding_proxy"]
    assert row["person_title_raw"] == "Prof."
    assert row["title"] == "Full Prof."
    assert "person_title_class" not in author_summary.columns
    assert 60018163 in row["university_affiliation_ids"]
    assert row["coaffiliated_professor_universities"] == ["Karl Franzens Uni Graz", "TU Wien"]
