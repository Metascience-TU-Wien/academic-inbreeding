# Copyright © 2026 TU Wien

from pathlib import Path

import pandas as pd
import pytest

from acib.mobility.config import MobilityConfig, MobilityOutputs
from acib.mobility.constants import (
    INBREEDING_DETECTION_STATUS,
    IS_INBREEDING_PROXY,
)
from acib.mobility.inbreeding import build_author_summary_row
from acib.mobility.input import load_mobility_inputs
from acib.mobility.openalex import (
    attach_openalex_current_affiliations,
    read_openalex_mobility_affiliations,
    read_openalex_mobility_articles,
)
from acib.mobility.schemas import MobilityAffiliationSchema
from acib.name_disambiguation.constants import (
    CURRENT_AFFILIATION_IDS,
    SELECTED_AUTHOR_IDS,
)


def test_current_affiliations_union_selected_author_ids_without_losing_strings(
    tmp_path: Path,
) -> None:
    pairs_path = tmp_path / "pairs.parquet"
    pd.DataFrame(
        {
            "author_id": [
                "https://openalex.org/A2",
                "https://openalex.org/A3",
                "https://openalex.org/A4",
                "https://openalex.org/A4",
                "https://openalex.org/A5",
            ],
            "institution_id": [
                "https://openalex.org/I2",
                "https://openalex.org/I3",
                "https://openalex.org/I4",
                "https://openalex.org/I4",
                "https://openalex.org/I5",
            ],
            "current_affiliation": [True, True, True, True, False],
        }
    ).to_parquet(pairs_path, index=False)
    authors = pd.DataFrame(
        {
            SELECTED_AUTHOR_IDS: [
                (
                    "['https://openalex.org/A2', 'https://openalex.org/A3', "
                    "'https://openalex.org/A4']"
                ),
                "['https://openalex.org/A5']",
            ]
        }
    )

    result = attach_openalex_current_affiliations(authors, pairs_path)

    assert result[CURRENT_AFFILIATION_IDS].tolist() == [
        [
            "https://openalex.org/I2",
            "https://openalex.org/I3",
            "https://openalex.org/I4",
        ],
        [],
    ]
    assert all(isinstance(identifier, str) for identifier in result.loc[0, CURRENT_AFFILIATION_IDS])


def test_current_affiliation_set_is_used_to_check_early_and_last_institutions() -> None:
    author_match = pd.DataFrame(
        {
            "sc_id": [7],
            SELECTED_AUTHOR_IDS: [
                (
                    "['https://openalex.org/A2', 'https://openalex.org/A3', "
                    "'https://openalex.org/A4']"
                )
            ],
            CURRENT_AFFILIATION_IDS: [
                [
                    "https://openalex.org/I2",
                    "https://openalex.org/I3",
                    "https://openalex.org/I4",
                ]
            ],
        }
    )
    publications = pd.DataFrame(
        [
            {
                "eid": "early",
                "author_afids": "https://openalex.org/I9",
                "inst_key": "https://openalex.org/I9",
                "cover_date": pd.Timestamp("2010-01-01"),
            },
            {
                "eid": "last-current",
                "author_afids": "https://openalex.org/I3",
                "inst_key": "https://openalex.org/I3",
                "cover_date": pd.Timestamp("2020-01-01"),
            },
        ]
    )

    result = build_author_summary_row(
        "https://openalex.org/A2",
        author_match,
        publications,
        publications,
        enable_university_affiliation_inbreeding_detection=True,
        first_affiliation_publication_count=1,
    )

    assert result["university_affiliation_ids"] == [
        "https://openalex.org/I2",
        "https://openalex.org/I3",
        "https://openalex.org/I4",
    ]
    assert result["last_university_affiliation_overlap_ids"] == ["https://openalex.org/I3"]
    assert result["first_university_affiliation_overlap_ids"] == []
    assert result[IS_INBREEDING_PROXY] is False
    assert result[INBREEDING_DETECTION_STATUS] == (
        "assigned_first_affiliation_not_current_university"
    )


def test_openalex_current_affiliations_without_current_ids_require_manual_review() -> None:
    author_match = pd.DataFrame(
        {
            "sc_id": [8],
            SELECTED_AUTHOR_IDS: ["['https://openalex.org/A8']"],
            CURRENT_AFFILIATION_IDS: [[]],
        }
    )
    publications = pd.DataFrame(
        [
            {
                "eid": "early",
                "author_afids": "https://openalex.org/I8",
                "inst_key": "https://openalex.org/I8",
                "cover_date": pd.Timestamp("2010-01-01"),
            },
            {
                "eid": "last",
                "author_afids": "https://openalex.org/I9",
                "inst_key": "https://openalex.org/I9",
                "cover_date": pd.Timestamp("2020-01-01"),
            },
        ]
    )

    result = build_author_summary_row(
        "https://openalex.org/A8",
        author_match,
        publications,
        publications,
        enable_university_affiliation_inbreeding_detection=True,
        first_affiliation_publication_count=1,
    )

    assert pd.isna(result[IS_INBREEDING_PROXY])
    assert result[INBREEDING_DETECTION_STATUS] == ("manual_review_no_current_openalex_affiliation")


def test_openalex_works_reader_filters_candidates_and_preserves_authorship_ids(
    tmp_path: Path,
) -> None:
    works_path = tmp_path / "works.parquet"
    pd.DataFrame(
        {
            "id": [
                "https://openalex.org/W1",
                "https://openalex.org/W2",
                "https://openalex.org/W3",
            ],
            "publication_date": ["2020-01-02", "2021-02-03", "2022-03-04"],
            "authorships": [
                [
                    {
                        "author": {"id": "https://openalex.org/A2"},
                        "affiliations": [{"institution_ids": ["https://openalex.org/I2"]}],
                    },
                    {
                        "author": {"id": "https://openalex.org/A20"},
                        "affiliations": [
                            {
                                "institution_ids": [
                                    "https://openalex.org/I20",
                                    "https://openalex.org/I21",
                                ]
                            }
                        ],
                    },
                ],
                [
                    {
                        "author": {"id": "https://openalex.org/A20"},
                        "affiliations": [{"institution_ids": []}],
                    }
                ],
                [
                    {
                        "author": {"id": "https://openalex.org/A3"},
                        "affiliations": [{"institution_ids": ["https://openalex.org/I3"]}],
                    }
                ],
            ],
        }
    ).to_parquet(works_path, index=False)

    result = read_openalex_mobility_articles(
        works_path,
        {"https://openalex.org/A2", "https://openalex.org/A3"},
        max_workers=2,
    )

    assert result["eid"].tolist() == [
        "https://openalex.org/W1",
        "https://openalex.org/W3",
    ]
    assert result["author_ids"].tolist() == [
        "https://openalex.org/A2;https://openalex.org/A20",
        "https://openalex.org/A3",
    ]
    assert result["author_afids"].tolist() == [
        "https://openalex.org/I2;https://openalex.org/I20-https://openalex.org/I21",
        "https://openalex.org/I3",
    ]
    assert result.loc[0, "cover_date"] == "2020-01-02"


def test_openalex_institution_dimension_uses_direct_string_ids(tmp_path: Path) -> None:
    institutions_path = tmp_path / "institutions.parquet"
    pd.DataFrame(
        {
            "affiliation_id": ["https://openalex.org/I3"],
            "name": ["Example University"],
            "type": ["education"],
            "address_part": ["Main Street"],
            "city": ["Vienna"],
            "country": ["Austria"],
            "country_code": ["AT"],
        }
    ).to_parquet(institutions_path, index=False)

    institutions = read_openalex_mobility_affiliations(institutions_path)
    validated = MobilityAffiliationSchema.validate(institutions, lazy=True)

    assert validated.loc[0, "affiliation_id"] == "https://openalex.org/I3"
    assert validated.loc[0, "preferred_name"] == "Example University"
    assert "affiliation_id_parent" not in institutions.columns


def test_enabled_professor_filter_limits_openalex_work_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authors_path = tmp_path / "selection.csv"
    metadata_path = tmp_path / "resolved.csv"
    works_path = tmp_path / "works.parquet"
    institutions_path = tmp_path / "institutions.parquet"
    candidate_pairs_path = tmp_path / "pairs.parquet"
    for path in (
        authors_path,
        metadata_path,
        works_path,
        institutions_path,
        candidate_pairs_path,
    ):
        path.touch()

    authors = pd.DataFrame(
        {
            "sc_id": [1, 2],
            "status": ["assigned_single_candidate", "assigned_single_candidate"],
            "selected_disambiguated_cluster_id": [101, 102],
            SELECTED_AUTHOR_IDS: [
                ["https://openalex.org/A2"],
                ["https://openalex.org/A3"],
            ],
            "possible_clusters": [[], []],
            "inferred_asjc_codes": [[], []],
            "scopus_asjc_codes": [[], []],
            "inferred_asjc_labels": [[], []],
            "reason": ["selected", "selected"],
        }
    )
    metadata = pd.DataFrame(
        {
            "sc_id": [1, 2],
            "author_id": ["https://openalex.org/A2", "https://openalex.org/A3"],
            "priority": [1, 1],
            "person_title": ["Prof. Dr.", "Dr."],
            "university_name": ["Example University", "Example University"],
        }
    )
    frames = {authors_path: authors, metadata_path: metadata}
    monkeypatch.setattr(
        "acib.mobility.input._read_mobility_dataframe",
        lambda path, **_kwargs: frames[path].copy(),
    )
    monkeypatch.setattr(
        "acib.mobility.input.attach_openalex_current_affiliations",
        lambda frame, _path: frame.assign(**{CURRENT_AFFILIATION_IDS: [[], []]}),
    )
    monkeypatch.setattr(
        "acib.mobility.input.read_openalex_mobility_affiliations",
        lambda _path: pd.DataFrame({"affiliation_id": ["https://openalex.org/I2"]}),
    )
    observed_candidate_ids: list[set[str]] = []
    observed_worker_counts: list[int] = []

    def record_candidate_ids(
        _path: Path,
        candidate_ids: set[str],
        *,
        max_workers: int,
    ) -> pd.DataFrame:
        observed_candidate_ids.append(candidate_ids)
        observed_worker_counts.append(max_workers)
        return pd.DataFrame(columns=["eid", "cover_date", "author_ids", "author_afids"])

    monkeypatch.setattr(
        "acib.mobility.input.read_openalex_mobility_articles",
        record_candidate_ids,
    )
    config = MobilityConfig(
        authors=authors_path,
        researcher_metadata=metadata_path,
        article_author=works_path,
        affiliations=institutions_path,
        candidate_authors=candidate_pairs_path,
        outputs=MobilityOutputs(directory=tmp_path / "outputs"),
        source="openalex",
        enable_coaffiliated_professor_merge=True,
        openalex_scan_workers=4,
    )

    loaded = load_mobility_inputs(config)

    assert observed_candidate_ids == [{"https://openalex.org/A2"}]
    assert observed_worker_counts == [4]
    assert len(loaded.authors) == 2
