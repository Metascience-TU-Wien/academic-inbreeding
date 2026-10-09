# Copyright © 2026 TU Wien

from __future__ import annotations

import pandas as pd
import pytest

from acib.constants import EID, RESEARCHER_ID, UNIVERSITY_NAME
from acib.name_matching.matching import (
    find_fuzzy_matches,
    match_researchers,
    names_match,
    normalize_name,
    simple_name_merge,
)


def test_simple_name_merge_preserves_strategy_priorities(
    researcher_names: pd.DataFrame, author_names: pd.DataFrame
) -> None:
    matches = simple_name_merge(researcher_names, author_names)

    expected = {1: ("P1", 1), 2: ("P2", 2), 3: ("P3", 3), 4: ("P4", 4)}
    for researcher_id, (eid, priority) in expected.items():
        accepted = matches.loc[matches[RESEARCHER_ID].eq(researcher_id) & matches[EID].eq(eid)]
        assert len(accepted) == 1
        assert accepted.iloc[0]["priority"] == priority


@pytest.mark.parametrize(
    ("name", "expected_tokens"),
    [
        ("  Dr. Anna-Marie Smith  ", ["anna", "marie", "smith"]),
        ("Prof. van der Meer", ["meer"]),
        ("", []),
        ("Dr. Ing.", []),
    ],
)
def test_normalize_name_filters_short_tokens_and_stopwords(
    name: str, expected_tokens: list[str]
) -> None:
    assert normalize_name(name) == expected_tokens


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("smith anna", "anna smith", True),
        ("smith anna", "smith anita", False),
        ("smith", "smith", False),
        ("", "smith anna", False),
    ],
)
def test_names_match_applies_token_guards(left: str, right: str, expected: bool) -> None:
    matched, _ = names_match(left, right)
    assert matched is expected


def test_find_fuzzy_matches_keeps_duplicate_author_rows() -> None:
    authors = pd.DataFrame({"name": ["John Doe", "John Doe", "Jane Doe"]})

    assert find_fuzzy_matches("Doe John", authors, "name") == ["John Doe", "John Doe"]


def test_organization_filter_restricts_authors_to_mapped_affiliations() -> None:
    researchers = pd.DataFrame(
        {
            RESEARCHER_ID: [1, 2],
            "person_name": ["Anna", "Nobody"],
            "person_surname": ["Smith", "Matches"],
            UNIVERSITY_NAME: ["TU Wien", "TU Wien"],
        }
    )
    authors = pd.DataFrame(
        {
            "author_id": [10, 20],
            EID: ["correct", "wrong"],
            "author_name": ["smith, anna", "smith, anna"],
            "affiliation_id": [60018163, 60025988],
        }
    )

    filtered = match_researchers(
        researchers,
        authors,
        org_mapping={"TU Wien": [60018163]},
        enable_organisation_filter=True,
    )
    unfiltered = match_researchers(
        researchers,
        authors,
        org_mapping={"TU Wien": [60018163]},
        enable_organisation_filter=False,
    )

    assert filtered.matched[EID].tolist() == ["correct"]
    assert set(unfiltered.matched[EID]) == {"correct", "wrong"}


def test_researcher_projection_does_not_require_university_when_filter_disabled() -> None:
    researchers = pd.DataFrame(
        {
            RESEARCHER_ID: [1],
            "person_name": ["Anna"],
            "person_surname": ["Smith"],
        }
    )
    authors = pd.DataFrame({"author_id": [10], EID: ["P1"], "author_name": ["smith, anna"]})

    result = match_researchers(
        researchers,
        authors,
        enable_organisation_filter=False,
    )

    assert result.matched[RESEARCHER_ID].tolist() == [1]


def test_organization_filter_handles_all_researchers_matched() -> None:
    researchers = pd.DataFrame(
        {
            RESEARCHER_ID: [1],
            "person_name": ["Anna"],
            "person_surname": ["Smith"],
            UNIVERSITY_NAME: ["TU Wien"],
        }
    )
    authors = pd.DataFrame(
        {
            "author_id": [10],
            EID: ["P1"],
            "author_name": ["smith, anna"],
            "affiliation_id": [60018163],
        }
    )

    result = match_researchers(
        researchers,
        authors,
        org_mapping={"TU Wien": [60018163]},
        enable_organisation_filter=True,
    )

    assert result.matched[RESEARCHER_ID].tolist() == [1]
    assert result.unmatched.empty
