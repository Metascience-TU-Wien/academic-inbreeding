# Copyright © 2026 TU Wien

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pandera.pandas as pa
import pytest

from acib.constants import AUTHOR_ID, EID, RESEARCHER_ID
from acib.name_matching.constants import MATCH_PRIORITY
from acib.schemas.matching import MatchingOutputSchema


def _matching_output() -> pd.DataFrame:
    return pd.DataFrame(
        {
            RESEARCHER_ID: pd.Series([10, 10], dtype="int64"),
            AUTHOR_ID: ["9007199254740993", 12],
            MATCH_PRIORITY: [1, 3],
            EID: ["P1", "P2"],
            "researcher_name": ["smith, anna", "smith, anna"],
            "author_name": ["smith, anna", "smith, a."],
            "name": ["Smith, Anna", "Smith, Anna"],
            "indexed_name": ["Smith A.", "Smith A."],
            "initials": ["A.", "A."],
            "given_name": ["Anna", "Anna"],
            "surname": ["Smith", "Smith"],
            "publication_range": ["2018-2022", "2018-2022"],
            "name_variants": ["[]", "[]"],
            "subject_areas": ["[]", "[]"],
            "person_name": ["Anna", "Anna"],
            "person_surname": ["Smith", "Smith"],
            "person_fullname": ["Anna Smith", "Anna Smith"],
            "extra_researcher_metadata": ["preserved", "preserved"],
        }
    )


def test_matching_output_coerces_numeric_string_ids_without_precision_loss() -> None:
    result = MatchingOutputSchema.validate(_matching_output(), lazy=True)

    assert result[AUTHOR_ID].tolist() == [9007199254740993, 12]
    assert result[AUTHOR_ID].dtype == "int64"
    assert result[RESEARCHER_ID].tolist() == [10, 10]
    assert result["extra_researcher_metadata"].tolist() == ["preserved", "preserved"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: frame.drop(columns=[RESEARCHER_ID]),
        lambda frame: frame.assign(**{AUTHOR_ID: ["not-an-id", "12"]}),
        lambda frame: frame.assign(**{MATCH_PRIORITY: [5, 1]}),
        lambda frame: frame.assign(**{RESEARCHER_ID: [10, pd.NA]}),
    ],
    ids=["missing-researcher-id", "bad-author-id", "bad-priority", "null-researcher-id"],
)
def test_matching_output_rejects_invalid_required_values(
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    with pytest.raises(pa.errors.SchemaErrors):
        MatchingOutputSchema.validate(mutate(_matching_output()), lazy=True)


def test_matching_output_accepts_nullable_passthrough_fields_and_empty_frames() -> None:
    passthrough = _matching_output().iloc[:1].copy()
    passthrough.loc[:, EID] = pd.NA
    passthrough.loc[:, "researcher_name"] = pd.NA
    passthrough.loc[:, "author_name"] = pd.NA
    validated = MatchingOutputSchema.validate(passthrough, lazy=True)
    assert validated.loc[0, EID] is pd.NA

    empty = _matching_output().iloc[:0].copy()
    assert MatchingOutputSchema.validate(empty, lazy=True).empty
