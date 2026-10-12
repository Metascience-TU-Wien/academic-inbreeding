# Copyright © 2026 TU Wien

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from acib.constants import AFFILIATION_ID, AUTHOR_ID, EID, GIVEN_NAME, SURNAME
from acib.name_matching.constants import INDEXED_NAME, NAME, NAME_VARIANTS
from acib.schemas.authors import ScopusAuthorInputSchema


def _candidate_authors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            AUTHOR_ID: pd.Series([10, 10], dtype="int64"),
            AFFILIATION_ID: pd.Series([60024895, 60023270], dtype="int64"),
            EID: ["P1", None],
            NAME: ["Smith, Anna", None],
            INDEXED_NAME: ["Smith A.", None],
            "initials": ["A.", None],
            GIVEN_NAME: ["Anna", None],
            SURNAME: ["Smith", None],
            "publication_range": ["2018-2022", None],
            NAME_VARIANTS: ["[]", None],
            "subject_areas": ["[]", None],
            "extra_candidate_metadata": ["keep", "also-keep"],
        }
    )


def test_scopus_author_schema_accepts_nullable_details_and_duplicate_profiles() -> None:
    authors = _candidate_authors()

    validated = ScopusAuthorInputSchema.validate(authors, lazy=True)

    assert validated[AUTHOR_ID].tolist() == [10, 10]
    assert validated[EID].isna().sum() == 1
    assert validated["extra_candidate_metadata"].tolist() == ["keep", "also-keep"]


def test_scopus_author_schema_requires_at_least_one_name_route() -> None:
    authors = _candidate_authors().drop(columns=[NAME, GIVEN_NAME, SURNAME])

    with pytest.raises(pa.errors.SchemaErrors):
        ScopusAuthorInputSchema.validate(authors, lazy=True)


def test_scopus_author_schema_requires_downstream_columns_and_affiliation_keys() -> None:
    authors = _candidate_authors().drop(columns=["publication_range"])

    with pytest.raises(pa.errors.SchemaErrors):
        ScopusAuthorInputSchema.validate(authors, lazy=True)


def test_scopus_author_schema_rejects_string_author_ids() -> None:
    authors = _candidate_authors()
    authors[AUTHOR_ID] = authors[AUTHOR_ID].astype(str)

    with pytest.raises(pa.errors.SchemaErrors):
        ScopusAuthorInputSchema.validate(authors, lazy=True)


def test_scopus_author_schema_accepts_empty_typed_input() -> None:
    authors = _candidate_authors().iloc[:0].copy()

    validated = ScopusAuthorInputSchema.validate(authors, lazy=True)

    assert validated.empty
