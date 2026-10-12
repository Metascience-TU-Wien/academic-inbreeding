# Copyright © 2026 TU Wien

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from acib.constants import RESEARCHER_ID, UNIVERSITY_NAME
from acib.name_matching.constants import (
    PERSON_FULLNAME,
    RESEARCHER_GIVEN_NAME,
    RESEARCHER_SURNAME,
    SCOPUS_AUTHOR_ID_INPUT,
)
from acib.schemas.researchers import ResearcherInputSchema


def _researchers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            RESEARCHER_ID: pd.Series([10, 11], dtype="int64"),
            UNIVERSITY_NAME: ["TU Wien", "BOKU Wien"],
            RESEARCHER_GIVEN_NAME: ["Anna", "Nina"],
            RESEARCHER_SURNAME: ["Smith", "Example"],
            PERSON_FULLNAME: ["Anna Smith", "Nina Example"],
            SCOPUS_AUTHOR_ID_INPUT: pd.Series([np.nan, 123456789.0], dtype="float64"),
            "arbitrary_metadata": ["keep-a", {"keep": "b"}],
        }
    )


def test_researcher_schema_preserves_optional_columns_and_extra_metadata() -> None:
    researchers = _researchers()

    validated = ResearcherInputSchema.validate(researchers, lazy=True)

    assert validated.columns.tolist() == researchers.columns.tolist()
    pd.testing.assert_frame_equal(validated, researchers)


def test_researcher_schema_allows_optional_name_and_mapping_columns_to_be_absent() -> None:
    researchers = pd.DataFrame(
        {
            RESEARCHER_ID: pd.Series([10], dtype="int64"),
            RESEARCHER_GIVEN_NAME: ["Anna"],
            RESEARCHER_SURNAME: ["Smith"],
        }
    )

    validated = ResearcherInputSchema.validate(researchers, lazy=True)

    assert validated.columns.tolist() == researchers.columns.tolist()


def test_researcher_schema_rejects_missing_or_duplicate_researcher_ids() -> None:
    missing = _researchers().drop(columns=RESEARCHER_ID)
    duplicate = _researchers().assign(**{RESEARCHER_ID: [10, 10]})

    with pytest.raises(pa.errors.SchemaErrors):
        ResearcherInputSchema.validate(missing, lazy=True)
    with pytest.raises(pa.errors.SchemaErrors):
        ResearcherInputSchema.validate(duplicate, lazy=True)


def test_researcher_schema_rejects_null_ids_and_missing_name_routes() -> None:
    null_id = _researchers().astype({RESEARCHER_ID: "Int64"})
    null_id.loc[0, RESEARCHER_ID] = pd.NA
    no_name_route = pd.DataFrame({RESEARCHER_ID: pd.Series([1], dtype="int64")})

    with pytest.raises(pa.errors.SchemaErrors):
        ResearcherInputSchema.validate(null_id, lazy=True)
    with pytest.raises(pa.errors.SchemaErrors):
        ResearcherInputSchema.validate(no_name_route, lazy=True)


@pytest.mark.parametrize("invalid_id", [123.5, np.inf, float(2**53)])
def test_researcher_schema_rejects_unsafe_existing_scopus_ids(invalid_id: float) -> None:
    researchers = _researchers()
    researchers.loc[0, SCOPUS_AUTHOR_ID_INPUT] = invalid_id

    with pytest.raises(pa.errors.SchemaErrors):
        ResearcherInputSchema.validate(researchers, lazy=True)


def test_researcher_schema_accepts_a_typed_empty_frame() -> None:
    empty = pd.DataFrame(
        {
            RESEARCHER_ID: pd.Series(dtype="int64"),
            RESEARCHER_GIVEN_NAME: pd.Series(dtype="object"),
            RESEARCHER_SURNAME: pd.Series(dtype="object"),
        }
    )

    validated = ResearcherInputSchema.validate(empty, lazy=True)

    assert validated.empty
