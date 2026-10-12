# Copyright © 2026 TU Wien

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from acib.constants import AFFILIATION_ID, AFFILIATION_ID_PARENT, AUTHOR_ID
from acib.schemas.affiliations import (
    AuthorAffiliationSchema,
    parse_nullable_affiliation_ids,
)


def test_parent_id_parser_preserves_integer_precision_and_nullable_dtype() -> None:
    raw = pd.Series(
        ["60024895.0", None, "9007199254740993.0"],
        name=AFFILIATION_ID_PARENT,
        dtype="string",
    )

    normalized = parse_nullable_affiliation_ids(raw)

    assert normalized.dtype == "Int64"
    assert normalized.tolist() == [60024895, pd.NA, 9007199254740993]


@pytest.mark.parametrize(
    "invalid_value",
    ["60024895.5", "not-an-id", "1e6", "9223372036854775808"],
)
def test_parent_id_parser_rejects_fractional_ambiguous_and_overflow_values(
    invalid_value: str,
) -> None:
    with pytest.raises(ValueError, match="(?i)affiliation identifier"):
        parse_nullable_affiliation_ids(pd.Series([invalid_value]))


def test_author_affiliation_schema_allows_repeated_authors_and_absent_parent_fields() -> None:
    affiliations = pd.DataFrame(
        {
            AUTHOR_ID: pd.Series([10, 10], dtype="int64"),
            AFFILIATION_ID: pd.Series([60024895, 60023270], dtype="int64"),
            "current_affiliation": [True, False],
            "raw_affiliation_metadata": ["university", "department"],
        }
    )

    validated = AuthorAffiliationSchema.validate(affiliations, lazy=True)

    assert validated[AUTHOR_ID].tolist() == [10, 10]
    assert validated.columns.tolist() == affiliations.columns.tolist()


@pytest.mark.parametrize(
    "affiliations",
    [
        pd.DataFrame({AFFILIATION_ID: pd.Series([60024895], dtype="int64")}),
        pd.DataFrame(
            {
                AUTHOR_ID: pd.Series([10], dtype="int64"),
                AFFILIATION_ID: pd.Series([pd.NA], dtype="Int64"),
            }
        ),
    ],
    ids=["missing-author-id", "null-affiliation-id"],
)
def test_author_affiliation_schema_rejects_missing_required_affiliation_keys(
    affiliations: pd.DataFrame,
) -> None:
    with pytest.raises(pa.errors.SchemaErrors):
        AuthorAffiliationSchema.validate(affiliations, lazy=True)
