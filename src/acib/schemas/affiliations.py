# Copyright © 2026 TU Wien

"""Author-affiliation input contracts and lossless ID parsing."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from acib.constants import (
    AFFILIATION_ID,
    AFFILIATION_ID_PARENT,
    AFFILIATION_ID_PARENT_CHATGPT,
    AFFILIATION_ID_PARENT_SPLINK_CHATGPT,
    AFFILIATION_PARENT_SOURCE,
    AFFILIATION_UMBRELLA_ID,
    AUTHOR_ID,
)
from acib.name_matching.constants import CURRENT_AFFILIATION

_INTEGRAL_AFFILIATION_ID = re.compile(r"[+-]?\d+(?:\.0+)?\Z")
_NULL_AFFILIATION_VALUES = {"", "nan", "none", "null", "<na>"}
_MIN_INT64 = -(2**63)
_MAX_INT64 = 2**63 - 1


def parse_nullable_affiliation_ids(values: pd.Series) -> pd.Series:
    """Parse affiliation IDs into pandas nullable integers without float conversion.

    CSV integer spellings with a zero-only decimal suffix (such as ``"60024895.0"``)
    are accepted. Fractional, non-finite, malformed, and out-of-range values are rejected.

    :param values: Raw affiliation identifier values.
    :return: Nullable 64-bit integer identifiers with the original index and name.
    :raises ValueError: If a non-null value is not an exact signed 64-bit integer.
    """
    text_values = values.astype("string").str.strip()
    missing = text_values.isna() | text_values.str.casefold().isin(_NULL_AFFILIATION_VALUES)
    valid_format = text_values.str.fullmatch(_INTEGRAL_AFFILIATION_ID).fillna(False)
    invalid = ~missing & ~valid_format
    if invalid.any():
        examples = text_values.loc[invalid].head(10).tolist()
        raise ValueError(f"Invalid affiliation identifier values: {examples!r}")

    integer_text = text_values.mask(missing).str.replace(r"\.0+\Z", "", regex=True)
    try:
        parsed = integer_text.astype("Int64")
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("Affiliation identifiers must fit in signed 64-bit integers") from exc
    parsed.name = values.name
    return parsed


class AuthorAffiliationSchema(pa.DataFrameModel):
    """Validate the repeated Scopus author-affiliation relationship rows.

    Author IDs are not unique: Scopus input contains one row per profile affiliation.
    Parent identifiers may be absent, and additional affiliation metadata is preserved.
    """

    author_id: Series[int] = pa.Field(alias=AUTHOR_ID, nullable=False)
    affiliation_id: Series[int] = pa.Field(alias=AFFILIATION_ID, nullable=False)
    affiliation_id_parent: Series[pd.Int64Dtype] | None = pa.Field(
        alias=AFFILIATION_ID_PARENT, nullable=True
    )
    affiliation_id_parent_chatgpt: Series[pd.Int64Dtype] | None = pa.Field(
        alias=AFFILIATION_ID_PARENT_CHATGPT, nullable=True
    )
    affiliation_id_parent_splink_chatgpt: Series[pd.Int64Dtype] | None = pa.Field(
        alias=AFFILIATION_ID_PARENT_SPLINK_CHATGPT, nullable=True
    )
    parent: Series[pd.Int64Dtype] | None = pa.Field(alias=AFFILIATION_PARENT_SOURCE, nullable=True)
    umbrella_id: Series[pd.Int64Dtype] | None = pa.Field(
        alias=AFFILIATION_UMBRELLA_ID, nullable=True
    )
    current_affiliation: Series[Any] | None = pa.Field(alias=CURRENT_AFFILIATION, nullable=True)

    class Config:
        """Allow unrelated metadata and require explicit identifier normalization."""

        strict = False
        coerce = False
