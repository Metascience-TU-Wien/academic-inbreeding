# Copyright © 2026 TU Wien

"""Scopus author-candidate DataFrame contracts."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from acib.constants import (
    EID,
    GIVEN_NAME,
    HISTORICAL_IDENTIFIER,
    SUBJECT_AREAS,
    SURNAME,
)
from acib.name_matching.constants import (
    AUTHOR_NAME,
    INDEXED_NAME,
    INITIALS,
    NAME,
    NAME_VARIANTS,
    PUBLICATION_RANGE,
)
from acib.schemas.affiliations import AuthorAffiliationSchema


class ScopusAuthorInputSchema(AuthorAffiliationSchema):
    """Validate Scopus candidate columns used here and by disambiguation.

    The EID and author-detail columns are structurally required by downstream
    disambiguation, but their values may be missing in the source records.
    """

    eid: Series[str] = pa.Field(alias=EID, nullable=True)
    author_name: Series[str] | None = pa.Field(alias=AUTHOR_NAME, nullable=True)
    name: Series[str] | None = pa.Field(alias=NAME, nullable=True)
    indexed_name: Series[str] = pa.Field(alias=INDEXED_NAME, nullable=True)
    initials: Series[str] = pa.Field(alias=INITIALS, nullable=True)
    given_name: Series[str] | None = pa.Field(alias=GIVEN_NAME, nullable=True)
    surname: Series[str] | None = pa.Field(alias=SURNAME, nullable=True)
    publication_range: Series[str] = pa.Field(alias=PUBLICATION_RANGE, nullable=True)
    name_variants: Series[str] = pa.Field(alias=NAME_VARIANTS, nullable=True)
    subject_areas: Series[str] = pa.Field(alias=SUBJECT_AREAS, nullable=True)
    historical_identifier: Series[str] | None = pa.Field(alias=HISTORICAL_IDENTIFIER, nullable=True)

    @pa.dataframe_check
    @classmethod
    def has_author_name_source(cls, frame: pd.DataFrame) -> bool:
        """Require an author name or a given-name/surname pair."""
        columns = set(frame.columns)
        return AUTHOR_NAME in columns or NAME in columns or {GIVEN_NAME, SURNAME}.issubset(columns)

    class Config:
        """Allow extra Scopus metadata without implicit coercion."""

        strict = False
        coerce = False
