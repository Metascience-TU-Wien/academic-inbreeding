# Copyright © 2026 TU Wien

"""Resolved name-matching output contract."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from acib.constants import (
    AFFILIATION_ID,
    AFFILIATION_ID_PARENT,
    AFFILIATION_ID_PARENT_CHATGPT,
    AFFILIATION_ID_PARENT_SPLINK_CHATGPT,
    AFFILIATION_IDS_CURRENT,
    AFFILIATION_NAME,
    AFFILIATION_PARENT_SOURCE,
    AFFILIATION_UMBRELLA_ID,
    AUTHOR_ID,
    EID,
    GIVEN_NAME,
    HISTORICAL_IDENTIFIER,
    RESEARCHER_ID,
    SUBJECT_AREAS,
    SURNAME,
)
from acib.name_matching.constants import (
    AUTHOR_NAME,
    INDEXED_NAME,
    INITIALS,
    MATCH_PRIORITY,
    MATCH_PRIORITY_VALUES,
    NAME,
    NAME_VARIANTS,
    PERSON_FULLNAME,
    PUBLICATION_RANGE,
    RESEARCHER_GIVEN_NAME,
    RESEARCHER_NAME,
    RESEARCHER_SURNAME,
)


class MatchingOutputSchema(pa.DataFrameModel):
    """Validate the resolved researcher-to-Scopus-author interface.

    Multiple output rows may share either ID. Passthrough rows may have null
    candidate metadata, while every row must link a researcher, an author ID,
    and a valid strategy priority.
    """

    sc_id: Series[int] = pa.Field(alias=RESEARCHER_ID, nullable=False)
    author_id: Series[int] = pa.Field(alias=AUTHOR_ID, nullable=False, coerce=True)
    priority: Series[int] = pa.Field(
        alias=MATCH_PRIORITY,
        nullable=False,
        isin=sorted(MATCH_PRIORITY_VALUES),
    )

    eid: Series[str] = pa.Field(alias=EID, nullable=True)
    researcher_name: Series[str] = pa.Field(alias=RESEARCHER_NAME, nullable=True)
    author_name: Series[str] = pa.Field(alias=AUTHOR_NAME, nullable=True)
    name: Series[str] = pa.Field(alias=NAME, nullable=True)
    indexed_name: Series[str] = pa.Field(alias=INDEXED_NAME, nullable=True)
    initials: Series[str] = pa.Field(alias=INITIALS, nullable=True)
    given_name: Series[str] = pa.Field(alias=GIVEN_NAME, nullable=True)
    surname: Series[str] = pa.Field(alias=SURNAME, nullable=True)
    publication_range: Series[str] = pa.Field(alias=PUBLICATION_RANGE, nullable=True)
    name_variants: Series[str] = pa.Field(alias=NAME_VARIANTS, nullable=True)
    subject_areas: Series[str] = pa.Field(alias=SUBJECT_AREAS, nullable=True)
    historical_identifier: Series[str] | None = pa.Field(alias=HISTORICAL_IDENTIFIER, nullable=True)
    affiliation_id: Series[pd.Int64Dtype] | None = pa.Field(
        alias=AFFILIATION_ID, nullable=True, coerce=True
    )
    affiliation_id_parent: Series[pd.Int64Dtype] | None = pa.Field(
        alias=AFFILIATION_ID_PARENT, nullable=True, coerce=True
    )
    affiliation_id_parent_chatgpt: Series[pd.Int64Dtype] | None = pa.Field(
        alias=AFFILIATION_ID_PARENT_CHATGPT, nullable=True, coerce=True
    )
    affiliation_id_parent_splink_chatgpt: Series[pd.Int64Dtype] | None = pa.Field(
        alias=AFFILIATION_ID_PARENT_SPLINK_CHATGPT, nullable=True, coerce=True
    )
    parent: Series[pd.Int64Dtype] | None = pa.Field(
        alias=AFFILIATION_PARENT_SOURCE, nullable=True, coerce=True
    )
    umbrella_id: Series[pd.Int64Dtype] | None = pa.Field(
        alias=AFFILIATION_UMBRELLA_ID, nullable=True, coerce=True
    )
    affiliation_name: Series[str] | None = pa.Field(alias=AFFILIATION_NAME, nullable=True)
    affiliation_ids_current: Series[Any] | None = pa.Field(
        alias=AFFILIATION_IDS_CURRENT, nullable=True
    )
    person_name: Series[str] | None = pa.Field(alias=RESEARCHER_GIVEN_NAME, nullable=True)
    person_surname: Series[str] | None = pa.Field(alias=RESEARCHER_SURNAME, nullable=True)
    person_fullname: Series[str] | None = pa.Field(alias=PERSON_FULLNAME, nullable=True)

    class Config:
        """Preserve all other source researcher metadata in the output."""

        strict = False
        coerce = False
