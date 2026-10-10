# Copyright © 2026 TU Wien

"""Researcher input DataFrame contracts."""

from __future__ import annotations

import math

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from acib.constants import RESEARCHER_ID, UNIVERSITY_NAME
from acib.name_matching.constants import (
    NAME,
    PERSON_FULLNAME,
    RESEARCHER_GIVEN_NAME,
    RESEARCHER_NAME,
    RESEARCHER_SURNAME,
    SCOPUS_AUTHOR_ID_INPUT,
)


class ResearcherInputSchema(pa.DataFrameModel):
    """Validate researcher identity and the available name-source columns.

    Unmodeled researcher metadata is retained and permitted.
    """

    sc_id: Series[int] = pa.Field(alias=RESEARCHER_ID, nullable=False, unique=True)
    university_name: Series[str] | None = pa.Field(alias=UNIVERSITY_NAME, nullable=True)
    person_name: Series[str] | None = pa.Field(alias=RESEARCHER_GIVEN_NAME, nullable=True)
    person_surname: Series[str] | None = pa.Field(alias=RESEARCHER_SURNAME, nullable=True)
    person_fullname: Series[str] | None = pa.Field(alias=PERSON_FULLNAME, nullable=True)
    researcher_name: Series[str] | None = pa.Field(alias=RESEARCHER_NAME, nullable=True)
    name: Series[str] | None = pa.Field(alias=NAME, nullable=True)
    scopus_author_id: Series[float] | None = pa.Field(alias=SCOPUS_AUTHOR_ID_INPUT, nullable=True)

    @pa.dataframe_check
    @classmethod
    def has_name_source(cls, frame: pd.DataFrame) -> bool:
        """Require at least one name representation the matcher can consume."""
        columns = set(frame.columns)
        return (
            RESEARCHER_NAME in columns
            or {RESEARCHER_GIVEN_NAME, RESEARCHER_SURNAME}.issubset(columns)
            or PERSON_FULLNAME in columns
            or NAME in columns
        )

    @pa.check("scopus_author_id")
    @classmethod
    def existing_scopus_ids_are_exact_integers(cls, values: pd.Series) -> pd.Series:
        """Reject fractional, non-finite, or float-unsafe Scopus identifiers."""
        return values.map(
            lambda value: pd.isna(value)
            or (
                math.isfinite(float(value))
                and float(value).is_integer()
                and abs(float(value)) < 2**53
            )
        )

    class Config:
        """Allow additional researcher metadata and disable implicit coercion."""

        strict = False
        coerce = False
