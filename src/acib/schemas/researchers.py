# Copyright © 2026 TU Wien

"""Researcher input DataFrame contracts."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from acib.constants import (
    INFERRED_ASJC_CODES,
    PERSON_TITLE,
    RESEARCHER_ID,
    UNIVERSITY_NAME,
)
from acib.name_disambiguation.constants import (
    CURRENT_AFFILIATION_IDS,
    FINAL_SELECTION_STATUS_VALUES,
    INFERRED_ASJC_LABELS,
    POSSIBLE_CLUSTER_HISTORY,
    POSSIBLE_CLUSTERS,
    SCOPUS_ASJC_CODES,
    SELECTED_AUTHOR_IDS,
    SELECTED_CLUSTER_ID,
    SELECTED_DISAMBIGUATED_CLUSTER_ID,
    SELECTED_HISTORY,
    SELECTION_REASON,
    SELECTION_STATUS,
)
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


class ResearcherDisambiguationSelectionSchema(pa.DataFrameModel):
    """Validate final disambiguation rows without coercing identifier values.

    The CSV boundary may infer Scopus researcher IDs as integers or floats with
    missing values; exact integral representations and digit-only strings are
    accepted and retained as supplied.
    """

    sc_id: Series[Any] = pa.Field(alias=RESEARCHER_ID, nullable=False, unique=True)
    status: Series[str] = pa.Field(
        alias=SELECTION_STATUS,
        nullable=False,
        isin=sorted(FINAL_SELECTION_STATUS_VALUES),
    )
    selected_cluster_id: Series[Any] | None = pa.Field(alias=SELECTED_CLUSTER_ID, nullable=True)
    selected_disambiguated_cluster_id: Series[Any] = pa.Field(
        alias=SELECTED_DISAMBIGUATED_CLUSTER_ID, nullable=True
    )
    selected_author_ids: Series[object] = pa.Field(alias=SELECTED_AUTHOR_IDS, nullable=False)
    possible_clusters: Series[object] = pa.Field(alias=POSSIBLE_CLUSTERS, nullable=False)
    inferred_asjc_codes: Series[object] = pa.Field(alias=INFERRED_ASJC_CODES, nullable=False)
    scopus_asjc_codes: Series[object] = pa.Field(alias=SCOPUS_ASJC_CODES, nullable=False)
    inferred_asjc_labels: Series[object] = pa.Field(alias=INFERRED_ASJC_LABELS, nullable=False)
    reason: Series[str] = pa.Field(alias=SELECTION_REASON, nullable=False)
    researcher_name: Series[str] | None = pa.Field(alias=RESEARCHER_NAME, nullable=True)
    person_title: Series[str] | None = pa.Field(alias=PERSON_TITLE, nullable=True)
    university_name: Series[str] | None = pa.Field(alias=UNIVERSITY_NAME, nullable=True)
    selected_history: Series[object] | None = pa.Field(alias=SELECTED_HISTORY, nullable=True)
    possible_cluster_history: Series[object] | None = pa.Field(
        alias=POSSIBLE_CLUSTER_HISTORY, nullable=True
    )
    current_affiliation_ids: Series[object] | None = pa.Field(
        alias=CURRENT_AFFILIATION_IDS, nullable=True
    )

    @pa.check(RESEARCHER_ID)
    @classmethod
    def researcher_ids_are_integral(cls, values: pd.Series) -> pd.Series:
        """Reject malformed researcher identifiers without changing their dtype.

        :param values: Researcher identifiers from the selection artifact.
        :return: Row-wise identifier validity mask.
        """
        return values.map(_is_valid_identifier)

    @pa.check(SELECTED_DISAMBIGUATED_CLUSTER_ID)
    @classmethod
    def selected_author_ids_are_integral(cls, values: pd.Series) -> pd.Series:
        """Allow missing selected IDs but reject fractional or malformed values.

        :param values: Selected Scopus author identifiers.
        :return: Row-wise identifier validity mask.
        """
        return values.map(
            lambda value: pd.isna(value)
            or (isinstance(value, str) and not value.strip())
            or _is_valid_identifier(value)
        )

    class Config:
        """Retain output enrichments not required by the core selection contract."""

        strict = False
        coerce = False


def _is_valid_identifier(value: Any) -> bool:
    """Check that an identifier is a nonnegative integral scalar or digit string.

    :param value: Identifier value to inspect.
    :return: Whether the value is an exact integral identifier.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, Integral):
        return int(value) >= 0
    if isinstance(value, float):
        return math.isfinite(value) and value.is_integer() and value >= 0
    return isinstance(value, str) and value.strip().isdigit()
