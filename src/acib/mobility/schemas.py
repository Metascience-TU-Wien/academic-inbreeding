# Copyright © 2026 TU Wien

"""Stage-local contracts for mobility publication and affiliation inputs."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from acib.constants import (
    AFFILIATION_ID,
    AFFILIATION_ID_PARENT,
    AFFILIATION_ID_PARENT_CHATGPT,
    AFFILIATION_ID_PARENT_SPLINK_CHATGPT,
    AFFILIATION_NAME,
    AFFILIATION_PARENT_SOURCE,
    AFFILIATION_UMBRELLA_ID,
    EID,
)
from acib.mobility.constants import (
    AFFILIATION_ADDRESS,
    AFFILIATION_PREFERRED_NAME,
    AFFILIATION_TYPE,
    ARTICLE_AFFILIATION_ID,
    ARTICLE_COVER_DATE,
    AUTHOR_AFFILIATION_TYPE,
    CITY,
    COUNTRY,
    IS_INBREEDING_PROXY,
    UNMAPPED_AFFILIATION_ID,
)
from acib.name_disambiguation.constants import AUTHOR_IDS

_OPENALEX_INSTITUTION_ID_PATTERN = re.compile(r"^https://openalex\.org/I\d+$")


class MobilityArticleAuthorSchema(pa.DataFrameModel):
    """Validate article-author-affiliation rows used by mobility processing.

    Unlike the disambiguation evidence contract, this input must preserve each
    publication's affiliation list alongside its author list and cover date.
    Additional article metadata remains available to the mobility algorithms.
    """

    eid: Series[Any] = pa.Field(alias=EID, nullable=True)
    author_ids: Series[Any] = pa.Field(alias=AUTHOR_IDS, nullable=True)
    author_afids: Series[Any] = pa.Field(alias=ARTICLE_AFFILIATION_ID, nullable=True)
    cover_date: Series[Any] = pa.Field(alias=ARTICLE_COVER_DATE, nullable=True)

    class Config:
        """Keep source serialization and extra article metadata unchanged."""

        strict = False
        coerce = False


class MobilityAffiliationSchema(pa.DataFrameModel):
    """Validate the tabular contract of the mobility GeoPackage.

    Geometry is validated and retained by GeoPandas outside this column
    contract. This schema deliberately differs from Scopus author-affiliation
    rows: it has no author ID and supplies institution-level address metadata.
    """

    affiliation_id: Series[Any] = pa.Field(alias=AFFILIATION_ID, nullable=False)
    affiliation_id_parent: Series[Any] | None = pa.Field(alias=AFFILIATION_ID_PARENT, nullable=True)
    affiliation_id_parent_chatgpt: Series[Any] | None = pa.Field(
        alias=AFFILIATION_ID_PARENT_CHATGPT, nullable=True
    )
    affiliation_id_parent_splink_chatgpt: Series[Any] | None = pa.Field(
        alias=AFFILIATION_ID_PARENT_SPLINK_CHATGPT, nullable=True
    )
    parent: Series[Any] | None = pa.Field(alias=AFFILIATION_PARENT_SOURCE, nullable=True)
    umbrella_id: Series[Any] | None = pa.Field(alias=AFFILIATION_UMBRELLA_ID, nullable=True)
    affiliation_name: Series[Any] | None = pa.Field(alias=AFFILIATION_NAME, nullable=True)
    preferred_name: Series[Any] | None = pa.Field(alias=AFFILIATION_PREFERRED_NAME, nullable=True)
    type: Series[Any] | None = pa.Field(alias=AFFILIATION_TYPE, nullable=True)
    address_part: Series[Any] | None = pa.Field(alias=AFFILIATION_ADDRESS, nullable=True)
    city: Series[Any] | None = pa.Field(alias=CITY, nullable=True)
    country: Series[Any] | None = pa.Field(alias=COUNTRY, nullable=True)

    @pa.check(AFFILIATION_ID)
    @classmethod
    def affiliation_ids_are_supported(cls, values: pd.Series) -> pd.Series:
        """Accept numeric Scopus IDs or stable OpenAlex institution URLs.

        :param values: Institution identifiers from the configured source.
        :return: Row-wise validity mask.
        """
        return values.map(_is_supported_affiliation_id)

    class Config:
        """Preserve additional institution and geometry metadata."""

        strict = False
        coerce = False


def _is_supported_affiliation_id(value: Any) -> bool:
    """Accept integral Scopus IDs and canonical OpenAlex institution URLs."""
    if type(value) in (bool, np.bool_):
        return False
    text = str(value)
    if _OPENALEX_INSTITUTION_ID_PATTERN.fullmatch(text):
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric.is_integer()


class MobilityAuthorAffiliationsOutputSchema(pa.DataFrameModel):
    """Validate per-author mobility affiliation-history output rows."""

    author_ids: Series[Any] = pa.Field(alias=AUTHOR_IDS, nullable=False)
    affiliation_id: Series[Any] = pa.Field(alias=AFFILIATION_ID, nullable=False)
    affiliation_type: Series[Any] | None = pa.Field(alias=AUTHOR_AFFILIATION_TYPE, nullable=True)
    affiliation_id_parent: Series[Any] | None = pa.Field(alias=AFFILIATION_ID_PARENT, nullable=True)

    @pa.check(AFFILIATION_ID)
    @classmethod
    def affiliation_ids_are_supported(cls, values: pd.Series) -> pd.Series:
        """Validate numeric Scopus IDs and stable OpenAlex institution URLs."""
        return values.map(_is_supported_affiliation_id)

    class Config:
        """Retain all additional history fields without coercion."""

        strict = False
        coerce = False


class MobilityAuthorInbreedingOutputSchema(pa.DataFrameModel):
    """Validate author-level inbreeding and mobility summary rows."""

    author_ids: Series[Any] = pa.Field(alias=AUTHOR_IDS, nullable=False)
    is_inbreeding_proxy: Series[Any] | None = pa.Field(alias=IS_INBREEDING_PROXY, nullable=True)

    @pa.check(IS_INBREEDING_PROXY)
    @classmethod
    def proxy_values_are_boolean(cls, values: pd.Series) -> pd.Series:
        """Allow only boolean inbreeding decisions or missing classifications.

        :param values: Author-level classification values.
        :return: Row-wise validity mask.
        """
        return values.map(lambda value: pd.isna(value) or isinstance(value, (bool, np.bool_)))

    class Config:
        """Retain scientific diagnostics without coercion."""

        strict = False
        coerce = False


class MobilityAuthorArticlesOutputSchema(pa.DataFrameModel):
    """Validate author-article rows with grouped affiliation identifiers."""

    eid: Series[Any] = pa.Field(alias=EID, nullable=False)
    author_ids: Series[Any] = pa.Field(alias=AUTHOR_IDS, nullable=False)
    author_afids: Series[str] = pa.Field(alias=ARTICLE_AFFILIATION_ID, nullable=False)
    unmapped_id: Series[str] = pa.Field(alias=UNMAPPED_AFFILIATION_ID, nullable=False)

    class Config:
        """Retain article metadata without coercion."""

        strict = False
        coerce = False
