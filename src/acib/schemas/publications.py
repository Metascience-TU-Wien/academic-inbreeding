# Copyright © 2026 TU Wien

"""Publication-level author-evidence DataFrame contracts."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from acib.constants import AUTHOR_ID, EID, SUBJECT_AREAS
from acib.name_disambiguation.constants import (
    ARTICLE_AFFILIATION_CITY,
    ARTICLE_AFFILIATION_COUNTRY,
    ARTICLE_AUTHOR_NAMES,
    ARTICLE_CITY,
    ARTICLE_COUNTRY,
    ARTICLE_COUNTRY_CODE,
    ARTICLE_FUND_NUMBER,
    ARTICLE_GRANT_ID,
    ARTICLE_GRANT_NUMBER,
    ARTICLE_GRANT_NUMBERS,
    ARTICLE_JOURNAL,
    ARTICLE_JOURNALS,
    ARTICLE_PUBLICATION_CITY,
    ARTICLE_PUBLICATION_COUNTRY,
    ARTICLE_PUBLICATION_NAME,
    ARTICLE_SOURCE_TITLE,
    ARTICLE_SOURCE_TITLE_SHORT,
    ARTICLE_SOURCE_TITLES,
    AUTHOR_IDS,
    CITED_EID,
    CITING_EID,
    ISO3_COUNTRY_CODE,
)


class ArticleAuthorEvidenceSchema(pa.DataFrameModel):
    """Validate article rows used to build author-publication evidence."""

    eid: Series[str] = pa.Field(alias=EID, nullable=True)
    author_ids: Series[Any] | None = pa.Field(alias=AUTHOR_IDS, nullable=True)
    author_id: Series[Any] | None = pa.Field(alias=AUTHOR_ID, nullable=True)
    subject_areas: Series[Any] | None = pa.Field(alias=SUBJECT_AREAS, nullable=True)
    author_names: Series[Any] | None = pa.Field(alias=ARTICLE_AUTHOR_NAMES, nullable=True)
    journal: Series[Any] | None = pa.Field(alias=ARTICLE_JOURNAL, nullable=True)
    journals: Series[Any] | None = pa.Field(alias=ARTICLE_JOURNALS, nullable=True)
    source_title: Series[Any] | None = pa.Field(alias=ARTICLE_SOURCE_TITLE, nullable=True)
    source_titles: Series[Any] | None = pa.Field(alias=ARTICLE_SOURCE_TITLES, nullable=True)
    srctitle: Series[Any] | None = pa.Field(alias=ARTICLE_SOURCE_TITLE_SHORT, nullable=True)
    publication_name: Series[Any] | None = pa.Field(alias=ARTICLE_PUBLICATION_NAME, nullable=True)
    city: Series[Any] | None = pa.Field(alias=ARTICLE_CITY, nullable=True)
    publication_city: Series[Any] | None = pa.Field(alias=ARTICLE_PUBLICATION_CITY, nullable=True)
    affiliation_city: Series[Any] | None = pa.Field(alias=ARTICLE_AFFILIATION_CITY, nullable=True)
    country: Series[Any] | None = pa.Field(alias=ARTICLE_COUNTRY, nullable=True)
    country_code: Series[Any] | None = pa.Field(alias=ARTICLE_COUNTRY_CODE, nullable=True)
    publication_country: Series[Any] | None = pa.Field(
        alias=ARTICLE_PUBLICATION_COUNTRY, nullable=True
    )
    affiliation_country: Series[Any] | None = pa.Field(
        alias=ARTICLE_AFFILIATION_COUNTRY, nullable=True
    )
    iso3_code: Series[Any] | None = pa.Field(alias=ISO3_COUNTRY_CODE, nullable=True)
    grant_number: Series[Any] | None = pa.Field(alias=ARTICLE_GRANT_NUMBER, nullable=True)
    grant_numbers: Series[Any] | None = pa.Field(alias=ARTICLE_GRANT_NUMBERS, nullable=True)
    grant_id: Series[Any] | None = pa.Field(alias=ARTICLE_GRANT_ID, nullable=True)
    fund_no: Series[Any] | None = pa.Field(alias=ARTICLE_FUND_NUMBER, nullable=True)

    @pa.dataframe_check
    @classmethod
    def has_author_identifier_column(cls, frame: pd.DataFrame) -> bool:
        """Require at least one author-ID representation for evidence indexing."""
        return AUTHOR_IDS in frame.columns or AUTHOR_ID in frame.columns

    class Config:
        """Preserve serialized evidence columns and allow other article metadata."""

        strict = False
        coerce = False


class ArticleReferenceSchema(pa.DataFrameModel):
    """Validate citation-edge columns while allowing the configured empty edge table."""

    citing_eid: Series[Any] = pa.Field(alias=CITING_EID, nullable=True)
    cited_eid: Series[Any] = pa.Field(alias=CITED_EID, nullable=True)

    @pa.check(CITING_EID)
    @classmethod
    def citing_identifiers_are_text(cls, values: pd.Series) -> pd.Series:
        """Keep EIDs textual and do not allow numeric float coercion."""
        return values.map(lambda value: pd.isna(value) or isinstance(value, str))

    @pa.check(CITED_EID)
    @classmethod
    def cited_identifiers_are_text(cls, values: pd.Series) -> pd.Series:
        """Keep EIDs textual and do not allow numeric float coercion."""
        return values.map(lambda value: pd.isna(value) or isinstance(value, str))

    class Config:
        """Keep citation identifiers and any additional edge metadata unchanged."""

        strict = False
        coerce = False
