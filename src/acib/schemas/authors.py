# Copyright © 2026 TU Wien

"""Scopus author and disambiguation artifact DataFrame contracts."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from acib.constants import (
    AUTHOR_ID,
    EID,
    GIVEN_NAME,
    HISTORICAL_IDENTIFIER,
    RESEARCHER_ID,
    SUBJECT_AREAS,
    SURNAME,
)
from acib.name_disambiguation.constants import (
    ARTICLE_PARENT_SUBJECT_COVERAGE,
    ARTICLE_PARENT_SUBJECT_OVERLAP_COUNT,
    AUTHOR_HISTORY,
    AUTHOR_IDS,
    CANONICAL_GIVEN_NAME,
    CANONICAL_INDEXED_NAME,
    CANONICAL_INITIALS,
    CANONICAL_NAME,
    CANONICAL_SURNAME,
    CLUSTER_ID,
    CLUSTER_SIZE,
    COMPONENTS,
    CONFIDENCE,
    CURRENT_AFFILIATION_OVERLAP_COUNT,
    DECISION_LABEL,
    DISAMBIGUATED_CLUSTER_ID,
    EXACT_SPECIFIC_COVERAGE,
    EXACT_SPECIFIC_OVERLAP_COUNT,
    GENDER_COMPATIBLE,
    INFERRED_CODE_COVERAGE,
    INFERRED_CODE_OVERLAP_COUNT,
    INFERRED_COVERAGE,
    LEFT_AUTHOR_ID,
    LEFT_CANONICAL_GIVEN_NAME,
    LEFT_CANONICAL_INDEXED_NAME,
    LEFT_CANONICAL_INITIALS,
    LEFT_CANONICAL_NAME,
    LEFT_CANONICAL_SURNAME,
    LEFT_HISTORY,
    LEFT_NAME,
    LEFT_ROW_ID,
    NAME_VARIANTS_PARSED,
    OVERLAP_COUNT,
    PUBLICATION_END,
    PUBLICATION_START,
    RAW_SCORE,
    RESEARCHER_GIVEN_NAME_COVERAGE,
    RESEARCHER_GIVEN_NAME_OVERLAP_COUNT,
    RIGHT_AUTHOR_ID,
    RIGHT_CANONICAL_GIVEN_NAME,
    RIGHT_CANONICAL_INDEXED_NAME,
    RIGHT_CANONICAL_INITIALS,
    RIGHT_CANONICAL_NAME,
    RIGHT_CANONICAL_SURNAME,
    RIGHT_HISTORY,
    RIGHT_NAME,
    RIGHT_ROW_ID,
    ROW_ID,
    SUBJECT_AREA_LABELS,
    SUBJECT_AREAS_PARSED,
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
    """Validate Scopus candidate fields required by matching and disambiguation."""

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
        """Require a full author name or both given and family names."""
        columns = set(frame.columns)
        return AUTHOR_NAME in columns or NAME in columns or {GIVEN_NAME, SURNAME} <= columns

    class Config:
        """Allow additional Scopus metadata and avoid implicit coercion."""

        strict = False
        coerce = False


class DisambiguationAuthorSchema(pa.DataFrameModel):
    """Validate canonical author rows at the pair-scoring boundary."""

    sc_id: Series[str] = pa.Field(alias=RESEARCHER_ID, nullable=False)
    author_id: Series[str] = pa.Field(alias=AUTHOR_ID, nullable=False)
    eid: Series[str] = pa.Field(alias=EID, nullable=False)
    name: Series[str] = pa.Field(alias=NAME, nullable=False)
    surname: Series[str] = pa.Field(alias=SURNAME, nullable=False)
    given_name: Series[str] = pa.Field(alias=GIVEN_NAME, nullable=False)
    indexed_name: Series[str] = pa.Field(alias=INDEXED_NAME, nullable=False)
    initials: Series[str] = pa.Field(alias=INITIALS, nullable=False)
    canonical_name: Series[str] = pa.Field(alias=CANONICAL_NAME, nullable=False)
    canonical_surname: Series[str] = pa.Field(alias=CANONICAL_SURNAME, nullable=False)
    canonical_given_name: Series[str] = pa.Field(alias=CANONICAL_GIVEN_NAME, nullable=False)
    canonical_indexed_name: Series[str] = pa.Field(alias=CANONICAL_INDEXED_NAME, nullable=False)
    canonical_initials: Series[str] = pa.Field(alias=CANONICAL_INITIALS, nullable=False)
    publication_start: Series[Any] = pa.Field(alias=PUBLICATION_START, nullable=True)
    publication_end: Series[Any] = pa.Field(alias=PUBLICATION_END, nullable=True)
    name_variants_parsed: Series[Any] = pa.Field(alias=NAME_VARIANTS_PARSED, nullable=False)
    subject_areas_parsed: Series[Any] = pa.Field(alias=SUBJECT_AREAS_PARSED, nullable=False)
    subject_area_labels: Series[Any] = pa.Field(alias=SUBJECT_AREA_LABELS, nullable=False)

    class Config:
        """Retain raw profile evidence and researcher metadata as extra columns."""

        strict = False
        coerce = False


class AuthorPairDecisionSchema(pa.DataFrameModel):
    """Validate scored author-pair records written to the decision artifact."""

    sc_id: Series[str] = pa.Field(alias=RESEARCHER_ID, nullable=False)
    left_author_id: Series[str] = pa.Field(alias=LEFT_AUTHOR_ID, nullable=False)
    right_author_id: Series[str] = pa.Field(alias=RIGHT_AUTHOR_ID, nullable=False)
    label: Series[str] = pa.Field(
        alias=DECISION_LABEL,
        nullable=False,
        isin={"same_author", "possible_same_author", "distinct"},
    )
    raw_score: Series[float] = pa.Field(alias=RAW_SCORE, nullable=False)
    confidence: Series[float] = pa.Field(alias=CONFIDENCE, nullable=False, in_range=(0.0, 1.0))
    components: Series[Any] = pa.Field(alias=COMPONENTS, nullable=False)
    left_row_id: Series[str] | None = pa.Field(alias=LEFT_ROW_ID, nullable=True)
    right_row_id: Series[str] | None = pa.Field(alias=RIGHT_ROW_ID, nullable=True)
    left_name: Series[str] | None = pa.Field(alias=LEFT_NAME, nullable=True)
    right_name: Series[str] | None = pa.Field(alias=RIGHT_NAME, nullable=True)
    left_canonical_name: Series[str] | None = pa.Field(alias=LEFT_CANONICAL_NAME, nullable=True)
    right_canonical_name: Series[str] | None = pa.Field(alias=RIGHT_CANONICAL_NAME, nullable=True)
    left_canonical_surname: Series[str] | None = pa.Field(
        alias=LEFT_CANONICAL_SURNAME, nullable=True
    )
    right_canonical_surname: Series[str] | None = pa.Field(
        alias=RIGHT_CANONICAL_SURNAME, nullable=True
    )
    left_canonical_given_name: Series[str] | None = pa.Field(
        alias=LEFT_CANONICAL_GIVEN_NAME, nullable=True
    )
    right_canonical_given_name: Series[str] | None = pa.Field(
        alias=RIGHT_CANONICAL_GIVEN_NAME, nullable=True
    )
    left_canonical_indexed_name: Series[str] | None = pa.Field(
        alias=LEFT_CANONICAL_INDEXED_NAME, nullable=True
    )
    right_canonical_indexed_name: Series[str] | None = pa.Field(
        alias=RIGHT_CANONICAL_INDEXED_NAME, nullable=True
    )
    left_canonical_initials: Series[str] | None = pa.Field(
        alias=LEFT_CANONICAL_INITIALS, nullable=True
    )
    right_canonical_initials: Series[str] | None = pa.Field(
        alias=RIGHT_CANONICAL_INITIALS, nullable=True
    )
    left_history: Series[Any] | None = pa.Field(alias=LEFT_HISTORY, nullable=True)
    right_history: Series[Any] | None = pa.Field(alias=RIGHT_HISTORY, nullable=True)

    class Config:
        """Preserve extra evidence details in the pair report."""

        strict = False
        coerce = False


class AuthorClusterMembershipSchema(pa.DataFrameModel):
    """Validate an author membership row in a researcher-level cluster."""

    sc_id: Series[str] = pa.Field(alias=RESEARCHER_ID, nullable=False)
    cluster_id: Series[int] = pa.Field(alias=CLUSTER_ID, nullable=False, ge=1)
    disambiguated_cluster_id: Series[str] = pa.Field(alias=DISAMBIGUATED_CLUSTER_ID, nullable=False)
    cluster_size: Series[int] = pa.Field(alias=CLUSTER_SIZE, nullable=False, ge=1)
    author_id: Series[str] = pa.Field(alias=AUTHOR_ID, nullable=False)
    row_id: Series[str] | None = pa.Field(alias=ROW_ID, nullable=True)
    name: Series[str] | None = pa.Field(alias=NAME, nullable=True)
    canonical_name: Series[str] | None = pa.Field(alias=CANONICAL_NAME, nullable=True)
    subject_areas: Series[Any] | None = pa.Field(alias=SUBJECT_AREAS, nullable=True)
    history: Series[Any] | None = pa.Field(alias=AUTHOR_HISTORY, nullable=True)

    class Config:
        """Allow extra author-profile diagnostics in cluster output."""

        strict = False
        coerce = False


class AuthorClusterSelectionScoreSchema(pa.DataFrameModel):
    """Validate cluster-level evidence rows used by final selection."""

    sc_id: Series[str] = pa.Field(alias=RESEARCHER_ID, nullable=False)
    cluster_id: Series[int] = pa.Field(alias=CLUSTER_ID, nullable=False, ge=1)
    disambiguated_cluster_id: Series[str] = pa.Field(alias=DISAMBIGUATED_CLUSTER_ID, nullable=False)
    author_ids: Series[Any] = pa.Field(alias=AUTHOR_IDS, nullable=False)
    cluster_size: Series[int] = pa.Field(alias=CLUSTER_SIZE, nullable=False, ge=1)
    current_affiliation_overlap_count: Series[int] = pa.Field(
        alias=CURRENT_AFFILIATION_OVERLAP_COUNT, nullable=False, ge=0
    )
    researcher_given_name_overlap_count: Series[int] = pa.Field(
        alias=RESEARCHER_GIVEN_NAME_OVERLAP_COUNT, nullable=False, ge=0
    )
    researcher_given_name_coverage: Series[float] = pa.Field(
        alias=RESEARCHER_GIVEN_NAME_COVERAGE, nullable=False, in_range=(0.0, 1.0)
    )
    article_parent_subject_overlap_count: Series[int] = pa.Field(
        alias=ARTICLE_PARENT_SUBJECT_OVERLAP_COUNT, nullable=False, ge=0
    )
    article_parent_subject_coverage: Series[float] = pa.Field(
        alias=ARTICLE_PARENT_SUBJECT_COVERAGE, nullable=False, in_range=(0.0, 1.0)
    )
    inferred_code_overlap_count: Series[int] = pa.Field(
        alias=INFERRED_CODE_OVERLAP_COUNT, nullable=False, ge=0
    )
    inferred_code_coverage: Series[float] = pa.Field(
        alias=INFERRED_CODE_COVERAGE, nullable=False, in_range=(0.0, 1.0)
    )
    exact_specific_overlap_count: Series[int] = pa.Field(
        alias=EXACT_SPECIFIC_OVERLAP_COUNT, nullable=False, ge=0
    )
    exact_specific_coverage: Series[float] = pa.Field(
        alias=EXACT_SPECIFIC_COVERAGE, nullable=False, in_range=(0.0, 1.0)
    )
    overlap_count: Series[int] = pa.Field(alias=OVERLAP_COUNT, nullable=False, ge=0)
    inferred_coverage: Series[float] = pa.Field(
        alias=INFERRED_COVERAGE, nullable=False, in_range=(0.0, 1.0)
    )
    gender_compatible: Series[bool] = pa.Field(alias=GENDER_COMPATIBLE, nullable=False)
    history: Series[Any] | None = pa.Field(alias=AUTHOR_HISTORY, nullable=True)

    class Config:
        """Allow detailed evidence diagnostics in score artifacts."""

        strict = False
        coerce = False
