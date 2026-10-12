# Copyright © 2026 TU Wien

"""Contracts for regression-specific external inputs and generated tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from acib.constants import EID, RESEARCHER_ID, TITLE
from acib.mobility.constants import FIRST_DATE, IS_INBREEDING_PROXY
from acib.mobility.schemas import MobilityAuthorInbreedingOutputSchema
from acib.name_disambiguation.constants import AUTHOR_IDS
from acib.regression.constants import (
    ACADEMIC_AGE,
    ARTICLE_IMPACT_ID,
    CLASS_NAME,
    DI1,
    DI5,
    FIRST_YEAR,
    FOCAL_EID,
    HAZEN_PERCENTILE,
    IS_INBRED,
    LEIDEN_P_TOP_1,
    LEIDEN_P_TOP_5,
    LEIDEN_P_TOP_10,
    N_PUBLIC,
    P_TOP_1,
    P_TOP_5,
    P_TOP_10,
    PP_TOP_1,
    PP_TOP_5,
    PP_TOP_10,
    PTOP_1_FRACTION,
    PTOP_5_FRACTION,
    PTOP_10_FRACTION,
    PUBYEAR,
)


class RegressionImpactInputSchema(pa.DataFrameModel):
    """Validate external article-impact rows merged during input construction.

    The source accepts either OpenAlex ``item_id`` or Scopus ``eid`` and
    supports the historical aliases for percentage outcomes.
    """

    item_id: Series[Any] | None = pa.Field(alias=ARTICLE_IMPACT_ID, nullable=True)
    eid: Series[Any] | None = pa.Field(alias=EID, nullable=True)
    hazen_perc_med: Series[Any] | None = pa.Field(alias=HAZEN_PERCENTILE, nullable=True)
    pp_top_1: Series[Any] | None = pa.Field(alias=PP_TOP_1, nullable=True)
    pp_top_5: Series[Any] | None = pa.Field(alias=PP_TOP_5, nullable=True)
    pp_top_10: Series[Any] | None = pa.Field(alias=PP_TOP_10, nullable=True)
    leiden_p_top_1: Series[Any] | None = pa.Field(alias=LEIDEN_P_TOP_1, nullable=True)
    leiden_p_top_5: Series[Any] | None = pa.Field(alias=LEIDEN_P_TOP_5, nullable=True)
    leiden_p_top_10: Series[Any] | None = pa.Field(alias=LEIDEN_P_TOP_10, nullable=True)
    p_top_1: Series[Any] | None = pa.Field(alias=P_TOP_1, nullable=True)
    p_top_5: Series[Any] | None = pa.Field(alias=P_TOP_5, nullable=True)
    p_top_10: Series[Any] | None = pa.Field(alias=P_TOP_10, nullable=True)
    ptop1_frac: Series[Any] | None = pa.Field(alias=PTOP_1_FRACTION, nullable=True)
    ptop5_frac: Series[Any] | None = pa.Field(alias=PTOP_5_FRACTION, nullable=True)
    ptop10_frac: Series[Any] | None = pa.Field(alias=PTOP_10_FRACTION, nullable=True)
    class_name: Series[str] | None = pa.Field(alias=CLASS_NAME, nullable=True)
    pubyear: Series[Any] | None = pa.Field(alias=PUBYEAR, nullable=True)

    @pa.dataframe_check
    @classmethod
    def has_supported_article_identifier(cls, frame: pd.DataFrame) -> bool:
        """Require one of the source-supported article identifier columns."""
        return ARTICLE_IMPACT_ID in frame.columns or EID in frame.columns

    class Config:
        """Preserve extra impact metadata and do not coerce serialized values."""

        strict = False
        coerce = False


class RegressionDisruptionInputSchema(pa.DataFrameModel):
    """Validate optional disruption-index records joined by focal EID."""

    focal_eid: Series[Any] = pa.Field(alias=FOCAL_EID, nullable=False)
    di1: Series[Any] | None = pa.Field(alias=DI1, nullable=True)
    di5: Series[Any] | None = pa.Field(alias=DI5, nullable=True)

    class Config:
        """Preserve any additional disruption indicators without coercion."""

        strict = False
        coerce = False


class RegressionMobilityAuthorInputSchema(MobilityAuthorInbreedingOutputSchema):
    """Extend the mobility proxy contract with regression-required fields.

    :param sc_id: Researcher identifier used for author-level aggregation.
    :param first_date: First observed publication date used for career entry.
    """

    sc_id: Series[Any] = pa.Field(alias=RESEARCHER_ID, nullable=False)
    is_inbreeding_proxy: Series[Any] = pa.Field(alias=IS_INBREEDING_PROXY, nullable=True)
    first_date: Series[Any] = pa.Field(alias=FIRST_DATE, nullable=True)


class RegressionInputOutputSchema(pa.DataFrameModel):
    """Validate the researcher-level regression input consumed downstream."""

    sc_id: Series[Any] = pa.Field(alias=RESEARCHER_ID, nullable=False, unique=True)
    is_inbred: Series[Any] = pa.Field(alias=IS_INBRED, nullable=False)
    title: Series[str] = pa.Field(alias=TITLE, nullable=False)
    academic_age: Series[Any] | None = pa.Field(alias=ACADEMIC_AGE, nullable=True)
    n_public: Series[Any] | None = pa.Field(alias=N_PUBLIC, nullable=True)
    first_year: Series[Any] | None = pa.Field(alias=FIRST_YEAR, nullable=True)

    @pa.check(IS_INBRED)
    @classmethod
    def inbreeding_values_are_boolean(cls, values: pd.Series) -> pd.Series:
        """Reject non-boolean regression classification values without coercion.

        :param values: Regression sample inbreeding indicator.
        :return: Row-wise boolean validity mask.
        """
        return values.map(lambda value: isinstance(value, (bool, np.bool_)))

    class Config:
        """Retain outcome and discipline-specific fields as allowed extras."""

        strict = False
        coerce = False


class ProfessorArticleOutputSchema(pa.DataFrameModel):
    """Validate prepared professor-publication rows used by regression plots."""

    eid: Series[Any] = pa.Field(alias=EID, nullable=True)
    author_ids: Series[Any] = pa.Field(alias=AUTHOR_IDS, nullable=False)
    sc_id: Series[Any] = pa.Field(alias=RESEARCHER_ID, nullable=False)
    title: Series[str] = pa.Field(alias=TITLE, nullable=False)
    is_inbred: Series[Any] = pa.Field(alias=IS_INBRED, nullable=False)

    @pa.check(IS_INBRED)
    @classmethod
    def inbreeding_values_are_boolean(cls, values: pd.Series) -> pd.Series:
        """Reject non-boolean professor-publication statuses without coercion.

        :param values: Publication-level inbreeding statuses.
        :return: Row-wise boolean validity mask.
        """
        return values.map(lambda value: isinstance(value, (bool, np.bool_)))

    class Config:
        """Preserve impact, subject-area, and mobility columns as allowed extras."""

        strict = False
        coerce = False


class RegressionCoefficientOutputSchema(pa.DataFrameModel):
    """Validate the long-form coefficient artifact emitted by fitted models."""

    dependent_variable: Series[str] = pa.Field(nullable=False)
    model_id: Series[str] = pa.Field(nullable=False)
    term: Series[str] = pa.Field(nullable=False)
    estimate: Series[Any] = pa.Field(nullable=True)
    std_error: Series[Any] = pa.Field(nullable=True)
    p_value: Series[Any] = pa.Field(nullable=True)
    conf_low: Series[Any] = pa.Field(nullable=True)
    conf_high: Series[Any] = pa.Field(nullable=True)
    n_obs: Series[int] = pa.Field(nullable=False)

    class Config:
        """Retain inference and optimizer diagnostics without coercion."""

        strict = False
        coerce = False


class RegressionFitStatisticsOutputSchema(pa.DataFrameModel):
    """Validate one fit-statistics record per dependent-variable/model pair."""

    dependent_variable: Series[str] = pa.Field(nullable=False)
    model_id: Series[str] = pa.Field(nullable=False)
    formula: Series[str] = pa.Field(nullable=False)
    n_obs: Series[int] = pa.Field(nullable=False)
    aic: Series[Any] = pa.Field(nullable=True)
    bic: Series[Any] = pa.Field(nullable=True)

    class Config:
        """Retain model-family and optimizer diagnostics without coercion."""

        strict = False
        coerce = False


class RegressionDescriptiveOutputSchema(pa.DataFrameModel):
    """Validate the dependent-variable descriptives table."""

    dependent_variable: Series[str] = pa.Field(nullable=False)
    mean: Series[Any] = pa.Field(nullable=True)
    median: Series[Any] = pa.Field(nullable=True)
    std: Series[Any] = pa.Field(nullable=True)
    n: Series[int] = pa.Field(nullable=False)

    class Config:
        """Retain additional descriptive fields without coercion."""

        strict = False
        coerce = False
