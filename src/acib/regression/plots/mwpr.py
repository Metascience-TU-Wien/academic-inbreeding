# Copyright © 2026 TU Wien

"""Publication-weighted mean percentile-rank aggregation for regression plots."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from acib.constants import HAZEN_PERCENTILE

HAZEN_PERCENTILE_COLUMN = HAZEN_PERCENTILE
MWPR_COLUMN = "mwpr"
_WEIGHT_COLUMN = "_mwpr_weight"
_WEIGHTED_VALUE_COLUMN = "_mwpr_weighted_value"


@dataclass(frozen=True, slots=True)
class MwprAggregationSpec:
    """Describe the grouping and publication weighting for an mwPR summary.

    :param group_columns: Columns defining independent summary groups.
    :param value_column: Column containing percentile ranks.
    :param publication_column: Publication identifier used for fractional weighting.
    :param researcher_column: Optional researcher identifier for summary counts.
    :param min_publications: Minimum distinct publications retained per group.
    """

    group_columns: tuple[str, ...]
    value_column: str = HAZEN_PERCENTILE_COLUMN
    weight_column: str | None = None
    publication_column: str | None = None
    researcher_column: str | None = None
    min_publications: float = 0.0

    @property
    def required_columns(self) -> set[str]:
        """Return input columns needed by this aggregation."""
        return {
            *self.group_columns,
            self.value_column,
            *([self.weight_column] if self.weight_column else []),
            *([self.publication_column] if self.publication_column else []),
            *([self.researcher_column] if self.researcher_column else []),
        }


def aggregate_mwpr(data: pd.DataFrame, *, spec: MwprAggregationSpec) -> pd.DataFrame:
    """Aggregate publication percentile ranks with fractional group weights.

    Each publication is counted once within a group. When a publication belongs
    to multiple groups, its value is split equally across those memberships,
    preserving the source publication-weighting semantics.

    :param data: Publication-level observations.
    :param spec: Grouping and aggregation settings.
    :return: Sorted group summaries, including mwPR and publication counts.
    :raises ValueError: If groups are empty or the minimum count is negative.
    :raises KeyError: If required columns are missing.
    """
    if not spec.group_columns:
        raise ValueError("At least one group column is required.")
    missing = spec.required_columns - set(data.columns)
    if missing:
        raise KeyError(f"mwPR input is missing required columns: {sorted(missing)}")
    if spec.min_publications < 0:
        raise ValueError("min_publications must be greater than or equal to 0.")

    if spec.weight_column is not None and spec.publication_column is not None:
        raise ValueError("Use either weight_column or publication_column, not both.")
    frame = data[list(spec.required_columns)].copy()
    frame[spec.value_column] = pd.to_numeric(frame[spec.value_column], errors="coerce")
    if spec.publication_column is not None:
        frame = frame.dropna(
            subset=[*spec.group_columns, spec.publication_column, spec.value_column]
        )
        frame = frame.drop_duplicates([spec.publication_column, *spec.group_columns]).copy()
        group_count = frame.groupby(spec.publication_column, dropna=False).size()
        frame = frame.merge(group_count.rename("_group_count"), on=spec.publication_column)
        frame[_WEIGHT_COLUMN] = 1.0 / frame["_group_count"]
    else:
        frame = frame.dropna(subset=[*spec.group_columns, spec.value_column])
        frame[_WEIGHT_COLUMN] = (
            1.0
            if spec.weight_column is None
            else pd.to_numeric(frame[spec.weight_column], errors="coerce")
        )
        frame = frame.dropna(subset=[_WEIGHT_COLUMN])
        frame = frame[frame[_WEIGHT_COLUMN].gt(0)].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *spec.group_columns,
                MWPR_COLUMN,
                "publication_count",
                "weighted_publication_count",
                "researcher_count",
                "row_count",
                "mean_percentile",
                "median_percentile",
                "weight_sum",
            ]
        )

    frame[_WEIGHTED_VALUE_COLUMN] = frame[spec.value_column] * frame[_WEIGHT_COLUMN]
    grouped = frame.groupby(list(spec.group_columns), dropna=False, observed=True)
    result = grouped.agg(
        row_count=(spec.value_column, "size"),
        mean_percentile=(spec.value_column, "mean"),
        median_percentile=(spec.value_column, "median"),
        _weighted_sum=(_WEIGHTED_VALUE_COLUMN, "sum"),
        weight_sum=(_WEIGHT_COLUMN, "sum"),
    ).reset_index()
    if spec.publication_column is not None:
        counts = grouped[spec.publication_column].nunique().rename("publication_count")
    elif spec.weight_column is not None:
        counts = grouped[_WEIGHT_COLUMN].sum().rename("publication_count")
    else:
        counts = grouped.size().rename("publication_count")
    result = result.merge(counts.reset_index(), on=list(spec.group_columns))
    result[MWPR_COLUMN] = result["_weighted_sum"] / result["weight_sum"]
    result["weighted_publication_count"] = result["weight_sum"]
    if spec.researcher_column is not None:
        researchers = (
            frame.dropna(subset=[spec.researcher_column])
            .groupby(list(spec.group_columns), dropna=False, observed=True)[spec.researcher_column]
            .nunique()
            .rename("researcher_count")
            .reset_index()
        )
        result = result.merge(researchers, on=list(spec.group_columns), how="left")
    result = result[result["publication_count"].ge(spec.min_publications)]
    return (
        result.drop(columns="_weighted_sum")
        .sort_values(list(spec.group_columns))
        .reset_index(drop=True)
    )
