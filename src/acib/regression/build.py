# Copyright © 2026 TU Wien

"""Build and persist generated regression input tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from acib.mobility.schemas import MobilityArticleAuthorSchema
from acib.regression.config import RegressionConfig
from acib.regression.fields import field_share_columns
from acib.regression.input import (
    RegressionInputBuildResult,
    build_regression_inputs,
    read_table,
)
from acib.regression.schemas import (
    ProfessorArticleOutputSchema,
    RegressionDisruptionInputSchema,
    RegressionImpactInputSchema,
    RegressionInputOutputSchema,
    RegressionMobilityAuthorInputSchema,
)

_logger = structlog.getLogger(__name__)


def build_regression_inputs_from_config(
    config: RegressionConfig,
) -> RegressionInputBuildResult:
    """Build configured regression tables and write all declared build artifacts.

    :param config: Resolved regression stage configuration.
    :return: Prepared professor-article and researcher-level regression inputs.
    :raises FileNotFoundError: If a configured source input is missing.
    """
    config.check_paths_exist()
    _logger.info(
        "Loading regression build inputs",
        article_author_file_path=str(config.article_author_file_path),
        mobility_author_proxy=str(config.mobility_author_proxy),
        article_impact_file_path=(
            str(config.article_impact_file_path) if config.article_impact_file_path else None
        ),
        disruption_indicator_file_path=(
            str(config.disruption_indicator_file_path)
            if config.disruption_indicator_file_path
            else None
        ),
    )
    disruption_indicators = (
        RegressionDisruptionInputSchema.validate(
            read_table(config.disruption_indicator_file_path), lazy=True
        )
        if config.disruption_indicator_file_path is not None
        else None
    )
    article_impacts = (
        RegressionImpactInputSchema.validate(read_table(config.article_impact_file_path), lazy=True)
        if config.article_impact_file_path is not None
        else None
    )
    author_articles = MobilityArticleAuthorSchema.validate(
        read_table(config.article_author_file_path), lazy=True
    )
    authors = RegressionMobilityAuthorInputSchema.validate(
        read_table(config.mobility_author_proxy), lazy=True
    )
    _logger.info(
        "Loaded regression build inputs",
        article_author_rows=len(author_articles),
        mobility_author_rows=len(authors),
        article_impact_rows=len(article_impacts) if article_impacts is not None else 0,
        disruption_indicator_rows=(
            len(disruption_indicators) if disruption_indicators is not None else 0
        ),
    )
    result = build_regression_inputs(
        author_articles=author_articles,
        authors=authors,
        article_impacts=article_impacts,
        disruption_indicators=disruption_indicators,
        min_publications=config.min_publications,
        min_first_year=config.min_first_year,
        professor_title_pattern=config.professor_title_pattern,
    )
    for frame, schema in (
        (result.professor_articles, ProfessorArticleOutputSchema),
        (result.regression_input, RegressionInputOutputSchema),
    ):
        if not frame.empty or not frame.columns.empty:
            schema.validate(frame, lazy=True)
    configured_outcomes = set(config.dependent_variables)
    missing_outcomes = configured_outcomes - set(result.regression_input.columns)
    if missing_outcomes:
        raise KeyError(
            "Regression input is missing configured outcome(s): "
            + ", ".join(sorted(missing_outcomes))
        )
    outputs = config.outputs
    _logger.info(
        "Writing generated regression inputs",
        professor_article_output_path=str(outputs.professor_article_input),
        regression_input_path=str(outputs.regression_input),
    )
    write_table(result.professor_articles, outputs.professor_article_input)
    write_table(
        professor_article_diagnostics(result.professor_articles),
        outputs.professor_article_diagnostics,
    )
    write_table(result.regression_input, outputs.regression_input)
    write_table(result.field_shares.diagnostics, outputs.asjc_field_share_diagnostics)
    write_table(
        regression_input_diagnostics(result.regression_input),
        outputs.regression_input_diagnostics,
    )
    return result


def regression_input_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize researcher counts and field-share row totals.

    :param data: Researcher-level regression table.
    :return: Metric/value diagnostic rows.
    """
    group_column = "sc_id" if "sc_id" in data.columns else data.columns[0]
    share_columns = field_share_columns(data)
    rows: list[dict[str, object]] = [
        {"metric": "regression_rows", "value": len(data)},
        {"metric": "unique_professors", "value": data[group_column].nunique(dropna=True)},
        {
            "metric": "duplicate_professor_rows",
            "value": int(data.duplicated(subset=[group_column]).sum()),
        },
    ]
    if share_columns:
        share_sums = data[share_columns].sum(axis=1)
        rows.extend(
            [
                {"metric": "field_share_sum_min", "value": share_sums.min()},
                {"metric": "field_share_sum_max", "value": share_sums.max()},
                {
                    "metric": "field_share_sum_off_one",
                    "value": int((~share_sums.sub(1.0).abs().lt(1e-10)).sum()),
                },
            ]
        )
    return pd.DataFrame(rows)


def professor_article_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize professor-publication rows and available class-name coverage.

    :param data: Professor-publication table.
    :return: Metric/value diagnostic rows.
    """
    rows: list[dict[str, object]] = [
        {"metric": "professor_article_rows", "value": len(data)},
    ]
    if "eid" in data.columns:
        rows.append({"metric": "used_item_ids", "value": data["eid"].nunique(dropna=True)})
    if "class_name" in data.columns:
        has_class_name = data["class_name"].map(has_text_value)
        rows.extend(
            [
                {
                    "metric": "professor_article_rows_with_class_name",
                    "value": int(has_class_name.sum()),
                },
                {
                    "metric": "professor_article_rows_without_class_name",
                    "value": int((~has_class_name).sum()),
                },
            ]
        )
        if "eid" in data.columns:
            item_coverage = (
                data.assign(_has_class_name=has_class_name).groupby("eid")["_has_class_name"].any()
            )
            rows.extend(
                [
                    {"metric": "used_item_ids_with_class_name", "value": int(item_coverage.sum())},
                    {
                        "metric": "used_item_ids_without_class_name",
                        "value": int((~item_coverage).sum()),
                    },
                ]
            )
    return pd.DataFrame(rows)


def has_text_value(value: object) -> bool:
    """Return whether a value contains meaningful non-null text.

    :param value: Candidate text value.
    :return: Whether the stripped value is not a null marker.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text and text.casefold() not in {"nan", "none", "null", "{}"})


def write_table(data: pd.DataFrame, path: Path) -> None:
    """Write a CSV or Parquet table after normalizing nested values.

    :param data: Table to persist.
    :param path: Explicit configured output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    output = prepare_table_for_output(data)
    if path.suffix == ".parquet":
        output.to_parquet(path, index=False)
    else:
        output.to_csv(path, index=False)


def prepare_table_for_output(data: pd.DataFrame) -> pd.DataFrame:
    """Serialize nested values in object columns for portable output.

    :param data: Input table.
    :return: Copy with nested Python values serialized as JSON.
    """
    result = data.copy()
    for column in result.select_dtypes(include="object").columns:
        if result[column].map(is_nested_value).any():
            result[column] = result[column].map(serialize_nested_value)
    return result


def is_nested_value(value: Any) -> bool:
    """Return whether a value is a supported nested Python container.

    :param value: Value to inspect.
    :return: Whether the value is a dictionary, list, tuple, or set.
    """
    return isinstance(value, (dict, list, tuple, set))


def serialize_nested_value(value: Any) -> Any:
    """Serialize supported nested containers as deterministic JSON text.

    :param value: Value to serialize.
    :return: JSON string for nested containers, otherwise the original value.
    """
    if not is_nested_value(value):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
