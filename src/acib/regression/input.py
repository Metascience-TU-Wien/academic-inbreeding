# Copyright © 2026 TU Wien

"""Prepare publication-level and researcher-level regression inputs."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from acib.common.titles import (
    TITLE_CLASS_COLUMN,
    TITLE_OUTPUT_COLUMN,
    classify_title_for_institution,
    professor_title_mask,
)
from acib.constants import (
    EID,
    PERSON_TITLE,
    RESEARCHER_ID,
    SUBJECT_AREAS,
    TITLE_CLASS,
)
from acib.mobility.constants import (
    ARTICLE_COVER_DATE as COVER_DATE,
    COAFFILIATION_PUBLICATION_SHARE,
    FIRST_DATE,
    IS_INBREEDING_PROXY,
    LAST_DATE,
    N_UNIQUE,
    PERSON_TITLE_RAW,
    PUBLICATION_GENERAL_ASJC_SUBJECT_AREA_PUBLICATION_COUNTS as ARTICLE_GENERAL_ASJC_COUNTS,
    PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS as ARTICLE_ASJC_FIELD_COLUMN,
)
from acib.mobility.inbreeding import (
    article_general_asjc_subject_area_publication_counts,
)
from acib.mobility.input import general_asjc_subject_area_dict
from acib.name_disambiguation.constants import (
    AUTHOR_IDS,
    SELECTED_DISAMBIGUATED_CLUSTER_ID as SELECTED_CLUSTER_ID,
)
from acib.regression.constants import (
    ARTICLE_IMPACT_COLUMN_ALIASES,
    ARTICLE_IMPACT_ID,
    ARTICLE_IMPACT_PASSTHROUGH_COLUMNS,
    AUTHOR_ARTICLE_COLUMNS,
    BROAD_ASJC_DESCRIPTIONS,
    CLASS_NAME,
    DI1,
    DI5,
    FIRST_YEAR,
    FRACTIONAL_TOP_COLUMNS,
    IMPACT_ASJC_FIELD_COLUMN,
    IMPACT_PUBLICATION_YEAR_RANGE,
    PP_TOP_10,
    PRIMARY_FIELD,
    PRIMARY_FIELD_SHARE,
    PROFESSOR_ARTICLE_OUTPUT_COLUMNS,
    PTOP_10_FRACTION,
    PUBLIC_PROD,
    PUBYEAR,
)
from acib.regression.fields import (
    FieldShareResult,
    impact_asjc_subject_area,
    professor_field_shares,
)

_logger = structlog.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegressionInputBuildResult:
    """Contain both prepared outputs and field-share diagnostics.

    :param professor_articles: Publication-level professor records.
    :param regression_input: Aggregated researcher-level regression rows.
    :param field_shares: Field-share tables and diagnostics.
    """

    professor_articles: pd.DataFrame
    regression_input: pd.DataFrame
    field_shares: FieldShareResult


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet table from disk.

    :param path: Input table path.
    :return: Loaded table.
    """
    table_path = Path(path)
    return (
        pd.read_parquet(table_path) if table_path.suffix == ".parquet" else pd.read_csv(table_path)
    )


def build_regression_input(
    *,
    author_articles: pd.DataFrame,
    authors: pd.DataFrame,
    article_impacts: pd.DataFrame | None = None,
    disruption_indicators: pd.DataFrame | None = None,
    min_publications: int = 4,
    min_first_year: int | None = None,
    professor_title_pattern: str = "prof",
) -> pd.DataFrame:
    """Build researcher-level regression data.

    :param author_articles: Publication-author records.
    :param authors: Researcher summary records.
    :param article_impacts: Optional impact indicators keyed by item ID or EID.
    :param disruption_indicators: Optional publication disruption metrics.
    :param min_publications: Minimum publication count; the source keeps counts strictly greater
        than ``min_publications - 1``.
    :param min_first_year: Optional lower bound on career start year.
    :param professor_title_pattern: Case-insensitive professor title pattern.
    :return: Aggregated regression input.
    """
    return build_regression_inputs(
        author_articles=author_articles,
        authors=authors,
        article_impacts=article_impacts,
        disruption_indicators=disruption_indicators,
        min_publications=min_publications,
        min_first_year=min_first_year,
        professor_title_pattern=professor_title_pattern,
    ).regression_input


def build_professor_article_input(
    *,
    author_articles: pd.DataFrame,
    authors: pd.DataFrame,
    article_impacts: pd.DataFrame | None = None,
    disruption_indicators: pd.DataFrame | None = None,
    min_publications: int = 4,
    min_first_year: int | None = None,
    professor_title_pattern: str = "prof",
) -> pd.DataFrame:
    """Build one row per professor-publication pair.

    Duplicate affiliation rows for the same researcher and publication are removed while a
    publication authored by distinct researchers remains represented for each researcher.

    :param author_articles: Publication-author records.
    :param authors: Researcher summary records.
    :param article_impacts: Optional impact indicators keyed by item ID or EID.
    :param disruption_indicators: Optional publication disruption metrics.
    :param min_publications: Minimum publications under the source's strict threshold semantics.
    :param min_first_year: Optional lower bound on career start year.
    :param professor_title_pattern: Case-insensitive professor title pattern.
    :return: Selected publication-level output columns.
    """
    result = prepare_author_articles(
        author_articles=author_articles,
        article_impacts=article_impacts,
        disruption_indicators=disruption_indicators,
    )
    authors_result = filter_regression_authors(
        authors=prepare_author_summary(authors),
        min_publications=min_publications,
        min_first_year=min_first_year,
        professor_title_pattern=professor_title_pattern,
    )
    return select_professor_article_output_columns(
        merge_professor_article_data(author_articles=result, authors=authors_result)
    )


def build_regression_inputs(
    *,
    author_articles: pd.DataFrame,
    authors: pd.DataFrame,
    article_impacts: pd.DataFrame | None = None,
    disruption_indicators: pd.DataFrame | None = None,
    min_publications: int = 4,
    min_first_year: int | None = None,
    professor_title_pattern: str = "prof",
) -> RegressionInputBuildResult:
    """Build publication-level and researcher-level regression inputs together.

    :param author_articles: Publication-author records.
    :param authors: Researcher summary records.
    :param article_impacts: Optional impact indicators keyed by item ID or EID.
    :param disruption_indicators: Optional publication disruption metrics.
    :param min_publications: Minimum publication count under source strict semantics.
    :param min_first_year: Optional lower bound on career start year.
    :param professor_title_pattern: Case-insensitive professor title pattern.
    :return: Both output tables and field-share result.
    """
    _logger.info(
        "Building regression inputs",
        author_article_rows=len(author_articles),
        author_rows=len(authors),
    )
    articles = prepare_author_articles(
        author_articles=author_articles,
        article_impacts=article_impacts,
        disruption_indicators=disruption_indicators,
    )
    author_summary = filter_regression_authors(
        authors=prepare_author_summary(authors),
        min_publications=min_publications,
        min_first_year=min_first_year,
        professor_title_pattern=professor_title_pattern,
    )
    professor_articles = merge_professor_article_data(
        author_articles=articles, authors=author_summary
    )
    group = regression_grouping_column(professor_articles)
    shares = professor_field_shares(
        professor_articles,
        group_column=group,
        field_column=regression_field_share_source_column(professor_articles),
    )
    regression = aggregate_regression_input(filter_regression_article_impacts(professor_articles))
    regression = regression.merge(shares.shares, on=group, how="left")
    return RegressionInputBuildResult(
        select_professor_article_output_columns(professor_articles), regression, shares
    )


# Remaining source-compatible transformations are implemented below.
def merge_article_impact_indicators(
    *, author_articles: pd.DataFrame, article_impacts: pd.DataFrame | None
) -> pd.DataFrame:
    """Merge impact records using ``item_id`` preferentially and preserve existing values.

    :param author_articles: Publication-author records keyed by EID.
    :param article_impacts: Optional impact records keyed by item ID or EID.
    :return: Joined records with canonical impact columns.
    :raises KeyError: If either nonempty input lacks its required identifier.
    """
    if article_impacts is None or article_impacts.empty:
        return author_articles
    if EID not in author_articles:
        raise KeyError("Article data is missing required column for impact merge: 'eid'")
    key = ARTICLE_IMPACT_ID if ARTICLE_IMPACT_ID in article_impacts else EID
    if key not in article_impacts:
        raise KeyError("Article impact data must contain 'item_id' or 'eid'.")
    impacts = normalize_article_impact_columns(article_impacts.rename(columns={key: EID}).copy())
    columns = [
        EID,
        *[c for c in ARTICLE_IMPACT_COLUMN_ALIASES if c in impacts],
        *[c for c in ARTICLE_IMPACT_PASSTHROUGH_COLUMNS if c in impacts],
    ]
    result = author_articles.merge(
        impacts[columns].drop_duplicates(subset=[EID]), on=EID, how="left", suffixes=("", "_impact")
    )
    for column in (*ARTICLE_IMPACT_COLUMN_ALIASES, *ARTICLE_IMPACT_PASSTHROUGH_COLUMNS):
        impact_column = f"{column}_impact"
        if impact_column in result:
            result[column] = (
                result[column].combine_first(result[impact_column])
                if column in result
                else result[impact_column]
            )
            result = result.drop(columns=[impact_column])
    return result


def normalize_article_impact_columns(author_articles: pd.DataFrame) -> pd.DataFrame:
    """Convert supported source impact aliases into canonical numeric columns.

    :param author_articles: Records that may contain impact aliases.
    :return: Copied data with available canonical indicators.
    """
    result = author_articles.copy()
    for output, aliases in ARTICLE_IMPACT_COLUMN_ALIASES.items():
        source = next((column for column in aliases if column in result), None)
        if source is not None:
            result[output] = pd.to_numeric(result[source], errors="coerce")
    return result


def normalize_fractional_top_columns(author_articles: pd.DataFrame) -> pd.DataFrame:
    """Ensure fractional top-percentile indicators remain numeric and present.

    :param author_articles: Publication records.
    :return: Copied data with numeric fractional indicators.
    """
    result = author_articles.copy()
    for column in FRACTIONAL_TOP_COLUMNS:
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def prepare_author_articles(
    *,
    author_articles: pd.DataFrame,
    article_impacts: pd.DataFrame | None = None,
    disruption_indicators: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalize publication rows, enrich with impact/ASJC data, and explode author IDs.

    :param author_articles: Publication-author records.
    :param article_impacts: Optional impact records.
    :param disruption_indicators: Optional disruption indicators.
    :return: Prepared publication-author records.
    """
    result = merge_article_impact_indicators(
        author_articles=author_articles, article_impacts=article_impacts
    )
    if disruption_indicators is not None and not disruption_indicators.empty:
        indicators = disruption_indicators.rename(columns={"focal_eid": EID}).copy()
        result = result.merge(indicators, on=EID, how="left", suffixes=("", "_di"))
        for col in (DI1, DI5):
            replacement = f"{col}_di"
            if replacement in result:
                result[col] = (
                    result[col].combine_first(result[replacement])
                    if col in result
                    else result[replacement]
                )
                result = result.drop(columns=[replacement])
    result = normalize_fractional_top_columns(normalize_article_impact_columns(result))
    if SUBJECT_AREAS in result and ARTICLE_ASJC_FIELD_COLUMN not in result:
        result[ARTICLE_ASJC_FIELD_COLUMN] = result[SUBJECT_AREAS].map(
            general_asjc_subject_area_dict
        )
    if CLASS_NAME in result and IMPACT_ASJC_FIELD_COLUMN not in result:
        result[IMPACT_ASJC_FIELD_COLUMN] = result[CLASS_NAME].map(impact_asjc_subject_area)
    if AUTHOR_IDS in result:
        result[AUTHOR_IDS] = result[AUTHOR_IDS].map(author_identifier_values)
        result = result.explode(AUTHOR_IDS, ignore_index=True)
        result[AUTHOR_IDS] = result[AUTHOR_IDS].map(normalize_author_identifier)
        result = result.dropna(subset=[AUTHOR_IDS]).reset_index(drop=True)
    return result


def author_identifier_values(value: Any) -> list[str]:
    """Parse author identifiers from collection, Python literal, or semicolon text.

    :param value: Serialized or structured author identifiers.
    :return: Parsed identifiers, excluding null-like values.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return []
    parsed = parse_python_literal(text)
    if isinstance(parsed, (list, tuple, set)):
        return [str(item) for item in parsed]
    return [part.strip() for part in text.split(";") if part.strip()]


def normalize_author_identifier(value: Any) -> str | None:
    """Normalize one author identifier and discard missing values.

    :param value: Author identifier.
    :return: Normalized identifier or ``None``.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return None
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def parse_python_literal(value: Any) -> Any:
    """Safely parse Python-literal text, mapping invalid or null-like values to ``None``.

    :param value: Value or serialized literal.
    :return: Parsed value, original non-string value, or ``None``.
    """
    if (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (isinstance(value, float) and np.isnan(value))
    ):
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def slugify_subject_category(category: str) -> str:
    """Convert a subject category into the historical count-column suffix.

    :param category: ASJC category label.
    :return: Lowercase underscore slug.
    """
    normalized = re.sub(r"^general\s+", "", category.strip(), flags=re.IGNORECASE)
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.lower()).strip("_")
    return normalized or "unknown"


def prepare_author_summary(authors: pd.DataFrame) -> pd.DataFrame:
    """Prepare researcher summaries and derive career and subject-area features.

    :param authors: Raw researcher summary records.
    :return: Normalized records with derived regression input columns.
    """
    result = authors.copy()
    raw_title = result.get(
        PERSON_TITLE,
        result.get(PERSON_TITLE_RAW, pd.Series("", index=result.index, dtype="object")),
    )
    classified_title = result.apply(
        lambda row: classify_title_for_institution(
            raw_title.loc[row.name],
            row.get("coaffiliated_professor_universities", row.get("university_name")),
        ),
        axis=1,
    )
    result[TITLE_CLASS_COLUMN] = classified_title
    if TITLE_OUTPUT_COLUMN in result and PERSON_TITLE_RAW not in result:
        existing_title = result[TITLE_OUTPUT_COLUMN]
        result[TITLE_OUTPUT_COLUMN] = existing_title.where(
            existing_title.notna() & existing_title.astype(str).str.strip().ne(""),
            result[TITLE_CLASS_COLUMN],
        )
    else:
        result[TITLE_OUTPUT_COLUMN] = result[TITLE_CLASS_COLUMN]
    if AUTHOR_IDS in result:
        result[AUTHOR_IDS] = result[AUTHOR_IDS].map(normalize_author_identifier)
    elif SELECTED_CLUSTER_ID in result:
        result[AUTHOR_IDS] = result[SELECTED_CLUSTER_ID].map(normalize_author_identifier)
    if ARTICLE_GENERAL_ASJC_COUNTS in result:
        result[[PRIMARY_FIELD, PRIMARY_FIELD_SHARE]] = (
            result[ARTICLE_GENERAL_ASJC_COUNTS].map(primary_asjc_field).apply(pd.Series)
        )
    result[FIRST_YEAR] = pd.to_datetime(result[FIRST_DATE], errors="coerce").dt.year
    if {N_UNIQUE, LAST_DATE}.issubset(result.columns):
        last_year = pd.to_datetime(result[LAST_DATE], errors="coerce").dt.year
        career_span = last_year - result[FIRST_YEAR] + 1
        result[PUBLIC_PROD] = pd.to_numeric(result[N_UNIQUE], errors="coerce").div(
            career_span.where(career_span.gt(0))
        )
    else:
        result[PUBLIC_PROD] = np.nan
    result[COAFFILIATION_PUBLICATION_SHARE] = safe_divide_missing_zero_denominator(
        result.get("n_coaffiliation_publications"), result.get(N_UNIQUE)
    )
    return add_subject_publication_count_columns(result)


def filter_regression_authors(
    *,
    authors: pd.DataFrame,
    min_publications: int,
    professor_title_pattern: str,
    min_first_year: int | None = None,
) -> pd.DataFrame:
    """Keep qualifying professor records in the regression sample.

    :param authors: Prepared researcher summary records.
    :param min_publications: Strict publication-count threshold.
    :param professor_title_pattern: Case-insensitive professor title pattern.
    :param min_first_year: Optional inclusive career-start year threshold.
    :return: Filtered copy of researcher records.
    :raises KeyError: If year filtering is requested without ``first_year``.
    """
    result = authors.copy()
    title_column = PERSON_TITLE if PERSON_TITLE in result.columns else PERSON_TITLE_RAW
    if title_column in result.columns and professor_title_pattern:
        result = result[
            professor_title_mask(
                result,
                title_column=title_column,
                professor_title_pattern=professor_title_pattern,
            )
        ]
    result = result[result[IS_INBREEDING_PROXY].notna()]
    if min_first_year is not None:
        if FIRST_YEAR not in result.columns:
            raise KeyError("Regression first-year filtering requires a 'first_year' column.")
        result = result[pd.to_numeric(result[FIRST_YEAR], errors="coerce").ge(min_first_year)]
    if N_UNIQUE in result.columns:
        result = result[pd.to_numeric(result[N_UNIQUE], errors="coerce").gt(min_publications - 1)]
    return result.copy()


def merge_author_article_regression_data(
    *, author_articles: pd.DataFrame, authors: pd.DataFrame
) -> pd.DataFrame:
    """Merge researcher and article data, then apply the impact-data filter.

    :param author_articles: Prepared publication-author records.
    :param authors: Filtered researcher records.
    :return: Publication records retained for regression aggregation.
    """
    merged = merge_professor_article_data(author_articles=author_articles, authors=authors)
    return filter_regression_article_impacts(merged)


def filter_regression_article_impacts(data: pd.DataFrame) -> pd.DataFrame:
    """Retain records with an available top-10 impact indicator when present.

    :param data: Merged publication-level records.
    :return: Filtered records with a fresh integer index.
    """
    result = data.copy()
    if PP_TOP_10 in result.columns:
        result = result[result[PP_TOP_10].notna()]
    return result.reset_index(drop=True)


def merge_professor_article_data(
    *, author_articles: pd.DataFrame, authors: pd.DataFrame
) -> pd.DataFrame:
    """Join article rows to author identities and remove duplicate professor-publication pairs.

    :param author_articles: Prepared publication-author records.
    :param authors: Prepared researcher summaries.
    :return: Joined, deduplicated professor-publication rows.
    """
    available = [column for column in AUTHOR_ARTICLE_COLUMNS if column in author_articles]
    merged = author_articles[available].merge(
        author_merge_frame(authors), on=AUTHOR_IDS, how="left"
    )
    merged = restore_article_asjc_field_column(merged)
    merged = merged[merged[IS_INBREEDING_PROXY].notna()].copy()
    merged = merged.drop_duplicates(professor_publication_key_columns(merged))
    merged["is_inbred"] = merged[IS_INBREEDING_PROXY]
    merged = add_author_subject_summary_from_articles(merged)
    return merged.reset_index(drop=True)


def restore_article_asjc_field_column(data: pd.DataFrame) -> pd.DataFrame:
    """Restore the article-side ASJC field following a merge suffix collision.

    :param data: Merged article-author data.
    :return: Data with the article field under its canonical column name.
    """
    article_column = f"{ARTICLE_ASJC_FIELD_COLUMN}_x"
    author_column = f"{ARTICLE_ASJC_FIELD_COLUMN}_y"
    if ARTICLE_ASJC_FIELD_COLUMN in data.columns or article_column not in data.columns:
        return data
    result = data.rename(columns={article_column: ARTICLE_ASJC_FIELD_COLUMN})
    return result.drop(columns=[author_column]) if author_column in result.columns else result


def author_merge_frame(authors: pd.DataFrame) -> pd.DataFrame:
    """Build identifier aliases used to match article rows to author summaries.

    :param authors: Prepared researcher summary records.
    :return: Deduplicated lookup rows keyed by normalized author ID.
    :raises KeyError: If no supported author identifier column is available.
    """
    frames = [author_id_lookup_frame(authors=authors, column=AUTHOR_IDS)]
    frames.append(author_id_lookup_frame(authors=authors, column=SELECTED_CLUSTER_ID))
    frames.append(selected_author_ids_lookup_frame(authors=authors))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise KeyError("Author data must contain an author identifier column for merging.")
    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=[AUTHOR_IDS])
        .drop_duplicates(subset=[AUTHOR_IDS, *author_identity_columns(authors)])
        .reset_index(drop=True)
    )


def author_id_lookup_frame(*, authors: pd.DataFrame, column: str) -> pd.DataFrame:
    """Create normalized author-ID lookup rows from one identifier column.

    :param authors: Researcher records.
    :param column: Identifier column to normalize.
    :return: Copy with canonical ``author_ids``, or an empty frame if unavailable.
    """
    if column not in authors.columns:
        return pd.DataFrame()
    result = authors.copy()
    result[AUTHOR_IDS] = result[column].map(normalize_author_identifier)
    return result


def selected_author_ids_lookup_frame(*, authors: pd.DataFrame) -> pd.DataFrame:
    """Expand and normalize the selected author-ID collection for lookup.

    :param authors: Researcher records.
    :return: One row per normalized selected author ID.
    """
    if "selected_author_ids" not in authors.columns:
        return pd.DataFrame()
    result = authors.copy()
    result[AUTHOR_IDS] = result["selected_author_ids"].map(author_identifier_values)
    result = result.explode(AUTHOR_IDS, ignore_index=True)
    result[AUTHOR_IDS] = result[AUTHOR_IDS].map(normalize_author_identifier)
    return result


def author_identity_columns(authors: pd.DataFrame) -> list[str]:
    """Return available stable researcher identity columns.

    :param authors: Researcher records.
    :return: Identity columns used to preserve distinct author matches.
    """
    return [column for column in (SELECTED_CLUSTER_ID, RESEARCHER_ID) if column in authors.columns]


def professor_publication_key_columns(data: pd.DataFrame) -> list[str]:
    """Choose the researcher-publication de-duplication key used by the source.

    :param data: Joined professor-publication rows.
    :return: Researcher and EID columns, or all columns if a key is unavailable.
    """
    professor_column = next(
        (column for column in (RESEARCHER_ID, SELECTED_CLUSTER_ID, AUTHOR_IDS) if column in data),
        None,
    )
    if professor_column is None or EID not in data.columns:
        return list(data.columns)
    return [professor_column, EID]


def add_author_subject_summary_from_articles(data: pd.DataFrame) -> pd.DataFrame:
    """Derive primary and broad subject summaries from each professor's articles.

    :param data: Joined professor-publication records.
    :return: Records with article-derived subject summaries when absent.
    """
    if PRIMARY_FIELD in data.columns or ARTICLE_ASJC_FIELD_COLUMN not in data.columns:
        return data
    key_columns = professor_publication_key_columns(data)
    if not key_columns:
        return data
    professor_column = key_columns[0]
    rows = []
    for professor_id, group in data.groupby(professor_column, sort=False, dropna=False):
        counts = article_general_asjc_subject_area_publication_counts(group)
        primary_field, primary_field_share = primary_asjc_field(counts)
        rows.append(
            {
                professor_column: professor_id,
                ARTICLE_GENERAL_ASJC_COUNTS: counts,
                PRIMARY_FIELD: primary_field,
                PRIMARY_FIELD_SHARE: primary_field_share,
                **broad_subject_publication_counts(counts),
            }
        )
    return data.merge(pd.DataFrame(rows), on=professor_column, how="left")


def select_professor_article_output_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Select the stable professor-publication output schema.

    :param data: Joined professor-publication rows.
    :return: Copy restricted to available output schema columns in canonical order.
    """
    columns = [column for column in PROFESSOR_ARTICLE_OUTPUT_COLUMNS if column in data]
    return data.loc[:, columns].copy()


def aggregate_regression_input(author_articles: pd.DataFrame) -> pd.DataFrame:
    """Aggregate publication rows into one regression record per researcher.

    :param author_articles: Impact-filtered professor-publication rows.
    :return: Researcher-level aggregates and derived outcome columns.
    """
    grouping_column = regression_grouping_column(author_articles)
    aggregation_data = author_articles.copy()
    if PUBYEAR in aggregation_data:
        impact_year = pd.to_numeric(aggregation_data[PUBYEAR], errors="coerce")
    elif COVER_DATE in aggregation_data:
        impact_year = pd.to_datetime(aggregation_data[COVER_DATE], errors="coerce").dt.year
    else:
        impact_year = pd.Series(np.nan, index=aggregation_data.index)
    impact_fraction = pd.to_numeric(
        aggregation_data.get("ptop10_frac", pd.Series(np.nan, index=aggregation_data.index)),
        errors="coerce",
    )
    impact_eids = aggregation_data.get(EID, pd.Series(pd.NA, index=aggregation_data.index))
    impact_available = (
        impact_fraction.notna()
        & impact_year.between(*IMPACT_PUBLICATION_YEAR_RANGE)
        & impact_eids.notna()
    )
    aggregation_data["_impact_eligible_eid"] = impact_eids.where(impact_available)
    aggregation_data["_impact_year"] = impact_year.where(impact_available)
    subject_count_columns = [c for c in aggregation_data.columns if c.startswith("n_public_")]
    aggregation = {
        SELECTED_CLUSTER_ID: (SELECTED_CLUSTER_ID, "first"),
        "pp_top_1": ("pp_top_1", "mean"),
        "pp_top_5": ("pp_top_5", "mean"),
        "pp_top_10": ("pp_top_10", "mean"),
        "P_top10": (
            PTOP_10_FRACTION,
            lambda values: pd.to_numeric(values, errors="coerce").sum(min_count=1),
        ),
        "P_top5": (
            "ptop5_frac",
            lambda values: pd.to_numeric(values, errors="coerce").sum(min_count=1),
        ),
        "P_top1": (
            "ptop1_frac",
            lambda values: pd.to_numeric(values, errors="coerce").sum(min_count=1),
        ),
        PUBLIC_PROD: (PUBLIC_PROD, "first"),
        "hazen_perc_med": ("hazen_perc_med", "mean"),
        "n_public": (N_UNIQUE, "first"),
        "academic_age": ("academic_age_years", "first"),
        "is_inbred": (IS_INBREEDING_PROXY, "first"),
        "n_affiliations": ("n_affiliations", "first"),
        "title": (TITLE_CLASS, "first"),
        "publication_mobility": ("has_publication_mobility", "first"),
        "short_publication_mobility": ("short_publication_mobility", "first"),
        "long_publication_mobility": ("long_publication_mobility", "first"),
        "n_coaffiliation_publications": ("n_coaffiliation_publications", "first"),
        "n_coaffiliations": ("n_coaffiliations", "first"),
        COAFFILIATION_PUBLICATION_SHARE: (COAFFILIATION_PUBLICATION_SHARE, "first"),
        "n_away_years": ("n_away_years", "first"),
        "n_away_years_domestic": ("n_away_years_domestic", "first"),
        "n_away_years_abroad": ("n_away_years_abroad", "first"),
        "total_coauthors_across_publications": ("total_coauthors_across_publications", "first"),
        "mean_coauthors_per_publication": ("mean_coauthors_per_publication", "first"),
        PRIMARY_FIELD: (PRIMARY_FIELD, "first"),
        PRIMARY_FIELD_SHARE: (PRIMARY_FIELD_SHARE, "first"),
        FIRST_YEAR: (FIRST_YEAR, "first"),
        DI1: (DI1, "median"),
        DI5: (DI5, "median"),
    }
    if N_UNIQUE not in author_articles.columns and EID in author_articles.columns:
        aggregation["n_public"] = (EID, "nunique")
    if EID in author_articles.columns:
        aggregation["n_public_impact"] = ("_impact_eligible_eid", "nunique")
        aggregation["_first_impact_year"] = ("_impact_year", "min")
        aggregation["_last_impact_year"] = ("_impact_year", "max")
    aggregation.update({column: (column, "first") for column in subject_count_columns})
    existing = {
        output: spec
        for output, spec in aggregation.items()
        if output != grouping_column and spec[0] in aggregation_data.columns
    }
    result = aggregation_data.groupby(grouping_column).agg(**existing)
    if {"_first_impact_year", "_last_impact_year"}.issubset(result.columns):
        impact_span = result["_last_impact_year"] - result["_first_impact_year"] + 1
        result["public_prod_impact"] = result["n_public_impact"].div(
            impact_span.where(impact_span.gt(0))
        )
        result = result.drop(columns=["_first_impact_year", "_last_impact_year"])
    else:
        result["public_prod_impact"] = np.nan
    return add_derived_regression_columns(result).reset_index()


def regression_grouping_column(author_articles: pd.DataFrame) -> str:
    """Select the available researcher key for regression grouping.

    :param author_articles: Publication-author records.
    :return: Researcher identifier column name.
    :raises KeyError: If no supported researcher key is present.
    """
    for column in (RESEARCHER_ID, SELECTED_CLUSTER_ID, AUTHOR_IDS):
        if column in author_articles.columns:
            return column
    raise KeyError(
        "Regression input requires sc_id, selected_disambiguated_cluster_id, or author_ids."
    )


def regression_field_share_source_column(author_articles: pd.DataFrame) -> str:
    """Choose the preferred available article field column for share calculation.

    :param author_articles: Publication-author records.
    :return: Impact ASJC field column when available, otherwise general ASJC field column.
    """
    return (
        IMPACT_ASJC_FIELD_COLUMN
        if IMPACT_ASJC_FIELD_COLUMN in author_articles.columns
        else ARTICLE_ASJC_FIELD_COLUMN
    )


def add_derived_regression_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add logarithmic, rate, and squared-age regression features.

    :param data: Aggregated regression records.
    :return: Copy with derived columns for available source fields.
    """
    result = data.copy()
    if {"n_public", "academic_age"}.issubset(result.columns):
        result["publication_rate"] = result["n_public"] / result["academic_age"]
        result["log_publication_rate"] = np.log1p(result["n_public"] / result["academic_age"])
    if "n_public" in result.columns:
        result["log_n_public"] = np.log1p(result["n_public"])
    if "n_affiliations" in result.columns:
        result["log_n_affiliations"] = np.log1p(result["n_affiliations"])
    if "academic_age" in result.columns:
        result["academic_age_sq"] = result["academic_age"] ** 2
    return result


def add_subject_publication_count_columns(authors: pd.DataFrame) -> pd.DataFrame:
    """Expand ASJC publication-count dictionaries into broad-field count columns.

    :param authors: Researcher records, optionally containing ASJC count dictionaries.
    :return: Copy with one integer count column per broad ASJC field.
    """
    if ARTICLE_GENERAL_ASJC_COUNTS not in authors.columns:
        return authors
    result = authors.copy()
    count_rows = result[ARTICLE_GENERAL_ASJC_COUNTS].map(broad_subject_publication_counts)
    count_frame = pd.DataFrame(list(count_rows), index=result.index).fillna(0).astype(int)
    for description in BROAD_ASJC_DESCRIPTIONS.values():
        column = f"n_public_{slugify_subject_category(description)}"
        result[column] = count_frame.get(column, 0)
    return result


def broad_subject_publication_counts(value: Any) -> dict[str, int]:
    """Aggregate detailed subject publication counts into broad ASJC columns.

    :param value: Parsed or serialized ASJC publication-count mapping.
    :return: Mapping from generated count-column name to summed publication count.
    """
    counts: dict[str, int] = {}
    for code, category, count in parse_subject_publication_count_entries(value):
        broad_code = code // 100 if code >= 100 else code
        broad_category = BROAD_ASJC_DESCRIPTIONS.get(broad_code, category)
        column = f"n_public_{slugify_subject_category(broad_category)}"
        counts[column] = counts.get(column, 0) + count
    return counts


def safe_divide(numerator: Any, denominator: Any) -> pd.Series:
    """Divide numeric values while treating zero-over-zero as zero.

    :param numerator: Numeric numerator values.
    :param denominator: Numeric denominator values.
    :return: Aligned quotient series with zero denominators handled as missing except 0/0.
    """
    numerator_series = pd.to_numeric(numerator, errors="coerce")
    denominator_series = pd.to_numeric(denominator, errors="coerce")
    if not isinstance(denominator_series, pd.Series):
        denominator_series = pd.Series([denominator_series])
    if not isinstance(numerator_series, pd.Series):
        numerator_series = pd.Series(numerator_series, index=denominator_series.index)
    else:
        numerator_series = numerator_series.reindex(denominator_series.index)
    result = numerator_series / denominator_series.replace(0, np.nan)
    return result.mask(denominator_series.eq(0) & numerator_series.fillna(0).eq(0), 0.0)


def safe_divide_missing_zero_denominator(numerator: Any, denominator: Any) -> pd.Series:
    """Divide numeric values, leaving zero denominators missing.

    :param numerator: Numeric numerator values.
    :param denominator: Numeric denominator values.
    :return: Aligned quotient series with zero denominators represented by ``NaN``.
    """
    numerator_series = pd.to_numeric(numerator, errors="coerce")
    denominator_series = pd.to_numeric(denominator, errors="coerce")
    if not isinstance(denominator_series, pd.Series):
        denominator_series = pd.Series([denominator_series])
    if not isinstance(numerator_series, pd.Series):
        numerator_series = pd.Series(numerator_series, index=denominator_series.index)
    else:
        numerator_series = numerator_series.reindex(denominator_series.index)
    return numerator_series / denominator_series.replace(0, np.nan)


def primary_asjc_field(value: Any) -> tuple[str | None, float | None]:
    """Find the most prevalent detailed ASJC category and its share.

    :param value: Parsed or serialized ASJC publication-count mapping.
    :return: Primary category and its fraction of all counted publications.
    """
    entries = parse_subject_publication_count_entries(value)
    if not entries:
        return None, None
    total = sum(count for _, _, count in entries)
    if total <= 0:
        return None, None
    _, category, count = max(entries, key=lambda item: (item[2], item[1]))
    return category, count / total


def parse_subject_publication_count_entries(value: Any) -> list[tuple[int, str, int]]:
    """Parse valid code/category/count entries from a subject-count mapping.

    :param value: Parsed or serialized mapping.
    :return: Entries as ``(ASJC code, category, count)`` tuples.
    """
    parsed = parse_python_literal(value)
    if not isinstance(parsed, dict):
        return []
    result: list[tuple[int, str, int]] = []
    for code, entry in parsed.items():
        if not isinstance(entry, dict):
            continue
        try:
            code_int = int(code)
            count = int(entry.get("n", 0))
        except (TypeError, ValueError):
            continue
        category = str(entry.get("category", "")).strip()
        if category:
            result.append((code_int, category, count))
    return result
