# Copyright © 2026 TU Wien

"""Classify academic inbreeding proxies and publication mobility histories."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from acib.common.titles import (
    TITLE_OUTPUT_COLUMN,
    adjust_title_for_institution,
    classify_title_for_institution,
)
from acib.constants import (
    AFFILIATION_ID as AFFILIATION_ID_COLUMN,
    AFFILIATION_ID_PARENT as AFFILIATION_ID_PARENT_COLUMN,
    AFFILIATION_NAME as AFFILIATION_NAME_COLUMN,
    AUTHOR_ID as AUTHOR_SCOPUS_ID_COLUMN,
    EID as EID_COLUMN,
    PERSON_TITLE as PERSON_TITLE_COLUMN,
    RESEARCHER_ID as RESEARCHER_ID_COLUMN_NAME,
    UNIVERSITY_NAME as UNIVERSITY_NAME_COLUMN,
)
from acib.mobility.config import MobilityOutputs
from acib.mobility.constants import (
    ACADEMIC_AGE_YEARS,
    ADDRESS as ADDRESS_COLUMN,
    AFFILIATION_ADDRESS as AFFILIATION_ADDRESS_COLUMN,
    AFFILIATION_PREFERRED_NAME as AFFILIATION_PREFERRED_NAME_COLUMN,
    AFFILIATION_TYPE as AFFILIATION_TYPE_COLUMN,
    ARTICLE_AFFILIATION_ID as ARTICLE_AFFILIATION_ID_COLUMN,
    ARTICLE_COVER_DATE as ARTICLE_COVER_DATE_COLUMN,
    ARTICLE_DOI,
    ARTICLE_E_ISSN as ARTICLE_E_ISSN_COLUMN,
    ARTICLE_FUND_ACRONYM as ARTICLE_FUND_ACRONYM_COLUMN,
    ARTICLE_FUND_SPONSOR as ARTICLE_FUND_SPONSOR_COLUMN,
    ARTICLE_ISSN as ARTICLE_ISSN_COLUMN,
    ARTICLE_SOURCE_ID as ARTICLE_SOURCE_ID_COLUMN,
    AUTHOR_AFFILIATION_TYPE as AUTHOR_AFFILIATION_TYPE_COLUMN,
    AUTHOR_PROFILE_AMBIGUITY_REASON as AUTHOR_PROFILE_AMBIGUITY_REASON_COLUMN,
    AUTHOR_PROFILE_AMBIGUOUS as AUTHOR_PROFILE_AMBIGUOUS_COLUMN,
    AUTHOR_PROFILE_ARTICLE_EIDS,
    CITY as CITY_COLUMN,
    COAFFILIATED_PROFESSOR_UNIVERSITIES as COAFFILIATED_PROFESSOR_UNIVERSITIES_COLUMN,
    COAFFILIATION_PUBLICATION_SHARE,
    COUNTRY as COUNTRY_COLUMN,
    DATE_FIRST,
    DATE_LAST,
    EARLY_AFFILIATION_FIRST_N_PUBLICATIONS,
    EARLY_AFFILIATION_METHOD,
    EARLY_AFFILIATION_METHODS,
    EARLY_WINDOW_END_YEAR,
    ESTIMATED_FIRST_AFFILIATION_COUNTS,
    ESTIMATED_FIRST_AFFILIATION_IDS,
    FIRST_AFFILIATION_PUBLICATION_COUNT_OBSERVED,
    FIRST_AFFILIATION_PUBLICATION_COUNT_REQUIRED,
    FIRST_AFFILIATION_YEAR_COUNT,
    FIRST_DATE,
    FIRST_INST_SET,
    FIRST_PUBLICATION_YEAR,
    FIRST_UNIVERSITY_AFFILIATION_OVERLAP_IDS,
    HAS_ANY_COAFFILIATION,
    HAS_PUBLICATION_MOBILITY,
    INBREEDING_DETECTION_METHOD,
    INBREEDING_DETECTION_STATUS,
    INBREEDING_MANUAL_REVIEW_REQUIRED,
    IS_COAFFILIATED_PROFESSOR as IS_COAFFILIATED_PROFESSOR_COLUMN,
    IS_INBREEDING_PROXY,
    IS_MERGED_PROFESSOR_PROFILE as IS_MERGED_PROFESSOR_PROFILE_COLUMN,
    LAST_DATE,
    LAST_INST_SET,
    LAST_UNIVERSITY_AFFILIATION_OVERLAP_IDS,
    LONG_PUBLICATION_MOBILITY,
    MAX_AWAY_SPELL_YEARS,
    MEAN_COAUTHORS_PER_PUBLICATION,
    MERGED_PROFESSOR_ROWS as MERGED_PROFESSOR_ROWS_COLUMN,
    MIN_PUBLICATION_HISTORY_REQUIREMENT_MET,
    MIN_PUBLICATION_HISTORY_YEARS,
    MOBILITY_AUTHOR_PROFILE_ID as MOBILITY_AUTHOR_PROFILE_ID_COLUMN,
    MOBILITY_INSTITUTION_KEY as MOBILITY_INSTITUTION_KEY_COLUMN,
    MOBILITY_RESEARCHER_NAME,
    N_AFFILIATIONS,
    N_AWAY_PUBLICATIONS,
    N_AWAY_YEARS,
    N_AWAY_YEARS_ABROAD,
    N_AWAY_YEARS_DOMESTIC,
    N_COAFFILIATED_PROFESSOR_UNIVERSITIES as N_COAFFILIATED_PROFESSOR_UNIVERSITIES_COLUMN,
    N_COAFFILIATION_PUBLICATIONS,
    N_COAFFILIATIONS,
    N_PARENT_INSTITUTIONS,
    N_PUBLICATIONS_IN_EARLY_WINDOW,
    N_ROWS,
    N_UNIQUE,
    N_UNIQUE_AFFILIATIONS,
    PERSON_TITLE_RAW,
    PROFESSOR_NAMES as PROFESSOR_NAMES_COLUMN,
    PROFESSOR_SC_IDS as PROFESSOR_SC_IDS_COLUMN,
    PROFESSOR_TITLE_CLASSES as PROFESSOR_TITLE_CLASSES_COLUMN,
    PROFESSOR_TITLES as PROFESSOR_TITLES_COLUMN,
    PUBLICATION_AUTHOR_COUNT as PUBLICATION_AUTHOR_COUNT_COLUMN,
    PUBLICATION_GENERAL_ASJC_SUBJECT_AREA_PUBLICATION_COUNTS as PUBLICATION_ASJC_COUNTS_COLUMN,
    PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS as PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS_COLUMN,
    PUBLICATION_HISTORY_YEARS,
    SHARED_INST_SET,
    SHORT_PUBLICATION_MOBILITY,
    TARGET_IN_EARLY_WINDOW,
    TOTAL_COAUTHORS_ACROSS_PUBLICATIONS,
    UNIQUE_AFFILIATIONS,
    UNIVERSITY_AFFILIATION_IDS,
    UNMAPPED_AFFILIATION_ID,
)
from acib.mobility.input import (
    add_mobility_author_profile_ids,
    parse_literal,
    process_input_data,
)
from acib.name_disambiguation.constants import (
    ARTICLE_AFFILIATION_CITY as ARTICLE_AFFILIATION_CITY_COLUMN,
    ARTICLE_AFFILIATION_COUNTRY as ARTICLE_AFFILIATION_COUNTRY_COLUMN,
    ARTICLE_FUND_NUMBER as ARTICLE_FUND_NO_COLUMN,
    ARTICLE_PUBLICATION_NAME,
    AUTHOR_IDS as ARTICLE_AUTHOR_ID_COLUMN,
    SELECTED_AUTHOR_IDS as DISAMBIGUATED_AUTHOR_IDS_COLUMN,
    SELECTED_DISAMBIGUATED_CLUSTER_ID as DISAMBIGUATED_AUTHOR_ID_COLUMN,
    SELECTED_HISTORY as DISAMBIGUATED_AUTHOR_HISTORY_COLUMN,
)
from acib.name_matching.constants import (
    RESEARCHER_NAME as RESEARCHER_PROCESSED_NAME_COLUMN,
    SCOPUS_ID_MAPPING,
)

_logger = structlog.getLogger(__name__)
_WORKER_RESOLVED_AUTHORS: pd.DataFrame | None = None
_WORKER_AFFILIATION_METADATA: pd.DataFrame | None = None
_WORKER_ENABLE_UNIVERSITY_AFFILIATION_INBREEDING_DETECTION = False
_WORKER_FIRST_AFFILIATION_PUBLICATION_COUNT = 3
_WORKER_EARLY_AFFILIATION_METHOD = "first_n_publications"
_WORKER_FIRST_AFFILIATION_YEAR_COUNT = 2
_WORKER_MIN_PUBLICATION_HISTORY_YEARS: int | None = None

_MAX_PLAUSIBLE_PUBLICATION_HISTORY_YEARS = 70
_PUBLICATION_EVENT_KEY_COLUMN = "publication_event_key"
_PUBLICATION_ORDER_COLUMN = "publication_order"
_MOBILITY_COUNTRY_SET_COLUMN = "_mobility_country_set"
_YEAR_COLUMN = "year"
_INSTITUTION_SET_COLUMN = "inst_set"
_COUNTRY_SET_COLUMN = "country_set"
_N_AFFILIATIONS_ON_PAPER_COLUMN = "n_affiliations_on_paper"
_IS_COAFFILIATED_COLUMN = "is_coaffiliated"
_HAS_HOME_COLUMN = "has_home"
_IS_AWAY_COLUMN = "is_away"

_AFFILIATION_METADATA_COLUMNS = [
    AFFILIATION_ID_COLUMN,
    AFFILIATION_PREFERRED_NAME_COLUMN,
    AFFILIATION_TYPE_COLUMN,
    AFFILIATION_ADDRESS_COLUMN,
    CITY_COLUMN,
    COUNTRY_COLUMN,
    AFFILIATION_ID_PARENT_COLUMN,
]


@dataclass(frozen=True, slots=True)
class InbreedingResult:
    """Tabular outputs produced by academic inbreeding analysis.

    :param author_affiliations: Per-author affiliation history records.
    :param author_summary: Per-profile analysis and classification records.
    :param author_articles: Author-article rows with grouped affiliation identifiers.
    """

    author_affiliations: pd.DataFrame
    author_summary: pd.DataFrame
    author_articles: pd.DataFrame


@dataclass(frozen=True, slots=True)
class InbreedingDetection:
    """Proxy classification and diagnostics for one author profile.

    Attributes hold the classification, review status, and evidence used to
    estimate early and current university affiliations.
    """

    is_inbreeding_proxy: bool | None
    method: str
    status: str
    manual_review_required: bool
    university_affiliation_ids: list[int]
    last_university_affiliation_overlap_ids: list[int]
    estimated_first_affiliation_ids: list[int]
    estimated_first_affiliation_counts: dict[int, int]
    first_university_affiliation_overlap_ids: list[int]
    first_affiliation_publication_count_required: int
    first_affiliation_publication_count_observed: int
    early_affiliation_method: str
    first_affiliation_year_count: int
    first_publication_year: int | None
    early_window_end_year: int | None
    n_publications_in_early_window: int | None
    publication_history_years: int | None
    min_publication_history_years: int | None
    min_publication_history_requirement_met: bool | None
    target_in_early_window: bool | None


def analyze_academic_inbreeding(
    *,
    author_affiliations: pd.DataFrame,
    articles: pd.DataFrame,
    resolved_authors: pd.DataFrame,
    enable_university_affiliation_inbreeding_detection: bool = False,
    first_affiliation_publication_count: int = 3,
    early_affiliation_method: str = EARLY_AFFILIATION_FIRST_N_PUBLICATIONS,
    first_affiliation_year_count: int = 2,
    min_publication_history_years: int | None = None,
    max_workers: int = 3,
) -> InbreedingResult:
    """Prepare mobility evidence and classify resolved academic profiles.

    :param author_affiliations: Affiliation metadata table.
    :param articles: Article-author-affiliation evidence.
    :param resolved_authors: Final selected researcher-author profiles.
    :return: Affiliation history, profile summary, and author-article rows.
    """
    validate_early_affiliation_parameters(
        early_affiliation_method=early_affiliation_method,
        first_affiliation_publication_count=first_affiliation_publication_count,
        first_affiliation_year_count=first_affiliation_year_count,
        min_publication_history_years=min_publication_history_years,
    )
    article_author_df, normalized_affiliations = process_input_data(
        author_affiliations=author_affiliations,
        articles=articles,
        resolved_authors=resolved_authors,
        max_workers=max_workers,
    )
    affiliation_history, author_summary = analyze_authors(
        article_author_df=article_author_df,
        resolved_authors=resolved_authors,
        affiliations=normalized_affiliations,
        enable_university_affiliation_inbreeding_detection=(
            enable_university_affiliation_inbreeding_detection
        ),
        first_affiliation_publication_count=first_affiliation_publication_count,
        early_affiliation_method=early_affiliation_method,
        first_affiliation_year_count=first_affiliation_year_count,
        min_publication_history_years=min_publication_history_years,
        max_workers=max_workers,
    )
    return InbreedingResult(
        author_affiliations=affiliation_history,
        author_summary=author_summary,
        author_articles=author_article_export(article_author_df),
    )


def analyze_authors(
    *,
    article_author_df: pd.DataFrame,
    resolved_authors: pd.DataFrame,
    affiliations: pd.DataFrame,
    enable_university_affiliation_inbreeding_detection: bool = False,
    first_affiliation_publication_count: int = 3,
    early_affiliation_method: str = EARLY_AFFILIATION_FIRST_N_PUBLICATIONS,
    first_affiliation_year_count: int = 2,
    min_publication_history_years: int | None = None,
    max_workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build affiliation histories and inbreeding proxy rows.

    :param article_author_df: Prepared publication-author-affiliation rows.
    :param resolved_authors: Resolved author profile records.
    :param affiliations: Normalized affiliation metadata.
    :param enable_university_affiliation_inbreeding_detection: Enable the stricter rule.
    :param first_affiliation_publication_count: Early publications to consider.
    :param early_affiliation_method: Method for estimating early affiliations.
    :param first_affiliation_year_count: Years in the early-affiliation window.
    :param min_publication_history_years: Optional minimum history required.
    :param max_workers: Maximum analysis worker processes.
    :return: Affiliation-history and author-summary frames.
    :raises ValueError: If early-affiliation options are invalid.
    """
    validate_early_affiliation_parameters(
        early_affiliation_method=early_affiliation_method,
        first_affiliation_publication_count=first_affiliation_publication_count,
        first_affiliation_year_count=first_affiliation_year_count,
        min_publication_history_years=min_publication_history_years,
    )
    affiliation_metadata = build_affiliation_metadata(affiliations=affiliations)
    resolved_authors = prepare_resolved_authors(resolved_authors=resolved_authors)
    article_author_df = prepare_article_author_frame(article_author_df=article_author_df)
    max_workers = max(1, int(max_workers))
    author_ids = sorted(article_author_df[ARTICLE_AUTHOR_ID_COLUMN].dropna().unique().tolist())
    _logger.info(
        "Analyzing academic inbreeding authors",
        author_count=len(author_ids),
        article_author_rows=len(article_author_df),
        max_workers=max_workers,
    )

    if max_workers == 1 or len(author_ids) <= 1:
        affiliation_rows, author_rows = analyze_author_chunk(
            article_author_chunk=article_author_df,
            resolved_authors=resolved_authors,
            affiliation_metadata=affiliation_metadata,
            enable_university_affiliation_inbreeding_detection=(
                enable_university_affiliation_inbreeding_detection
            ),
            first_affiliation_publication_count=first_affiliation_publication_count,
            early_affiliation_method=early_affiliation_method,
            first_affiliation_year_count=first_affiliation_year_count,
            min_publication_history_years=min_publication_history_years,
        )
        _logger.info(
            "Finished academic inbreeding author analysis",
            author_summary_rows=len(author_rows),
            affiliation_rows=len(affiliation_rows),
        )
        return pd.DataFrame(affiliation_rows), pd.DataFrame(author_rows)

    author_id_chunks = split_author_ids(author_ids=author_ids, max_workers=max_workers)
    _logger.info(
        "Processing academic inbreeding author chunks",
        chunk_count=len(author_id_chunks),
        max_workers=max_workers,
    )

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=initialize_inbreeding_worker,
        initargs=(
            resolved_authors,
            affiliation_metadata,
            enable_university_affiliation_inbreeding_detection,
            first_affiliation_publication_count,
            early_affiliation_method,
            first_affiliation_year_count,
            min_publication_history_years,
        ),
    ) as executor:
        chunk_results = process_author_chunks_bounded(
            executor=executor,
            article_author_df=article_author_df,
            author_id_chunks=author_id_chunks,
            max_workers=max_workers,
        )

    affiliation_rows = [
        row for chunk_affiliations, _ in chunk_results for row in chunk_affiliations
    ]
    author_rows = [row for _, chunk_authors in chunk_results for row in chunk_authors]
    _logger.info(
        "Finished academic inbreeding author analysis",
        author_summary_rows=len(author_rows),
        affiliation_rows=len(affiliation_rows),
    )
    return pd.DataFrame(affiliation_rows), pd.DataFrame(author_rows)


def initialize_inbreeding_worker(
    resolved_authors: pd.DataFrame,
    affiliation_metadata: pd.DataFrame,
    enable_university_affiliation_inbreeding_detection: bool,
    first_affiliation_publication_count: int,
    early_affiliation_method: str,
    first_affiliation_year_count: int,
    min_publication_history_years: int | None,
) -> None:
    """Set process-local inputs and options used by chunk workers.

    :param resolved_authors: Resolved author profiles.
    :param affiliation_metadata: Normalized affiliation records.
    :param enable_university_affiliation_inbreeding_detection: Enable the stricter rule.
    :param first_affiliation_publication_count: Early publications to consider.
    :param early_affiliation_method: Method for estimating early affiliations.
    :param first_affiliation_year_count: Years in the early-affiliation window.
    :param min_publication_history_years: Optional minimum history required.
    """
    global _WORKER_RESOLVED_AUTHORS
    global _WORKER_AFFILIATION_METADATA
    global _WORKER_ENABLE_UNIVERSITY_AFFILIATION_INBREEDING_DETECTION
    global _WORKER_FIRST_AFFILIATION_PUBLICATION_COUNT
    global _WORKER_EARLY_AFFILIATION_METHOD
    global _WORKER_FIRST_AFFILIATION_YEAR_COUNT
    global _WORKER_MIN_PUBLICATION_HISTORY_YEARS
    _WORKER_RESOLVED_AUTHORS = resolved_authors
    _WORKER_AFFILIATION_METADATA = affiliation_metadata
    _WORKER_ENABLE_UNIVERSITY_AFFILIATION_INBREEDING_DETECTION = (
        enable_university_affiliation_inbreeding_detection
    )
    _WORKER_FIRST_AFFILIATION_PUBLICATION_COUNT = first_affiliation_publication_count
    _WORKER_EARLY_AFFILIATION_METHOD = early_affiliation_method
    _WORKER_FIRST_AFFILIATION_YEAR_COUNT = first_affiliation_year_count
    _WORKER_MIN_PUBLICATION_HISTORY_YEARS = min_publication_history_years


def analyze_author_chunk_worker(
    article_author_chunk: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Analyze a worker chunk using process-local configuration.

    :param article_author_chunk: Publication rows for a subset of authors.
    :return: Affiliation and author-summary record lists.
    :raises RuntimeError: If the worker has not been initialized.
    """
    if _WORKER_RESOLVED_AUTHORS is None or _WORKER_AFFILIATION_METADATA is None:
        raise RuntimeError("Academic inbreeding worker was not initialized.")
    return analyze_author_chunk(
        article_author_chunk=article_author_chunk,
        resolved_authors=_WORKER_RESOLVED_AUTHORS,
        affiliation_metadata=_WORKER_AFFILIATION_METADATA,
        enable_university_affiliation_inbreeding_detection=(
            _WORKER_ENABLE_UNIVERSITY_AFFILIATION_INBREEDING_DETECTION
        ),
        first_affiliation_publication_count=_WORKER_FIRST_AFFILIATION_PUBLICATION_COUNT,
        early_affiliation_method=_WORKER_EARLY_AFFILIATION_METHOD,
        first_affiliation_year_count=_WORKER_FIRST_AFFILIATION_YEAR_COUNT,
        min_publication_history_years=_WORKER_MIN_PUBLICATION_HISTORY_YEARS,
    )


def split_author_ids(author_ids: list[int], max_workers: int) -> list[list[int]]:
    """Partition author identifiers into balanced worker chunks.

    :param author_ids: Author identifiers to partition.
    :param max_workers: Maximum number of worker processes.
    :return: Ordered identifier chunks.
    """
    if not author_ids:
        return []
    target_chunk_count = min(len(author_ids), max_workers * 4)
    chunk_size = max(1, math.ceil(len(author_ids) / target_chunk_count))
    return [
        author_ids[index : index + chunk_size] for index in range(0, len(author_ids), chunk_size)
    ]


def article_chunk_for_author_ids(
    *,
    article_author_df: pd.DataFrame,
    author_ids: list[int],
) -> pd.DataFrame:
    """Select article-author rows for the requested identifiers.

    :param article_author_df: Complete prepared article-author frame.
    :param author_ids: Author identifiers to select.
    :return: Copy containing matching rows.
    """
    return article_author_df.loc[
        article_author_df[ARTICLE_AUTHOR_ID_COLUMN].isin(author_ids)
    ].copy()


def process_author_chunks_bounded(
    *,
    executor: ProcessPoolExecutor,
    article_author_df: pd.DataFrame,
    author_id_chunks: list[list[int]],
    max_workers: int,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Submit bounded author chunks and return results in input order.

    :param executor: Process pool for chunk analysis.
    :param article_author_df: Complete prepared article-author frame.
    :param author_id_chunks: Ordered chunks of author identifiers.
    :param max_workers: Maximum number of in-flight tasks.
    :return: Per-chunk affiliation and author-summary records.
    """
    results: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    next_chunk_index = 0
    futures = {}

    def submit_next() -> None:
        """Submit the next available author chunk, if any remain."""
        nonlocal next_chunk_index
        if next_chunk_index >= len(author_id_chunks):
            return
        chunk = article_chunk_for_author_ids(
            article_author_df=article_author_df,
            author_ids=author_id_chunks[next_chunk_index],
        )
        future = executor.submit(analyze_author_chunk_worker, chunk)
        futures[future] = next_chunk_index
        next_chunk_index += 1

    for _ in range(min(max_workers, len(author_id_chunks))):
        submit_next()

    while futures:
        for future in as_completed(list(futures)):
            chunk_index = futures.pop(future)
            results[chunk_index] = future.result()
            submit_next()
            break

    return [results[index] for index in range(len(author_id_chunks))]


def analyze_author_chunk(
    *,
    article_author_chunk: pd.DataFrame,
    resolved_authors: pd.DataFrame,
    affiliation_metadata: pd.DataFrame,
    enable_university_affiliation_inbreeding_detection: bool,
    first_affiliation_publication_count: int,
    early_affiliation_method: str,
    first_affiliation_year_count: int,
    min_publication_history_years: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    affiliation_rows: list[dict[str, Any]] = []
    author_rows: list[dict[str, Any]] = []
    for author_id, group in article_author_chunk.groupby(ARTICLE_AUTHOR_ID_COLUMN, sort=True):
        author_match = resolved_authors.loc[resolved_authors[ARTICLE_AUTHOR_ID_COLUMN] == author_id]
        prepared_group = prepare_author_publications(
            group=group,
            affiliation_metadata=affiliation_metadata,
        )
        if author_match.empty or prepared_group.empty:
            continue

        for _, match_row in author_match.iterrows():
            single_author_match = match_row.to_frame().T
            author_rows.append(
                build_author_summary_row(
                    author_id=author_id,
                    author_match=single_author_match,
                    raw_group=group,
                    prepared_group=prepared_group,
                    enable_university_affiliation_inbreeding_detection=(
                        enable_university_affiliation_inbreeding_detection
                    ),
                    first_affiliation_publication_count=first_affiliation_publication_count,
                    early_affiliation_method=early_affiliation_method,
                    first_affiliation_year_count=first_affiliation_year_count,
                    min_publication_history_years=min_publication_history_years,
                )
            )
            affiliation_rows.extend(
                build_author_affiliation_rows(
                    author_id=author_id,
                    author_match=single_author_match,
                    prepared_group=prepared_group,
                )
            )
    return affiliation_rows, author_rows


def validate_early_affiliation_parameters(
    *,
    early_affiliation_method: str,
    first_affiliation_publication_count: int,
    first_affiliation_year_count: int,
    min_publication_history_years: int | None,
) -> None:
    """Validate supported early-affiliation options.

    :param early_affiliation_method: Selected estimation method.
    :param first_affiliation_publication_count: Number of early publications.
    :param first_affiliation_year_count: Number of early years.
    :param min_publication_history_years: Optional minimum history length.
    :raises ValueError: If a method or numeric option is unsupported.
    """
    if early_affiliation_method not in EARLY_AFFILIATION_METHODS:
        allowed = ", ".join(sorted(EARLY_AFFILIATION_METHODS))
        raise ValueError(
            f"early_affiliation_method must be one of {allowed}; got "
            f"{early_affiliation_method!r}"
        )
    if first_affiliation_publication_count < 1:
        raise ValueError("first_affiliation_publication_count must be at least 1")
    if first_affiliation_year_count < 1:
        raise ValueError("first_affiliation_year_count must be at least 1")
    if min_publication_history_years is not None and min_publication_history_years < 0:
        raise ValueError("min_publication_history_years must be greater than or equal to 0")


def build_affiliation_metadata(affiliations: pd.DataFrame) -> pd.DataFrame:
    """Select available affiliation metadata and retain one row per ID.

    :param affiliations: Source affiliation table.
    :return: Deduplicated analysis metadata.
    """
    existing_columns = [
        column for column in _AFFILIATION_METADATA_COLUMNS if column in affiliations
    ]
    return affiliations[existing_columns].drop_duplicates(subset=[AFFILIATION_ID_COLUMN]).copy()


def prepare_resolved_authors(resolved_authors: pd.DataFrame) -> pd.DataFrame:
    """Expand resolved profiles into numeric mobility author identifiers.

    :param resolved_authors: Selected author profile rows.
    :return: Deduplicated records keyed by numeric mobility author ID.
    :raises KeyError: If no supported author identity column is available.
    """
    resolved_authors = add_mobility_author_profile_ids(resolved_authors)
    frames: list[pd.DataFrame] = []
    if MOBILITY_AUTHOR_PROFILE_ID_COLUMN in resolved_authors.columns:
        frame = resolved_authors.copy()
        frame[ARTICLE_AUTHOR_ID_COLUMN] = frame[MOBILITY_AUTHOR_PROFILE_ID_COLUMN]
        frames.append(frame)
    for column in (DISAMBIGUATED_AUTHOR_ID_COLUMN, AUTHOR_SCOPUS_ID_COLUMN):
        if column in resolved_authors.columns:
            frame = resolved_authors.copy()
            frame[ARTICLE_AUTHOR_ID_COLUMN] = frame[column]
            frames.append(frame)
            break
    if not frames:
        raise KeyError(
            f"Resolved authors must contain '{DISAMBIGUATED_AUTHOR_ID_COLUMN}' or "
            f"'{AUTHOR_SCOPUS_ID_COLUMN}'."
        )
    resolved_authors = pd.concat(frames, ignore_index=True)
    resolved_authors = resolved_authors.loc[
        resolved_authors[ARTICLE_AUTHOR_ID_COLUMN].notna()
    ].copy()
    resolved_authors[ARTICLE_AUTHOR_ID_COLUMN] = pd.to_numeric(
        resolved_authors[ARTICLE_AUTHOR_ID_COLUMN], errors="coerce"
    )
    resolved_authors = resolved_authors.dropna(subset=[ARTICLE_AUTHOR_ID_COLUMN]).copy()
    resolved_authors[ARTICLE_AUTHOR_ID_COLUMN] = resolved_authors[ARTICLE_AUTHOR_ID_COLUMN].astype(
        "int64"
    )
    dedupe_columns = [
        column
        for column in (ARTICLE_AUTHOR_ID_COLUMN, RESEARCHER_ID_COLUMN_NAME)
        if column in resolved_authors.columns
    ]
    return resolved_authors.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)


def prepare_article_author_frame(article_author_df: pd.DataFrame) -> pd.DataFrame:
    """Coerce core article-author fields and discard incomplete rows.

    :param article_author_df: Raw exploded article-author rows.
    :return: Clean rows with numeric identifiers and parsed dates.
    """
    article_author_df = article_author_df.copy()
    article_author_df[ARTICLE_AUTHOR_ID_COLUMN] = pd.to_numeric(
        article_author_df[ARTICLE_AUTHOR_ID_COLUMN], errors="coerce"
    )
    article_author_df[ARTICLE_AFFILIATION_ID_COLUMN] = pd.to_numeric(
        article_author_df[ARTICLE_AFFILIATION_ID_COLUMN], errors="coerce"
    )
    article_author_df[ARTICLE_COVER_DATE_COLUMN] = pd.to_datetime(
        article_author_df[ARTICLE_COVER_DATE_COLUMN], errors="coerce"
    )
    article_author_df = article_author_df.dropna(
        subset=[ARTICLE_AUTHOR_ID_COLUMN, ARTICLE_AFFILIATION_ID_COLUMN, ARTICLE_COVER_DATE_COLUMN]
    ).copy()
    article_author_df[ARTICLE_AUTHOR_ID_COLUMN] = article_author_df[
        ARTICLE_AUTHOR_ID_COLUMN
    ].astype("int64")
    article_author_df[ARTICLE_AFFILIATION_ID_COLUMN] = article_author_df[
        ARTICLE_AFFILIATION_ID_COLUMN
    ].astype("int64")
    return article_author_df


def prepare_author_publications(
    group: pd.DataFrame,
    affiliation_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Join affiliation metadata and order one author's publication rows.

    :param group: Publication rows for one author.
    :param affiliation_metadata: Normalized affiliation lookup table.
    :return: Joined rows with harmonized institution keys.
    """
    merged = group.merge(
        affiliation_metadata,
        how="left",
        left_on=ARTICLE_AFFILIATION_ID_COLUMN,
        right_on=AFFILIATION_ID_COLUMN,
    )
    parent_ids = pd.to_numeric(merged[AFFILIATION_ID_PARENT_COLUMN], errors="coerce")
    affiliation_ids = pd.to_numeric(merged[ARTICLE_AFFILIATION_ID_COLUMN], errors="coerce")
    merged[MOBILITY_INSTITUTION_KEY_COLUMN] = parent_ids.where(parent_ids.notna(), affiliation_ids)
    return merged.sort_values(
        [ARTICLE_COVER_DATE_COLUMN, ARTICLE_AFFILIATION_ID_COLUMN],
        ascending=[False, False],
    )


def institution_sets(
    group: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp, set[Any], set[Any], set[Any]]:
    """Summarize the first, last, and shared institution sets.

    :param group: Prepared publication rows for one author.
    :return: Boundary dates, boundary institution sets, and their intersection.
    """
    first_date = group[ARTICLE_COVER_DATE_COLUMN].min()
    last_date = group[ARTICLE_COVER_DATE_COLUMN].max()
    first_set = institution_set_at_date(group=group, date=first_date)
    last_set = institution_set_at_date(group=group, date=last_date)
    return first_date, last_date, first_set, last_set, first_set & last_set


def institution_set_at_date(group: pd.DataFrame, date: pd.Timestamp) -> set[Any]:
    """Return institutions recorded on the given date.

    :param group: Prepared publication rows for one author.
    :param date: Date whose institution set is requested.
    :return: Nonmissing institution identifiers.
    """
    return set(
        group.loc[group[ARTICLE_COVER_DATE_COLUMN] == date, MOBILITY_INSTITUTION_KEY_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )


def affiliation_set_at_date(
    group: pd.DataFrame,
    date: pd.Timestamp,
    *,
    column: str = ARTICLE_AFFILIATION_ID_COLUMN,
) -> set[int]:
    """Return normalized affiliation IDs recorded on a given date.

    :param group: Prepared publication rows.
    :param date: Date whose affiliations are requested.
    :param column: Affiliation identifier field.
    :return: Integral affiliation identifiers.
    """
    if column not in group.columns or pd.isna(date):
        return set()
    return normalized_int_set(group.loc[group[ARTICLE_COVER_DATE_COLUMN] == date, column])


def normalized_int_set(values: Any) -> set[int]:
    """Convert scalar or collection values to a set of integral numbers.

    :param values: Values to normalize.
    :return: Nonmissing integral values.
    """
    if values is None:
        return set()
    if isinstance(values, pd.Series):
        iterable = values.tolist()
    elif isinstance(values, (list, tuple, set)):
        iterable = list(values)
    else:
        iterable = [values]

    result: set[int] = set()
    for value in iterable:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass

        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            continue
        if float(numeric).is_integer():
            result.add(int(numeric))
    return result


def scopus_affiliation_ids_for_university(university_name: Any) -> list[int]:
    """Look up Scopus affiliation IDs for a university name.

    :param university_name: University name to match.
    :return: Sorted unique affiliation IDs, or an empty list if unmatched.
    """
    if university_name is None:
        return []

    name = str(university_name).strip()
    if not name or name.lower() in {"nan", "none", "null"}:
        return []

    mapped_ids = SCOPUS_ID_MAPPING.get(name)
    if mapped_ids is None:
        normalized_name = name.casefold()
        mapped_ids = next(
            (
                ids
                for mapped_name, ids in SCOPUS_ID_MAPPING.items()
                if mapped_name.casefold() == normalized_name
            ),
            [],
        )
    return sorted(normalized_int_set(mapped_ids))


def scopus_affiliation_ids_for_universities(university_names: list[str]) -> list[int]:
    """Combine Scopus affiliation IDs for multiple university names.

    :param university_names: University names to look up.
    :return: Sorted unique affiliation IDs.
    """
    affiliation_ids: set[int] = set()
    for university_name in university_names:
        affiliation_ids.update(scopus_affiliation_ids_for_university(university_name))
    return sorted(affiliation_ids)


def professor_university_names(author_match: pd.DataFrame) -> list[str]:
    """Get co-affiliated university names, falling back to the university field.

    :param author_match: Matching profile rows.
    :return: Normalized university names.
    """
    universities = parse_text_list(
        first_value(frame=author_match, column=COAFFILIATED_PROFESSOR_UNIVERSITIES_COLUMN)
    )
    if universities:
        return universities
    return parse_text_list(first_value(frame=author_match, column=UNIVERSITY_NAME_COLUMN))


def parse_text_list(value: Any) -> list[str]:
    """Parse and deduplicate nonempty text values while preserving order.

    :param value: Scalar, sequence, or serialized sequence.
    :return: Unique normalized text values.
    """
    parsed = parse_literal(value=value)
    values = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item is None or pd.isna(item):
            continue
        text = str(item).strip()
        key = text.casefold()
        if text and key not in {"nan", "none", "null"} and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def first_publication_affiliation_estimate(
    group: pd.DataFrame,
    *,
    publication_count: int,
) -> tuple[list[int], dict[int, int], int, str]:
    """Estimate early affiliations from the first distinct publications.

    :param group: Publication-affiliation rows for one author.
    :param publication_count: Number of earliest publications to inspect.
    :return: Selected IDs, occurrence counts, observed publication count, and status.
    :raises ValueError: If ``publication_count`` is less than one.
    """
    if publication_count < 1:
        raise ValueError("first_affiliation_publication_count must be at least 1")
    if EID_COLUMN not in group.columns:
        return [], {}, 0, "manual_review_missing_eid_column"

    required_columns = [EID_COLUMN, ARTICLE_COVER_DATE_COLUMN, ARTICLE_AFFILIATION_ID_COLUMN]
    optional_columns = [
        column
        for column in (ARTICLE_DOI, ARTICLE_SOURCE_ID_COLUMN, ARTICLE_PUBLICATION_NAME)
        if column in group.columns
    ]
    publication_rows = group[required_columns + optional_columns].dropna(
        subset=[EID_COLUMN, ARTICLE_COVER_DATE_COLUMN, ARTICLE_AFFILIATION_ID_COLUMN]
    )
    if publication_rows.empty:
        return [], {}, 0, "manual_review_no_publication_affiliations"

    publication_rows = publication_rows.assign(
        **{_PUBLICATION_EVENT_KEY_COLUMN: conservative_publication_event_keys(publication_rows)}
    )

    publications = (
        publication_rows[[_PUBLICATION_EVENT_KEY_COLUMN, ARTICLE_COVER_DATE_COLUMN, EID_COLUMN]]
        .drop_duplicates(subset=[_PUBLICATION_EVENT_KEY_COLUMN])
        .sort_values([ARTICLE_COVER_DATE_COLUMN, EID_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )
    observed_publications = len(publications)
    if observed_publications < publication_count:
        return [], {}, observed_publications, "manual_review_not_enough_publications"

    first_publications = publications.head(publication_count).copy()
    first_publications[_PUBLICATION_ORDER_COLUMN] = range(len(first_publications))

    affiliations = publication_rows.merge(
        first_publications[[_PUBLICATION_EVENT_KEY_COLUMN, _PUBLICATION_ORDER_COLUMN]],
        on=_PUBLICATION_EVENT_KEY_COLUMN,
        how="inner",
    ).copy()
    affiliations[ARTICLE_AFFILIATION_ID_COLUMN] = pd.to_numeric(
        affiliations[ARTICLE_AFFILIATION_ID_COLUMN], errors="coerce"
    )
    affiliations = affiliations.dropna(subset=[ARTICLE_AFFILIATION_ID_COLUMN])
    if affiliations.empty:
        return [], {}, observed_publications, "manual_review_no_first_affiliations"

    affiliations[ARTICLE_AFFILIATION_ID_COLUMN] = affiliations[
        ARTICLE_AFFILIATION_ID_COLUMN
    ].astype("int64")
    publication_affiliations = affiliations.drop_duplicates(
        subset=[_PUBLICATION_EVENT_KEY_COLUMN, ARTICLE_AFFILIATION_ID_COLUMN]
    )

    raw_counts = (
        publication_affiliations[ARTICLE_AFFILIATION_ID_COLUMN]
        .value_counts(sort=False)
        .astype(int)
        .to_dict()
    )
    counts = {int(affiliation_id): int(count) for affiliation_id, count in raw_counts.items()}
    max_count = max(counts.values())
    tied_ids = {affiliation_id for affiliation_id, count in counts.items() if count == max_count}
    first_seen = (
        publication_affiliations[
            publication_affiliations[ARTICLE_AFFILIATION_ID_COLUMN].isin(tied_ids)
        ]
        .groupby(ARTICLE_AFFILIATION_ID_COLUMN)[_PUBLICATION_ORDER_COLUMN]
        .min()
    )
    earliest_order = first_seen.min()
    selected_ids = sorted(
        int(affiliation_id)
        for affiliation_id, order in first_seen.items()
        if order == earliest_order
    )

    status = "assigned"
    if len(tied_ids) > 1:
        status = "assigned_by_first_publication_tie_break"
    if len(selected_ids) > 1:
        status = "assigned_multiple_first_publication_affiliations"

    return selected_ids, dict(sorted(counts.items())), observed_publications, status


def normalized_publication_doi(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    doi = str(value).strip().casefold()
    if doi in {"", "nan", "none", "null"}:
        return ""
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :].strip()
            break
    return doi


def conservative_publication_event_keys(publications: pd.DataFrame) -> pd.Series:
    eid_key = "eid:" + publications[EID_COLUMN].astype(str)
    required_columns = {ARTICLE_DOI, ARTICLE_SOURCE_ID_COLUMN, ARTICLE_PUBLICATION_NAME}
    if not required_columns.issubset(publications.columns):
        return eid_key

    doi = publications[ARTICLE_DOI].map(normalized_publication_doi)
    date = publications[ARTICLE_COVER_DATE_COLUMN].map(normalized_publication_text)
    source = publications[ARTICLE_SOURCE_ID_COLUMN].map(normalized_publication_text)
    title = publications[ARTICLE_PUBLICATION_NAME].map(normalized_publication_text)
    exact_duplicate_key_available = doi.ne("") & date.ne("") & source.ne("") & title.ne("")
    duplicate_key = "doi:" + doi + "|date:" + date + "|source:" + source + "|title:" + title
    return duplicate_key.where(exact_duplicate_key_available, eid_key)


def normalized_publication_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = " ".join(str(value).strip().casefold().split())
    if text in {"", "nan", "none", "null"}:
        return ""
    return text


def publication_rows_for_early_affiliation(group: pd.DataFrame) -> pd.DataFrame:
    if EID_COLUMN not in group.columns:
        return pd.DataFrame()
    required_columns = [EID_COLUMN, ARTICLE_COVER_DATE_COLUMN, ARTICLE_AFFILIATION_ID_COLUMN]
    optional_columns = [
        column
        for column in (ARTICLE_DOI, ARTICLE_SOURCE_ID_COLUMN, ARTICLE_PUBLICATION_NAME)
        if column in group.columns
    ]
    missing_columns = [column for column in required_columns if column not in group.columns]
    if missing_columns:
        return pd.DataFrame()
    publication_rows = group[required_columns + optional_columns].dropna(
        subset=[EID_COLUMN, ARTICLE_COVER_DATE_COLUMN, ARTICLE_AFFILIATION_ID_COLUMN]
    )
    if publication_rows.empty:
        return publication_rows
    return publication_rows.assign(
        **{_PUBLICATION_EVENT_KEY_COLUMN: conservative_publication_event_keys(publication_rows)}
    )


def publication_event_frame(publication_rows: pd.DataFrame) -> pd.DataFrame:
    if publication_rows.empty:
        return pd.DataFrame(
            columns=[_PUBLICATION_EVENT_KEY_COLUMN, ARTICLE_COVER_DATE_COLUMN, EID_COLUMN]
        )
    return (
        publication_rows[[_PUBLICATION_EVENT_KEY_COLUMN, ARTICLE_COVER_DATE_COLUMN, EID_COLUMN]]
        .drop_duplicates(subset=[_PUBLICATION_EVENT_KEY_COLUMN])
        .sort_values([ARTICLE_COVER_DATE_COLUMN, EID_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )


def first_year_affiliation_presence_estimate(
    group: pd.DataFrame,
    *,
    year_count: int,
) -> tuple[list[int], dict[int, int], int, str, int | None, int | None]:
    """Estimate early affiliations from the first calendar-year window.

    :param group: Publication-affiliation rows for one author.
    :param year_count: Number of years in the early window.
    :return: IDs, counts, publication count, status, and window boundary years.
    :raises ValueError: If ``year_count`` is less than one.
    """
    if year_count < 1:
        raise ValueError("first_affiliation_year_count must be at least 1")
    if EID_COLUMN not in group.columns:
        return [], {}, 0, "manual_review_missing_eid_column", None, None

    publication_rows = publication_rows_for_early_affiliation(group)
    if publication_rows.empty:
        return [], {}, 0, "manual_review_no_publication_affiliations", None, None

    publications = publication_event_frame(publication_rows)
    if publications.empty:
        return [], {}, 0, "manual_review_no_publication_affiliations", None, None

    publication_years = pd.to_datetime(
        publications[ARTICLE_COVER_DATE_COLUMN], errors="coerce"
    ).dt.year.dropna()
    if publication_years.empty:
        return [], {}, 0, "manual_review_no_publication_affiliations", None, None

    first_year = int(publication_years.min())
    early_window_end_year = first_year + year_count - 1
    publication_year_by_event = pd.to_datetime(
        publications[ARTICLE_COVER_DATE_COLUMN], errors="coerce"
    ).dt.year
    early_publications = publications.loc[
        publication_year_by_event.between(first_year, early_window_end_year, inclusive="both")
    ].copy()
    observed_publications = len(early_publications)
    if early_publications.empty:
        return (
            [],
            {},
            0,
            "manual_review_no_publication_affiliations",
            first_year,
            early_window_end_year,
        )

    affiliations = publication_rows.merge(
        early_publications[[_PUBLICATION_EVENT_KEY_COLUMN]],
        on=_PUBLICATION_EVENT_KEY_COLUMN,
        how="inner",
    ).copy()
    affiliations[ARTICLE_AFFILIATION_ID_COLUMN] = pd.to_numeric(
        affiliations[ARTICLE_AFFILIATION_ID_COLUMN], errors="coerce"
    )
    affiliations = affiliations.dropna(subset=[ARTICLE_AFFILIATION_ID_COLUMN])
    if affiliations.empty:
        return (
            [],
            {},
            observed_publications,
            "manual_review_no_first_affiliations",
            first_year,
            early_window_end_year,
        )

    affiliations[ARTICLE_AFFILIATION_ID_COLUMN] = affiliations[
        ARTICLE_AFFILIATION_ID_COLUMN
    ].astype("int64")
    publication_affiliations = affiliations.drop_duplicates(
        subset=[_PUBLICATION_EVENT_KEY_COLUMN, ARTICLE_AFFILIATION_ID_COLUMN]
    )
    raw_counts = (
        publication_affiliations[ARTICLE_AFFILIATION_ID_COLUMN]
        .value_counts(sort=False)
        .astype(int)
        .to_dict()
    )
    counts = {int(affiliation_id): int(count) for affiliation_id, count in raw_counts.items()}
    early_affiliation_ids = sorted(counts)
    return (
        early_affiliation_ids,
        dict(sorted(counts.items())),
        observed_publications,
        "assigned",
        first_year,
        early_window_end_year,
    )


def first_publication_year(group: pd.DataFrame) -> int | None:
    """Return the earliest valid publication year in a group.

    :param group: Publication rows.
    :return: Earliest year, or ``None`` when no usable date exists.
    """
    if ARTICLE_COVER_DATE_COLUMN not in group.columns:
        return None
    dates = pd.to_datetime(group[ARTICLE_COVER_DATE_COLUMN], errors="coerce").dropna()
    if dates.empty:
        return None
    return int(dates.dt.year.min())


def detect_inbreeding_proxy(
    *,
    author_match: pd.DataFrame,
    prepared_group: pd.DataFrame,
    first_set: set[Any],
    shared_set: set[Any],
    enable_university_affiliation_inbreeding_detection: bool,
    first_affiliation_publication_count: int,
    early_affiliation_method: str,
    first_affiliation_year_count: int,
    min_publication_history_years: int | None,
) -> InbreedingDetection:
    first_date, last_date, _, _, _ = institution_sets(group=prepared_group)
    publication_history_years = academic_age_years(first_date=first_date, last_date=last_date)
    observed_first_year = first_publication_year(prepared_group)
    min_history_met = (
        None
        if min_publication_history_years is None
        else publication_history_years is not None
        and publication_history_years >= min_publication_history_years
    )

    if normalized_bool(first_value(frame=author_match, column=AUTHOR_PROFILE_AMBIGUOUS_COLUMN)):
        return InbreedingDetection(
            is_inbreeding_proxy=None,
            method=(
                "university_first_publication_affiliation"
                if enable_university_affiliation_inbreeding_detection
                else "first_last_intersection"
            ),
            status="manual_review_overlapping_selected_author_profile",
            manual_review_required=True,
            university_affiliation_ids=[],
            last_university_affiliation_overlap_ids=[],
            estimated_first_affiliation_ids=[],
            estimated_first_affiliation_counts={},
            first_university_affiliation_overlap_ids=[],
            first_affiliation_publication_count_required=first_affiliation_publication_count,
            first_affiliation_publication_count_observed=0,
            early_affiliation_method=early_affiliation_method,
            first_affiliation_year_count=first_affiliation_year_count,
            first_publication_year=observed_first_year,
            early_window_end_year=None,
            n_publications_in_early_window=None,
            publication_history_years=publication_history_years,
            min_publication_history_years=min_publication_history_years,
            min_publication_history_requirement_met=min_history_met,
            target_in_early_window=None,
        )

    if (
        publication_history_years is not None
        and publication_history_years > _MAX_PLAUSIBLE_PUBLICATION_HISTORY_YEARS
    ):
        return InbreedingDetection(
            is_inbreeding_proxy=None,
            method=(
                "university_first_publication_affiliation"
                if enable_university_affiliation_inbreeding_detection
                else "first_last_intersection"
            ),
            status="manual_review_publication_history_too_long",
            manual_review_required=True,
            university_affiliation_ids=[],
            last_university_affiliation_overlap_ids=[],
            estimated_first_affiliation_ids=[],
            estimated_first_affiliation_counts={},
            first_university_affiliation_overlap_ids=[],
            first_affiliation_publication_count_required=first_affiliation_publication_count,
            first_affiliation_publication_count_observed=0,
            early_affiliation_method=early_affiliation_method,
            first_affiliation_year_count=first_affiliation_year_count,
            first_publication_year=observed_first_year,
            early_window_end_year=None,
            n_publications_in_early_window=None,
            publication_history_years=publication_history_years,
            min_publication_history_years=min_publication_history_years,
            min_publication_history_requirement_met=min_history_met,
            target_in_early_window=None,
        )

    if not enable_university_affiliation_inbreeding_detection:
        return InbreedingDetection(
            is_inbreeding_proxy=bool(shared_set),
            method="first_last_intersection",
            status="assigned",
            manual_review_required=False,
            university_affiliation_ids=[],
            last_university_affiliation_overlap_ids=[],
            estimated_first_affiliation_ids=sorted(normalized_int_set(first_set)),
            estimated_first_affiliation_counts={},
            first_university_affiliation_overlap_ids=[],
            first_affiliation_publication_count_required=first_affiliation_publication_count,
            first_affiliation_publication_count_observed=0,
            early_affiliation_method=early_affiliation_method,
            first_affiliation_year_count=first_affiliation_year_count,
            first_publication_year=observed_first_year,
            early_window_end_year=None,
            n_publications_in_early_window=None,
            publication_history_years=publication_history_years,
            min_publication_history_years=min_publication_history_years,
            min_publication_history_requirement_met=min_history_met,
            target_in_early_window=None,
        )

    university_affiliation_ids = scopus_affiliation_ids_for_universities(
        professor_university_names(author_match)
    )
    if not university_affiliation_ids:
        return InbreedingDetection(
            is_inbreeding_proxy=None,
            method="university_first_publication_affiliation",
            status="manual_review_no_university_affiliation_mapping",
            manual_review_required=True,
            university_affiliation_ids=[],
            last_university_affiliation_overlap_ids=[],
            estimated_first_affiliation_ids=[],
            estimated_first_affiliation_counts={},
            first_university_affiliation_overlap_ids=[],
            first_affiliation_publication_count_required=first_affiliation_publication_count,
            first_affiliation_publication_count_observed=0,
            early_affiliation_method=early_affiliation_method,
            first_affiliation_year_count=first_affiliation_year_count,
            first_publication_year=observed_first_year,
            early_window_end_year=None,
            n_publications_in_early_window=None,
            publication_history_years=publication_history_years,
            min_publication_history_years=min_publication_history_years,
            min_publication_history_requirement_met=min_history_met,
            target_in_early_window=None,
        )

    university_set = set(university_affiliation_ids)
    last_affiliation_ids = affiliation_set_at_date(group=prepared_group, date=last_date)
    last_overlap = sorted(last_affiliation_ids & university_set)

    if min_history_met is False:
        return InbreedingDetection(
            is_inbreeding_proxy=None,
            method="university_first_publication_affiliation",
            status="not_classified_publication_history_too_short",
            manual_review_required=False,
            university_affiliation_ids=university_affiliation_ids,
            last_university_affiliation_overlap_ids=last_overlap,
            estimated_first_affiliation_ids=[],
            estimated_first_affiliation_counts={},
            first_university_affiliation_overlap_ids=[],
            first_affiliation_publication_count_required=first_affiliation_publication_count,
            first_affiliation_publication_count_observed=0,
            early_affiliation_method=early_affiliation_method,
            first_affiliation_year_count=first_affiliation_year_count,
            first_publication_year=observed_first_year,
            early_window_end_year=None,
            n_publications_in_early_window=None,
            publication_history_years=publication_history_years,
            min_publication_history_years=min_publication_history_years,
            min_publication_history_requirement_met=False,
            target_in_early_window=None,
        )

    if early_affiliation_method == EARLY_AFFILIATION_FIRST_N_PUBLICATIONS:
        first_ids, counts, observed_count, first_status = first_publication_affiliation_estimate(
            prepared_group,
            publication_count=first_affiliation_publication_count,
        )
        early_window_end_year = None
        n_publications_in_early_window = None
        target_in_early_window = None
    else:
        (
            first_ids,
            counts,
            observed_count,
            first_status,
            observed_first_year,
            early_window_end_year,
        ) = first_year_affiliation_presence_estimate(
            prepared_group,
            year_count=first_affiliation_year_count,
        )
        n_publications_in_early_window = observed_count
        target_in_early_window = (
            bool(set(first_ids) & university_set)
            if not first_status.startswith("manual_review")
            else None
        )

    first_overlap = sorted(set(first_ids) & university_set)
    manual_review_required = first_status.startswith("manual_review")

    if manual_review_required:
        is_inbreeding_proxy = None
        status = first_status
    else:
        is_inbreeding_proxy = bool(last_overlap and first_overlap)
        status = "assigned" if first_status == "assigned" else first_status
        if not last_overlap:
            status = "assigned_no_current_university_affiliation"
        elif not first_overlap:
            status = "assigned_first_affiliation_not_current_university"

    return InbreedingDetection(
        is_inbreeding_proxy=is_inbreeding_proxy,
        method="university_first_publication_affiliation",
        status=status,
        manual_review_required=manual_review_required,
        university_affiliation_ids=university_affiliation_ids,
        last_university_affiliation_overlap_ids=last_overlap,
        estimated_first_affiliation_ids=first_ids,
        estimated_first_affiliation_counts=counts,
        first_university_affiliation_overlap_ids=first_overlap,
        first_affiliation_publication_count_required=first_affiliation_publication_count,
        first_affiliation_publication_count_observed=observed_count,
        early_affiliation_method=early_affiliation_method,
        first_affiliation_year_count=first_affiliation_year_count,
        first_publication_year=observed_first_year,
        early_window_end_year=early_window_end_year,
        n_publications_in_early_window=n_publications_in_early_window,
        publication_history_years=publication_history_years,
        min_publication_history_years=min_publication_history_years,
        min_publication_history_requirement_met=min_history_met,
        target_in_early_window=target_in_early_window,
    )


def detect_publication_mobility_spell(
    prepared_group: pd.DataFrame,
    home_inst_set: Any = None,
    *,
    eid_column: str = EID_COLUMN,
    inst_column: str = MOBILITY_INSTITUTION_KEY_COLUMN,
    date_column: str = ARTICLE_COVER_DATE_COLUMN,
    long_stay_threshold_years: int = 2,
) -> dict[Any, Any]:
    """
    Detect publication-affiliation mobility for one researcher.

    General outputs:
    - unique affiliations observed in the publication history
    - number of unique affiliations
    - number of author-specific co-affiliated publications
    - number of author-specific co-affiliation instances

    ``prepared_group`` must contain only rows for the focal author. Affiliation
    sets are collapsed at the harmonized parent-organization level before
    co-affiliations are counted.

    If home_inst_set is provided:
    - strict mobility is counted only when the home institution is absent
      from a publication.
    - co-affiliations like {A, B} do not count as away if A is home.
    - domestic away years are away years without a non-Austrian country.
    - short mobility: max away spell <= threshold years.
    - long mobility:  max away spell > threshold years.
    """

    empty_result = {
        HAS_PUBLICATION_MOBILITY: False,
        SHORT_PUBLICATION_MOBILITY: False,
        LONG_PUBLICATION_MOBILITY: False,
        MAX_AWAY_SPELL_YEARS: 0,
        N_AWAY_PUBLICATIONS: 0,
        N_AWAY_YEARS: 0,
        N_AWAY_YEARS_DOMESTIC: 0,
        N_AWAY_YEARS_ABROAD: 0,
        HAS_ANY_COAFFILIATION: False,
        N_COAFFILIATION_PUBLICATIONS: 0,
        N_COAFFILIATIONS: 0,
        COAFFILIATION_PUBLICATION_SHARE: pd.NA,
        N_UNIQUE_AFFILIATIONS: 0,
        UNIQUE_AFFILIATIONS: [],
    }

    required_columns = [eid_column, inst_column, date_column]
    missing_columns = [col for col in required_columns if col not in prepared_group.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    tmp = prepared_group.dropna(subset=required_columns).copy()
    if tmp.empty:
        return empty_result.copy()

    tmp[date_column] = pd.to_datetime(tmp[date_column], errors="coerce")
    tmp = tmp.dropna(subset=[date_column])
    if tmp.empty:
        return empty_result.copy()

    country_column = publication_country_column(tmp)
    tmp[_MOBILITY_COUNTRY_SET_COLUMN] = (
        tmp[country_column].map(normalized_country_set) if country_column else [set()] * len(tmp)
    )

    paper_sets = (
        tmp.groupby(eid_column)
        .agg(
            **{
                ARTICLE_COVER_DATE_COLUMN: (date_column, "min"),
                _INSTITUTION_SET_COLUMN: (inst_column, lambda values: set(values.dropna())),
                _COUNTRY_SET_COLUMN: (_MOBILITY_COUNTRY_SET_COLUMN, union_sets),
            }
        )
        .sort_values(ARTICLE_COVER_DATE_COLUMN)
        .reset_index()
    )

    if paper_sets.empty:
        return empty_result.copy()

    paper_sets[_YEAR_COLUMN] = paper_sets[ARTICLE_COVER_DATE_COLUMN].dt.year

    unique_affiliations = sorted(
        {aff for inst_set in paper_sets[_INSTITUTION_SET_COLUMN] for aff in inst_set}
    )

    n_unique_affiliations = len(unique_affiliations)

    paper_sets[_N_AFFILIATIONS_ON_PAPER_COLUMN] = paper_sets[_INSTITUTION_SET_COLUMN].apply(len)

    paper_sets[_IS_COAFFILIATED_COLUMN] = paper_sets[_N_AFFILIATIONS_ON_PAPER_COLUMN] > 1

    n_coaffiliation_publications = int(paper_sets[_IS_COAFFILIATED_COLUMN].sum())

    n_coaffiliations = int(
        paper_sets.loc[paper_sets[_IS_COAFFILIATED_COLUMN], _N_AFFILIATIONS_ON_PAPER_COLUMN]
        .sub(1)
        .sum()
    )
    n_publications_with_known_affiliations = len(paper_sets)

    base_result = {
        **empty_result,
        HAS_ANY_COAFFILIATION: n_coaffiliation_publications > 0,
        N_COAFFILIATION_PUBLICATIONS: n_coaffiliation_publications,
        N_COAFFILIATIONS: n_coaffiliations,
        COAFFILIATION_PUBLICATION_SHARE: safe_ratio(
            n_coaffiliation_publications,
            n_publications_with_known_affiliations,
        ),
        N_UNIQUE_AFFILIATIONS: n_unique_affiliations,
        UNIQUE_AFFILIATIONS: unique_affiliations,
    }

    if not home_inst_set:
        return base_result

    home_inst_set = set(home_inst_set)

    paper_sets[_HAS_HOME_COLUMN] = paper_sets[_INSTITUTION_SET_COLUMN].apply(
        lambda s: bool(set(s) & home_inst_set)
    )

    paper_sets[_IS_AWAY_COLUMN] = ~paper_sets[_HAS_HOME_COLUMN]

    n_away_publications = int(paper_sets[_IS_AWAY_COLUMN].sum())

    away_years = sorted(paper_sets.loc[paper_sets[_IS_AWAY_COLUMN], _YEAR_COLUMN].dropna().unique())

    n_away_years = len(away_years)
    away_abroad_years = sorted(
        paper_sets.loc[
            paper_sets[_IS_AWAY_COLUMN]
            & paper_sets[_COUNTRY_SET_COLUMN].map(has_non_austrian_country),
            _YEAR_COLUMN,
        ]
        .dropna()
        .unique()
    )
    n_away_years_abroad = len(away_abroad_years)
    n_away_years_domestic = len(set(away_years) - set(away_abroad_years))

    result = {
        **base_result,
        N_AWAY_PUBLICATIONS: n_away_publications,
        N_AWAY_YEARS: n_away_years,
        N_AWAY_YEARS_DOMESTIC: n_away_years_domestic,
        N_AWAY_YEARS_ABROAD: n_away_years_abroad,
    }

    if n_away_publications == 0:
        return result

    max_spell = 1
    current_spell = 1

    for previous_year, current_year in pairwise(away_years):
        if current_year == previous_year + 1:
            current_spell += 1
        else:
            max_spell = max(max_spell, current_spell)
            current_spell = 1

    max_spell = max(max_spell, current_spell)

    result.update(
        {
            HAS_PUBLICATION_MOBILITY: True,
            SHORT_PUBLICATION_MOBILITY: max_spell <= long_stay_threshold_years,
            LONG_PUBLICATION_MOBILITY: max_spell > long_stay_threshold_years,
            MAX_AWAY_SPELL_YEARS: int(max_spell),
        }
    )

    return result


def publication_country_column(frame: pd.DataFrame) -> str | None:
    """Choose the available country field for mobility analysis.

    :param frame: Publication-affiliation rows.
    :return: Preferred country column, or ``None`` when absent.
    """
    if ARTICLE_AFFILIATION_COUNTRY_COLUMN in frame.columns:
        return ARTICLE_AFFILIATION_COUNTRY_COLUMN
    if COUNTRY_COLUMN in frame.columns:
        return COUNTRY_COLUMN
    return None


def safe_ratio(numerator: int, denominator: int) -> Any:
    """Divide values, returning pandas' missing marker for a zero denominator.

    :param numerator: Value to divide.
    :param denominator: Divisor.
    :return: Ratio or ``pd.NA``.
    """
    if denominator == 0:
        return pd.NA
    return numerator / denominator


def normalized_country_set(value: Any) -> set[str]:
    """Normalize one or more country values into a case-folded set.

    :param value: Scalar or collection of country names.
    :return: Nonmissing normalized country names.
    """
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value).split(";")

    countries = set()
    for country in values:
        if country is None or pd.isna(country):
            continue
        normalized = str(country).strip().casefold()
        if normalized and normalized not in {"nan", "none", "null"}:
            countries.add(normalized)
    return countries


def union_sets(values: pd.Series) -> set[Any]:
    """Combine set-valued entries from a series.

    :param values: Series containing iterable set values.
    :return: Union of all values.
    """
    result: set[Any] = set()
    for value in values:
        result |= set(value)
    return result


def has_non_austrian_country(countries: set[str]) -> bool:
    """Check whether a country set includes a country other than Austria.

    :param countries: Normalized country names.
    :return: Whether any country is not Austria.
    """
    return any(country != "austria" for country in countries)


def build_author_summary_row(
    author_id: int,
    author_match: pd.DataFrame,
    raw_group: pd.DataFrame,
    prepared_group: pd.DataFrame,
    *,
    enable_university_affiliation_inbreeding_detection: bool = False,
    first_affiliation_publication_count: int = 3,
    early_affiliation_method: str = EARLY_AFFILIATION_FIRST_N_PUBLICATIONS,
    first_affiliation_year_count: int = 2,
    min_publication_history_years: int | None = None,
) -> dict[str, Any]:
    first_date, last_date, first_set, last_set, shared_set = institution_sets(group=prepared_group)
    inbreeding_detection = detect_inbreeding_proxy(
        author_match=author_match,
        prepared_group=prepared_group,
        first_set=first_set,
        shared_set=shared_set,
        enable_university_affiliation_inbreeding_detection=(
            enable_university_affiliation_inbreeding_detection
        ),
        first_affiliation_publication_count=first_affiliation_publication_count,
        early_affiliation_method=early_affiliation_method,
        first_affiliation_year_count=first_affiliation_year_count,
        min_publication_history_years=min_publication_history_years,
    )

    return_mobility = detect_publication_mobility_spell(
        prepared_group=prepared_group,
        home_inst_set=inbreeding_detection.university_affiliation_ids,
        long_stay_threshold_years=2,
    )

    result_dict = {
        RESEARCHER_ID_COLUMN_NAME: first_value(
            frame=author_match, column=RESEARCHER_ID_COLUMN_NAME
        ),
        ARTICLE_AUTHOR_ID_COLUMN: author_id,
        DISAMBIGUATED_AUTHOR_ID_COLUMN: first_value(
            frame=author_match, column=DISAMBIGUATED_AUTHOR_ID_COLUMN
        )
        or author_id,
        DISAMBIGUATED_AUTHOR_IDS_COLUMN: first_value(
            frame=author_match, column=DISAMBIGUATED_AUTHOR_IDS_COLUMN
        ),
        DISAMBIGUATED_AUTHOR_HISTORY_COLUMN: first_value(
            frame=author_match, column=DISAMBIGUATED_AUTHOR_HISTORY_COLUMN
        ),
        MOBILITY_AUTHOR_PROFILE_ID_COLUMN: first_value(
            frame=author_match, column=MOBILITY_AUTHOR_PROFILE_ID_COLUMN
        ),
        AUTHOR_PROFILE_AMBIGUOUS_COLUMN: first_value(
            frame=author_match, column=AUTHOR_PROFILE_AMBIGUOUS_COLUMN
        ),
        AUTHOR_PROFILE_AMBIGUITY_REASON_COLUMN: first_value(
            frame=author_match, column=AUTHOR_PROFILE_AMBIGUITY_REASON_COLUMN
        ),
        PERSON_TITLE_RAW: first_value(frame=author_match, column=PERSON_TITLE_COLUMN),
        TITLE_OUTPUT_COLUMN: final_author_title(author_match),
        MOBILITY_RESEARCHER_NAME: first_value(
            frame=author_match, column=RESEARCHER_PROCESSED_NAME_COLUMN
        ),
        UNIVERSITY_NAME_COLUMN: first_value(frame=author_match, column=UNIVERSITY_NAME_COLUMN),
        PROFESSOR_SC_IDS_COLUMN: first_value(frame=author_match, column=PROFESSOR_SC_IDS_COLUMN),
        PROFESSOR_NAMES_COLUMN: first_value(frame=author_match, column=PROFESSOR_NAMES_COLUMN),
        PROFESSOR_TITLES_COLUMN: first_value(frame=author_match, column=PROFESSOR_TITLES_COLUMN),
        PROFESSOR_TITLE_CLASSES_COLUMN: first_value(
            frame=author_match,
            column=PROFESSOR_TITLE_CLASSES_COLUMN,
        ),
        IS_COAFFILIATED_PROFESSOR_COLUMN: first_value(
            frame=author_match,
            column=IS_COAFFILIATED_PROFESSOR_COLUMN,
        ),
        COAFFILIATED_PROFESSOR_UNIVERSITIES_COLUMN: first_value(
            frame=author_match,
            column=COAFFILIATED_PROFESSOR_UNIVERSITIES_COLUMN,
        ),
        N_COAFFILIATED_PROFESSOR_UNIVERSITIES_COLUMN: first_value(
            frame=author_match,
            column=N_COAFFILIATED_PROFESSOR_UNIVERSITIES_COLUMN,
        ),
        IS_MERGED_PROFESSOR_PROFILE_COLUMN: first_value(
            frame=author_match,
            column=IS_MERGED_PROFESSOR_PROFILE_COLUMN,
        ),
        MERGED_PROFESSOR_ROWS_COLUMN: first_value(
            frame=author_match,
            column=MERGED_PROFESSOR_ROWS_COLUMN,
        ),
        FIRST_DATE: first_date,
        LAST_DATE: last_date,
        ACADEMIC_AGE_YEARS: academic_age_years(first_date=first_date, last_date=last_date),
        FIRST_INST_SET: sorted(first_set),
        LAST_INST_SET: sorted(last_set),
        SHARED_INST_SET: sorted(shared_set),
        IS_INBREEDING_PROXY: inbreeding_detection.is_inbreeding_proxy,
        INBREEDING_DETECTION_METHOD: inbreeding_detection.method,
        INBREEDING_DETECTION_STATUS: inbreeding_detection.status,
        INBREEDING_MANUAL_REVIEW_REQUIRED: inbreeding_detection.manual_review_required,
        UNIVERSITY_AFFILIATION_IDS: inbreeding_detection.university_affiliation_ids,
        LAST_UNIVERSITY_AFFILIATION_OVERLAP_IDS: (
            inbreeding_detection.last_university_affiliation_overlap_ids
        ),
        ESTIMATED_FIRST_AFFILIATION_IDS: inbreeding_detection.estimated_first_affiliation_ids,
        ESTIMATED_FIRST_AFFILIATION_COUNTS: (
            inbreeding_detection.estimated_first_affiliation_counts
        ),
        FIRST_UNIVERSITY_AFFILIATION_OVERLAP_IDS: (
            inbreeding_detection.first_university_affiliation_overlap_ids
        ),
        FIRST_AFFILIATION_PUBLICATION_COUNT_REQUIRED: (
            inbreeding_detection.first_affiliation_publication_count_required
        ),
        FIRST_AFFILIATION_PUBLICATION_COUNT_OBSERVED: (
            inbreeding_detection.first_affiliation_publication_count_observed
        ),
        EARLY_AFFILIATION_METHOD: inbreeding_detection.early_affiliation_method,
        FIRST_AFFILIATION_YEAR_COUNT: inbreeding_detection.first_affiliation_year_count,
        FIRST_PUBLICATION_YEAR: inbreeding_detection.first_publication_year,
        EARLY_WINDOW_END_YEAR: inbreeding_detection.early_window_end_year,
        N_PUBLICATIONS_IN_EARLY_WINDOW: inbreeding_detection.n_publications_in_early_window,
        PUBLICATION_HISTORY_YEARS: inbreeding_detection.publication_history_years,
        MIN_PUBLICATION_HISTORY_YEARS: inbreeding_detection.min_publication_history_years,
        MIN_PUBLICATION_HISTORY_REQUIREMENT_MET: (
            inbreeding_detection.min_publication_history_requirement_met
        ),
        TARGET_IN_EARLY_WINDOW: inbreeding_detection.target_in_early_window,
        AUTHOR_PROFILE_ARTICLE_EIDS: author_article_eids(raw_group),
        N_UNIQUE: unique_publication_count(frame=raw_group),
        **coauthor_summary(frame=raw_group),
        PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS_COLUMN: article_general_asjc_subject_areas(
            frame=raw_group
        ),
        PUBLICATION_ASJC_COUNTS_COLUMN: (
            article_general_asjc_subject_area_publication_counts(frame=raw_group)
        ),
        N_ROWS: len(prepared_group),
        N_AFFILIATIONS: prepared_group[ARTICLE_AFFILIATION_ID_COLUMN].nunique(),
        N_PARENT_INSTITUTIONS: pd.Series(
            prepared_group[MOBILITY_INSTITUTION_KEY_COLUMN].dropna().unique()
        ).nunique(),
        ARTICLE_AFFILIATION_ID_COLUMN: unique_values(
            frame=raw_group, column=ARTICLE_AFFILIATION_ID_COLUMN
        ),
        ARTICLE_FUND_NO_COLUMN: unique_values(frame=raw_group, column=ARTICLE_FUND_NO_COLUMN),
        ARTICLE_FUND_SPONSOR_COLUMN: unique_values(
            frame=raw_group, column=ARTICLE_FUND_SPONSOR_COLUMN
        ),
        ARTICLE_FUND_ACRONYM_COLUMN: unique_values(
            frame=raw_group, column=ARTICLE_FUND_ACRONYM_COLUMN
        ),
        ARTICLE_AFFILIATION_CITY_COLUMN: unique_values(
            frame=raw_group, column=ARTICLE_AFFILIATION_CITY_COLUMN
        ),
        ARTICLE_AFFILIATION_COUNTRY_COLUMN: unique_values(
            frame=raw_group, column=ARTICLE_AFFILIATION_COUNTRY_COLUMN
        ),
        ARTICLE_SOURCE_ID_COLUMN: unique_values(frame=raw_group, column=ARTICLE_SOURCE_ID_COLUMN),
        ARTICLE_ISSN_COLUMN: unique_values(frame=raw_group, column=ARTICLE_ISSN_COLUMN),
        ARTICLE_E_ISSN_COLUMN: unique_values(frame=raw_group, column=ARTICLE_E_ISSN_COLUMN),
    }
    result_dict.update(return_mobility)
    return result_dict


def academic_age_years(first_date: pd.Timestamp, last_date: pd.Timestamp) -> int | None:
    """Calculate the year difference between first and last publication dates.

    :param first_date: Earliest publication date.
    :param last_date: Latest publication date.
    :return: Year difference, or ``None`` if either date is missing.
    """
    if pd.isna(first_date) or pd.isna(last_date):
        return None
    return int(last_date.year - first_date.year)


def build_author_affiliation_rows(
    author_id: int,
    author_match: pd.DataFrame,
    prepared_group: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for affiliation_id, affiliation_group in sorted_affiliation_groups(group=prepared_group):
        affiliation = affiliation_group.iloc[0]
        rows.append(
            {
                RESEARCHER_ID_COLUMN_NAME: first_value(
                    frame=author_match, column=RESEARCHER_ID_COLUMN_NAME
                ),
                ARTICLE_AUTHOR_ID_COLUMN: author_id,
                MOBILITY_RESEARCHER_NAME: first_value(
                    frame=author_match, column=RESEARCHER_PROCESSED_NAME_COLUMN
                ),
                AFFILIATION_ID_COLUMN: affiliation_id,
                AFFILIATION_ID_PARENT_COLUMN: affiliation.get(AFFILIATION_ID_PARENT_COLUMN),
                AFFILIATION_NAME_COLUMN: affiliation.get(AFFILIATION_PREFERRED_NAME_COLUMN),
                AUTHOR_AFFILIATION_TYPE_COLUMN: affiliation.get(AFFILIATION_TYPE_COLUMN),
                ADDRESS_COLUMN: affiliation.get(AFFILIATION_ADDRESS_COLUMN),
                CITY_COLUMN: affiliation.get(CITY_COLUMN),
                COUNTRY_COLUMN: affiliation.get(COUNTRY_COLUMN),
                DATE_FIRST: affiliation_group[ARTICLE_COVER_DATE_COLUMN].min(),
                DATE_LAST: affiliation_group[ARTICLE_COVER_DATE_COLUMN].max(),
                N_ROWS: len(affiliation_group),
            }
        )
    return rows


def sorted_affiliation_groups(group: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    """Group affiliation rows and order groups by latest observed date.

    :param group: Prepared publication-affiliation rows.
    :return: Affiliation IDs and corresponding row groups, newest first.
    """
    grouped = group.groupby(ARTICLE_AFFILIATION_ID_COLUMN, sort=False)
    ordered_ids = grouped[ARTICLE_COVER_DATE_COLUMN].max().sort_values(ascending=False).index
    return [
        (int(affiliation_id), grouped.get_group(affiliation_id)) for affiliation_id in ordered_ids
    ]


def unique_values(frame: pd.DataFrame, column: str) -> list[Any]:
    """Return distinct nonmissing values from an available column.

    :param frame: Source records.
    :param column: Column to read.
    :return: Distinct values in source order, or an empty list if absent.
    """
    if column not in frame.columns:
        return []
    return frame[column].dropna().drop_duplicates().tolist()


def unique_publication_count(frame: pd.DataFrame) -> int:
    """Count distinct publication identifiers in a frame.

    :param frame: Publication records.
    :return: Number of distinct nonmissing publications.
    """
    if EID_COLUMN not in frame.columns:
        return 0
    return int(frame[EID_COLUMN].dropna().nunique())


def author_article_eids(frame: pd.DataFrame) -> list[Any]:
    """List unique article identifiers in publication date order.

    :param frame: Article-author rows.
    :return: First occurrence of each article identifier.
    """
    if EID_COLUMN not in frame.columns:
        return []
    articles = frame.dropna(subset=[EID_COLUMN]).copy()
    if articles.empty:
        return []

    sort_columns = [
        column for column in [ARTICLE_COVER_DATE_COLUMN, EID_COLUMN] if column in articles
    ]
    if sort_columns:
        articles = articles.sort_values(sort_columns, kind="mergesort")

    result = []
    seen = set()
    for eid in articles[EID_COLUMN].tolist():
        key = str(eid)
        if key in seen:
            continue
        seen.add(key)
        result.append(eid)
    return result


def author_article_export(frame: pd.DataFrame) -> pd.DataFrame:
    """Group affiliation rows into one author-article row with delimited IDs.

    :param frame: Prepared publication-author-affiliation rows.
    :return: One row per author and article, retaining resolved and original IDs.
    :raises ValueError: If an exported author identifier is not integral.
    """
    if EID_COLUMN not in frame.columns or ARTICLE_AUTHOR_ID_COLUMN not in frame.columns:
        return pd.DataFrame()
    article_rows = frame.dropna(subset=[EID_COLUMN, ARTICLE_AUTHOR_ID_COLUMN]).copy()
    author_ids = pd.to_numeric(article_rows[ARTICLE_AUTHOR_ID_COLUMN], errors="raise")
    if not author_ids.mod(1).eq(0).all():
        raise ValueError("Author identifiers in the article export must be integral.")
    article_rows[ARTICLE_AUTHOR_ID_COLUMN] = author_ids.astype("int64")
    if article_rows.empty:
        return article_rows
    sort_columns = [
        column for column in [ARTICLE_COVER_DATE_COLUMN, EID_COLUMN] if column in article_rows
    ]
    if sort_columns:
        article_rows = article_rows.sort_values(sort_columns, kind="mergesort")

    grouped_rows: list[dict[str, Any]] = []
    for _, group in article_rows.groupby(
        [ARTICLE_AUTHOR_ID_COLUMN, EID_COLUMN], sort=False, dropna=False
    ):
        row = group.iloc[0].to_dict()
        row[ARTICLE_AFFILIATION_ID_COLUMN] = (
            _joined_unique_ids(group[ARTICLE_AFFILIATION_ID_COLUMN])
            if ARTICLE_AFFILIATION_ID_COLUMN in group
            else ""
        )
        row[UNMAPPED_AFFILIATION_ID] = (
            _joined_unique_ids(group[UNMAPPED_AFFILIATION_ID], split_delimited_values=True)
            if UNMAPPED_AFFILIATION_ID in group
            else ""
        )
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows, columns=article_rows.columns).reset_index(drop=True)


def _joined_unique_ids(values: pd.Series, *, split_delimited_values: bool = False) -> str:
    """Join unique identifier values in encounter order.

    :param values: Identifier values, including optional lists of IDs.
    :param split_delimited_values: Split original source values on hyphens.
    :return: Semicolon-delimited identifier string.
    """
    identifiers: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple)):
            items = value
        elif pd.isna(value):
            continue
        else:
            items = [value]
        for item in items:
            if pd.isna(item):
                continue
            tokens = str(item).split("-") if split_delimited_values else [str(item)]
            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                try:
                    numeric_value = float(token)
                    if numeric_value.is_integer():
                        token = str(int(numeric_value))
                except ValueError:
                    pass
                if token not in seen:
                    seen.add(token)
                    identifiers.append(token)
    return ";".join(identifiers)


def coauthor_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize total and mean coauthors across distinct publications.

    :param frame: Article-author rows with publication author counts.
    :return: Summary fields, containing missing markers when unavailable.
    """
    empty = {
        TOTAL_COAUTHORS_ACROSS_PUBLICATIONS: pd.NA,
        MEAN_COAUTHORS_PER_PUBLICATION: pd.NA,
    }
    required_columns = {EID_COLUMN, PUBLICATION_AUTHOR_COUNT_COLUMN}
    if not required_columns.issubset(frame.columns):
        return empty

    publication_counts = frame[[EID_COLUMN, PUBLICATION_AUTHOR_COUNT_COLUMN]].dropna(
        subset=[EID_COLUMN]
    )
    publication_counts[PUBLICATION_AUTHOR_COUNT_COLUMN] = pd.to_numeric(
        publication_counts[PUBLICATION_AUTHOR_COUNT_COLUMN], errors="coerce"
    )
    publication_counts = publication_counts.dropna(subset=[PUBLICATION_AUTHOR_COUNT_COLUMN])
    publication_counts = publication_counts.drop_duplicates(subset=[EID_COLUMN])
    if publication_counts.empty:
        return empty

    coauthors = (publication_counts[PUBLICATION_AUTHOR_COUNT_COLUMN] - 1).clip(lower=0)
    total_coauthors = int(coauthors.sum())
    publication_count = len(publication_counts)
    return {
        TOTAL_COAUTHORS_ACROSS_PUBLICATIONS: total_coauthors,
        MEAN_COAUTHORS_PER_PUBLICATION: total_coauthors / publication_count,
    }


def article_general_asjc_subject_areas(frame: pd.DataFrame) -> dict[int, str]:
    """Combine ASJC level-two labels present in article records.

    :param frame: Article-author rows.
    :return: Sorted mapping from subject codes to labels.
    """
    if PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS_COLUMN not in frame.columns:
        return {}

    result: dict[int, str] = {}
    for value in frame[PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS_COLUMN].dropna():
        if isinstance(value, dict):
            for code, label in value.items():
                try:
                    result[int(code)] = str(label)
                except (TypeError, ValueError):
                    continue
    return dict(sorted(result.items()))


def article_general_asjc_subject_area_publication_counts(
    frame: pd.DataFrame,
) -> dict[int, dict[str, int | str]]:
    """Count distinct publications by ASJC level-two subject area.

    :param frame: Article-author rows with publication-level ASJC summaries.
    :return: Sorted code-to-category-and-count mapping.
    """
    if (
        EID_COLUMN not in frame.columns
        or PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS_COLUMN not in frame.columns
    ):
        return {}

    publication_subjects = frame[
        [EID_COLUMN, PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS_COLUMN]
    ].dropna(subset=[EID_COLUMN])
    publication_subjects = publication_subjects.drop_duplicates(subset=[EID_COLUMN])

    counts: dict[int, int] = {}
    labels: dict[int, str] = {}
    for value in publication_subjects[PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS_COLUMN].dropna():
        if not isinstance(value, dict):
            continue
        for code, label in value.items():
            try:
                code_int = int(code)
            except (TypeError, ValueError):
                continue
            counts[code_int] = counts.get(code_int, 0) + 1
            labels[code_int] = str(label)

    return {code: {"category": labels[code], "n": counts[code]} for code in sorted(counts)}


def first_value(frame: pd.DataFrame, column: str) -> Any:
    """Return the first nonmissing value from an available column.

    :param frame: Source records.
    :param column: Field to inspect.
    :return: First nonmissing value, or ``None`` when unavailable.
    """
    if column not in frame.columns or frame.empty:
        return None
    values = frame[column].dropna()
    return values.iloc[0] if not values.empty else None


def normalized_bool(value: Any) -> bool:
    """Interpret common affirmative scalar spellings as true.

    :param value: Value to interpret.
    :return: Whether the value represents a true value.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    text = str(value).strip().casefold()
    return text in {"1", "true", "yes", "y"}


def final_author_title(author_match: pd.DataFrame) -> str:
    """Resolve the final output title using available university metadata.

    :param author_match: Matching profile records.
    :return: Institution-adjusted title classification.
    """
    universities = first_value(
        frame=author_match, column=COAFFILIATED_PROFESSOR_UNIVERSITIES_COLUMN
    )
    if universities is None:
        universities = first_value(frame=author_match, column=UNIVERSITY_NAME_COLUMN)
    title = first_value(frame=author_match, column=TITLE_OUTPUT_COLUMN)
    if title is not None and str(title).strip():
        return adjust_title_for_institution(str(title).strip(), universities=universities)
    return classify_title_for_institution(
        first_value(frame=author_match, column=PERSON_TITLE_COLUMN),
        universities=universities,
    )


def write_inbreeding_outputs(result: InbreedingResult, outputs: MobilityOutputs) -> None:
    """Write the three mobility analysis artifacts to CSV files.

    :param result: Tabular analysis outputs.
    :param outputs: Explicit paths for the three mobility artifacts.
    """
    artifacts = (
        (result.author_affiliations, outputs.author_affiliations),
        (result.author_summary, outputs.author_inbreeding_proxy),
        (result.author_articles, outputs.author_articles),
    )
    for frame, path in artifacts:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)


def write_unique_affiliation_chunks(
    affiliations: pd.DataFrame,
    output_path: Path,
    *,
    chunk_size: int = 1000,
) -> None:
    """Write affiliation rows as numbered CSV chunks.

    :param affiliations: Rows to write.
    :param output_path: Base path used to derive numbered chunk filenames.
    :param chunk_size: Maximum rows per output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for chunk_index in range(math.ceil(len(affiliations) / chunk_size)):
        chunk = affiliations.iloc[chunk_index * chunk_size : (chunk_index + 1) * chunk_size]
        chunk.to_csv(output_path.with_stem(f"{output_path.stem}_{chunk_index + 1}"), index=False)
