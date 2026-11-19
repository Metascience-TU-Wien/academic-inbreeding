# Copyright © 2026 TU Wien

"""Adapt the configured OpenAlex Works and institution tables for mobility."""

from __future__ import annotations

import ast
import re
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import duckdb  # type: ignore[import-not-found]
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from acib.constants import AFFILIATION_NAME, EID
from acib.mobility.constants import (
    AFFILIATION_PREFERRED_NAME,
    ARTICLE_AFFILIATION_ID,
    ARTICLE_COVER_DATE,
)
from acib.name_disambiguation.constants import (
    AUTHOR_IDS,
    CURRENT_AFFILIATION_IDS,
    SELECTED_AUTHOR_IDS,
)

_OPENALEX_AUTHOR_ID_PATTERN = re.compile(r"^https://openalex\.org/A\d+$")
_WORK_SCAN_BATCH_SIZE = 50_000
_WORK_PROGRESS_INTERVAL_ROWS = 100_000
_logger = structlog.getLogger(__name__)


def read_openalex_mobility_articles(
    path: Path,
    candidate_author_ids: set[str],
    *,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Read candidate-authored Works and flatten author-specific institutions.

    Only Works containing one of the selected author IDs are retained. Their
    complete authorship lists are preserved so mobility can calculate
    publication-level coauthor counts before restricting rows to selected
    profiles.

    :param path: Configured OpenAlex Works Parquet file.
    :param candidate_author_ids: Stable OpenAlex author IDs from disambiguation.
    :param max_workers: Maximum concurrent Arrow-batch filtering workers.
    :return: Mobility-shaped article-author rows.
    :raises ValueError: If a candidate ID is invalid or ``max_workers`` is not positive.
    """
    invalid_ids = {
        identifier
        for identifier in candidate_author_ids
        if not _OPENALEX_AUTHOR_ID_PATTERN.fullmatch(identifier)
    }
    if invalid_ids:
        raise ValueError(
            "OpenAlex mobility requires canonical author IDs; invalid values include "
            + ", ".join(sorted(invalid_ids)[:5])
        )
    if max_workers < 1:
        raise ValueError("openalex_scan_workers must be a positive integer")

    columns = [EID, ARTICLE_COVER_DATE, AUTHOR_IDS, ARTICLE_AFFILIATION_ID]
    if not candidate_author_ids:
        return pd.DataFrame(columns=columns)

    parquet_file = pq.ParquetFile(path)
    total_works = parquet_file.metadata.num_rows
    _logger.info(
        "Scanning OpenAlex Works for mobility",
        path=str(path),
        total_works=total_works,
        candidate_author_count=len(candidate_author_ids),
        batch_size=_WORK_SCAN_BATCH_SIZE,
        max_workers=max_workers,
        rss_mb=_current_rss_megabytes(),
    )
    records: list[dict[str, Any]] = []
    works_scanned = 0
    next_progress_row = _WORK_PROGRESS_INTERVAL_ROWS
    candidate_ids = sorted(candidate_author_ids)
    scanner = parquet_file.iter_batches(
        batch_size=_WORK_SCAN_BATCH_SIZE,
        columns=["id", "publication_date", "authorships"],
        use_threads=True,
    )
    pending: deque[Future[list[dict[str, Any]]]] = deque()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch in scanner:
            works_scanned += batch.num_rows
            pending.append(executor.submit(_filter_adapt_work_batch, batch, candidate_ids))
            if len(pending) >= max_workers:
                records.extend(pending.popleft().result())
            if works_scanned >= next_progress_row:
                _logger.info(
                    "OpenAlex Works scan progress for mobility",
                    works_scanned=works_scanned,
                    total_works=total_works,
                    percent_complete=(
                        round(100 * works_scanned / total_works, 1) if total_works else 100.0
                    ),
                    retained_works=len(records),
                    rss_mb=_current_rss_megabytes(),
                )
                next_progress_row += _WORK_PROGRESS_INTERVAL_ROWS
        while pending:
            records.extend(pending.popleft().result())

    _logger.info(
        "Finished OpenAlex Works scan for mobility",
        total_works=total_works,
        works_scanned=works_scanned,
        retained_works=len(records),
        rss_mb=_current_rss_megabytes(),
    )
    return pd.DataFrame.from_records(records, columns=columns)


def read_openalex_mobility_affiliations(path: Path) -> pd.DataFrame:
    """Read the OpenAlex institution dimension without inventing parent IDs.

    OpenAlex institution IDs are retained as strings. The observed dimension
    supplies institution metadata but has no parent-ID columns; mobility
    therefore uses each institution ID directly.

    :param path: Configured OpenAlex institution Parquet file.
    :return: Institution metadata in the mobility affiliation schema.
    """
    affiliations = pd.read_parquet(path)
    affiliations[AFFILIATION_NAME] = affiliations["name"]
    affiliations[AFFILIATION_PREFERRED_NAME] = affiliations[AFFILIATION_NAME]
    return affiliations


def attach_openalex_current_affiliations(
    authors: pd.DataFrame,
    candidate_authors_path: Path,
) -> pd.DataFrame:
    """Attach current institution IDs from the observed OpenAlex pair input.

    :param authors: Final author-selection rows.
    :param candidate_authors_path: Configured OpenAlex author-affiliation pairs.
    :return: Copy with current institution IDs for each selected author profile.
    :raises ValueError: If the source input lacks the observed required columns.
    """
    if candidate_authors_path.suffix.casefold() != ".parquet":
        raise ValueError("OpenAlex mobility candidate-author input must be Parquet")
    columns = {"author_id", "institution_id", "current_affiliation"}
    source_columns = set(pq.ParquetFile(candidate_authors_path).schema_arrow.names)
    missing_columns = columns - source_columns
    if missing_columns:
        raise ValueError(
            "OpenAlex mobility candidate-author input is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    pairs = pd.read_parquet(candidate_authors_path, columns=sorted(columns))
    pairs = pairs.loc[
        pairs["current_affiliation"].eq(True), ["author_id", "institution_id"]
    ].dropna()
    pairs = pairs.drop_duplicates()
    current_by_author = pairs.groupby("author_id", sort=False)["institution_id"].agg(list).to_dict()

    result = authors.copy()
    result[CURRENT_AFFILIATION_IDS] = result[SELECTED_AUTHOR_IDS].map(
        lambda value: _current_ids_for_selected_authors(value, current_by_author)
    )
    return result


def selected_openalex_author_ids(authors: pd.DataFrame) -> set[str]:
    """Collect stable OpenAlex author IDs from final disambiguation selections.

    :param authors: Final-author-selection table.
    :return: Unique selected OpenAlex author URLs.
    """
    identifiers: set[str] = set()
    if SELECTED_AUTHOR_IDS in authors.columns:
        for value in authors[SELECTED_AUTHOR_IDS].dropna():
            identifiers.update(_openalex_author_ids_from_value(value))
    return identifiers


def _openalex_author_ids_from_value(value: Any) -> set[str]:
    """Parse a selected-ID CSV cell and retain canonical author URLs only."""
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        text = str(value).strip()
        if not text or text.casefold() in {"nan", "none", "null"}:
            return set()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return set()
        items = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
    return {
        str(item)
        for item in items
        if isinstance(item, str) and _OPENALEX_AUTHOR_ID_PATTERN.fullmatch(item)
    }


def _current_ids_for_selected_authors(
    value: Any,
    current_by_author: dict[str, list[str]],
) -> list[str]:
    """Flatten current institution IDs for one serialized author selection."""
    selected_ids = _openalex_author_ids_from_value(value)
    return sorted(
        {
            institution_id
            for author_id in selected_ids
            for institution_id in current_by_author.get(author_id, [])
        }
    )


def _adapt_work(work: dict[str, Any]) -> dict[str, Any]:
    """Convert one nested OpenAlex Work to Mobility's aligned-list input."""
    author_ids: list[str] = []
    author_affiliations: list[str] = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        author_id = author.get("id")
        if not isinstance(author_id, str) or not _OPENALEX_AUTHOR_ID_PATTERN.fullmatch(author_id):
            continue
        institution_ids = {
            institution_id
            for affiliation in authorship.get("affiliations") or []
            for institution_id in affiliation.get("institution_ids") or []
            if isinstance(institution_id, str)
        }
        author_ids.append(author_id)
        author_affiliations.append("-".join(sorted(institution_ids)))

    return {
        EID: work.get("id"),
        ARTICLE_COVER_DATE: work.get("publication_date"),
        AUTHOR_IDS: ";".join(author_ids),
        ARTICLE_AFFILIATION_ID: ";".join(author_affiliations),
    }


def _filter_adapt_work_batch(
    input_batch: pa.RecordBatch,
    candidate_author_ids: list[str],
) -> list[dict[str, Any]]:
    """Filter and adapt one bounded batch using a worker-local DuckDB connection.

    :param input_batch: One batch from the configured OpenAlex Works Parquet file.
    :param candidate_author_ids: Sorted stable author IDs to retain.
    :return: Mobility rows for candidate-authored Works in the batch.
    """
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET threads = 1")
        connection.register("mobility_work_batch", pa.Table.from_batches([input_batch]))
        matching_batch = connection.execute(
            """
            SELECT id, publication_date, authorships
            FROM mobility_work_batch
            WHERE list_has_any(
                list_transform(authorships, authorship -> authorship.author.id),
                ?
            )
            """,
            [candidate_author_ids],
        ).to_arrow_table()
    finally:
        connection.close()
    return [_adapt_work(work) for work in matching_batch.to_pylist()]


def _current_rss_megabytes() -> float | None:
    """Read current resident memory from Linux procfs without extra dependencies."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None
