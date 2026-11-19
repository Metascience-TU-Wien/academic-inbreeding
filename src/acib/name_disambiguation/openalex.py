# Copyright © 2026 TU Wien

"""Bounded readers and evidence adaptation for the observed OpenAlex Works export."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb  # type: ignore[import-not-found]
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from acib.constants import EID
from acib.name_disambiguation.constants import (
    ARTICLE_AUTHOR_NAMES,
    ARTICLE_GRANT_NUMBERS,
    ARTICLE_JOURNAL,
    OPENALEX_AUTHOR_AFFILIATION_EVIDENCE,
    OPENALEX_AUTHOR_ORCID_EVIDENCE,
    OPENALEX_WORK_FIELD_IDS,
)
from acib.name_disambiguation.diagnostics import current_rss_megabytes

_WORK_COLUMNS = (
    "id",
    "authorships",
    "primary_location",
    "locations",
    "awards",
    "topics",
    "primary_topic",
)
_AFFILIATION_COLUMNS = ("affiliation_id", "city", "country_code", "country")
_WORK_ID_PATTERN = re.compile(r"https://openalex\.org/W(0|[1-9]\d*)\Z")
_ORCID_PATTERN = re.compile(r"\d{4}-\d{4}-\d{4}-[\dX]{4}\Z", re.IGNORECASE)
_WORK_PROGRESS_INTERVAL_ROWS = 100_000
_logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class OpenAlexCitationIndex:
    """Compact directional citation sets using numeric OpenAlex Work IDs.

    :param cited_by_work: Numeric cited Work IDs keyed by candidate citing Work ID.
    :param citing_by_work: Numeric citing Work IDs keyed by candidate cited Work ID.
    """

    cited_by_work: dict[int, set[int]]
    citing_by_work: dict[int, set[int]]

    @staticmethod
    def encode_work_id(work_id: str) -> int:
        """Convert a canonical OpenAlex Work URL to its numeric identifier.

        :param work_id: Canonical stable OpenAlex Work ID.
        :return: Numeric Work identifier, reversible to its canonical URL.
        :raises ValueError: If the ID is not a canonical OpenAlex Work URL.
        """
        match = _WORK_ID_PATTERN.fullmatch(work_id)
        if match is None:
            raise ValueError(f"Invalid OpenAlex Work ID: {work_id!r}")
        return int(match.group(1))

    def release(self) -> None:
        """Drop the source maps after the scoring index has consumed them."""
        self.cited_by_work.clear()
        self.citing_by_work.clear()


def load_openalex_work_evidence(
    works_path: Path,
    *,
    candidate_author_ids: set[str],
    affiliations_path: Path,
    batch_size: int = 50_000,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Vector-filter Parquet batches in DuckDB and adapt matching Works.

    Only the fields used by current publication evidence are projected. Topic and
    field identifiers are intentionally not adapted to the Scopus ASJC contract.

    :param works_path: Configured OpenAlex Works Parquet file.
    :param candidate_author_ids: Stable OpenAlex author IDs in the matcher artifact.
    :param affiliations_path: Configured OpenAlex institution dimension.
    :param batch_size: Maximum input Works records held in an Arrow batch.
    :param max_workers: Maximum concurrent Works-batch filtering workers.
    :return: Canonical publication-evidence rows for candidate authors' Works.
    """
    if not candidate_author_ids:
        return _empty_work_frame()
    if max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    cities = _read_affiliation_cities(affiliations_path)
    total_rows = pq.ParquetFile(works_path).metadata.num_rows
    _logger.info(
        "Scanning OpenAlex Works",
        path=str(works_path),
        total_works=total_rows,
        candidate_author_count=len(candidate_author_ids),
        batch_size=batch_size,
        max_workers=max_workers,
        rss_mb=current_rss_megabytes(),
    )
    rows: list[dict[str, Any]] = []
    works_scanned = 0
    next_progress_row = _WORK_PROGRESS_INTERVAL_ROWS
    candidate_ids = sorted(candidate_author_ids)
    parquet_file = pq.ParquetFile(works_path)
    scanner = parquet_file.iter_batches(
        batch_size=batch_size,
        columns=list(_WORK_COLUMNS),
        use_threads=True,
    )
    pending: deque[Future[list[dict[str, Any]]]] = deque()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for input_batch in scanner:
            works_scanned += input_batch.num_rows
            pending.append(
                executor.submit(
                    _filter_adapt_openalex_batch,
                    input_batch,
                    candidate_ids,
                    candidate_author_ids,
                    cities,
                )
            )
            if len(pending) >= max_workers:
                rows.extend(pending.popleft().result())
            if works_scanned >= next_progress_row:
                _logger.info(
                    "OpenAlex Works scan progress",
                    works_scanned=works_scanned,
                    total_works=total_rows,
                    percent_complete=(
                        round(100 * works_scanned / total_rows, 1) if total_rows else 100.0
                    ),
                    candidate_works_retained=len(rows),
                    rss_mb=current_rss_megabytes(),
                )
                next_progress_row += _WORK_PROGRESS_INTERVAL_ROWS
        while pending:
            rows.extend(pending.popleft().result())
    _logger.info(
        "Finished OpenAlex Works scan",
        total_works=total_rows,
        works_scanned=works_scanned,
        candidate_works_retained=len(rows),
        rss_mb=current_rss_megabytes(),
    )
    return pd.DataFrame(rows, columns=_WORK_EVIDENCE_COLUMNS)


def _filter_adapt_openalex_batch(
    input_batch: pa.RecordBatch,
    candidate_ids: list[str],
    candidate_author_ids: set[str],
    cities: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Filter and adapt one Arrow batch in a worker-local DuckDB connection.

    :param input_batch: Raw Works rows from the configured Parquet file.
    :param candidate_ids: Sorted stable OpenAlex author IDs for vector filtering.
    :param candidate_author_ids: Candidate IDs used to retain matching authors.
    :param cities: Institution locations keyed by stable institution ID.
    :return: Evidence rows for candidate-authored Works in this batch.
    """
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET threads = 1")
        connection.register("work_batch", pa.Table.from_batches([input_batch]))
        matching_batch = connection.execute(
            """
            SELECT
                id,
                authorships,
                primary_location,
                locations,
                awards,
                list_transform(topics, topic -> topic.field.id) AS openalex_work_field_ids,
                primary_topic.field.id AS primary_topic_field_id
            FROM work_batch
            WHERE list_has_any(
                list_transform(authorships, authorship -> authorship.author.id),
                ?
            )
            """,
            [candidate_ids],
        ).to_arrow_table()
    finally:
        connection.close()
    rows: list[dict[str, Any]] = []
    for work in matching_batch.to_pylist():
        authorships = [
            authorship
            for authorship in (work.get("authorships") or [])
            if isinstance(authorship, dict)
        ]
        author_ids = [
            author_id
            for authorship in authorships
            if (author_id := _author_id(authorship)) is not None
        ]
        rows.append(_adapt_work(work, authorships, author_ids, candidate_author_ids, cities))
    return rows


def load_openalex_citation_edges(
    citation_path: Path,
    *,
    candidate_work_ids: set[str],
    batch_size: int = 250_000,
) -> OpenAlexCitationIndex:
    """Semi-join the configured citation Parquet against candidate Works in DuckDB.

    The complete citation table is never materialized in pandas. Only edges with
    at least one endpoint in candidate-authored Works are returned.

    :param citation_path: Parquet edge table with ``citing_id`` and ``cited_id``.
    :param candidate_work_ids: Stable Work IDs from the filtered Works scan.
    :param batch_size: Number of matching edges fetched from DuckDB per batch.
    :return: Compact forward and reverse citation sets for candidate Works.
    """
    if not candidate_work_ids:
        return OpenAlexCitationIndex({}, {})
    total_edges = pq.ParquetFile(citation_path).metadata.num_rows
    _logger.info(
        "Filtering OpenAlex citation edges",
        path=str(citation_path),
        total_edges=total_edges,
        candidate_work_count=len(candidate_work_ids),
        rss_mb=current_rss_megabytes(),
    )
    connection = duckdb.connect(":memory:")
    citing_to_cited: dict[int, set[int]] = defaultdict(set)
    cited_to_citing: dict[int, set[int]] = defaultdict(set)
    try:
        connection.register(
            "candidate_work_ids",
            pd.DataFrame({"work_id": list(candidate_work_ids)}, dtype="string"),
        )
        escaped_path = str(citation_path).replace("'", "''")
        connection.execute(
            f"""
            SELECT citing_id, cited_id
            FROM read_parquet('{escaped_path}')
            WHERE citing_id IN (SELECT work_id FROM candidate_work_ids)
               OR cited_id IN (SELECT work_id FROM candidate_work_ids)
            """
        )
        reader = connection.to_arrow_reader(batch_size)
        fetched_edges = 0
        chunk_count = 0
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            for edge in batch.to_pylist():
                citing_id = _string(edge.get("citing_id"))
                cited_id = _string(edge.get("cited_id"))
                if not citing_id or not cited_id:
                    continue
                numeric_citing_id = OpenAlexCitationIndex.encode_work_id(citing_id)
                numeric_cited_id = OpenAlexCitationIndex.encode_work_id(cited_id)
                if citing_id in candidate_work_ids:
                    citing_to_cited[numeric_citing_id].add(numeric_cited_id)
                if cited_id in candidate_work_ids:
                    cited_to_citing[numeric_cited_id].add(numeric_citing_id)
            fetched_edges += batch.num_rows
            chunk_count += 1
            if chunk_count % 10 == 0:
                _logger.info(
                    "OpenAlex citation-edge index progress",
                    chunks_processed=chunk_count,
                    edges_processed=fetched_edges,
                    total_edges=total_edges,
                    candidate_citing_works=len(citing_to_cited),
                    candidate_cited_works=len(cited_to_citing),
                    rss_mb=current_rss_megabytes(),
                )
        _logger.info(
            "Finished OpenAlex citation-edge indexing",
            candidate_edges=fetched_edges,
            candidate_citing_works=len(citing_to_cited),
            candidate_cited_works=len(cited_to_citing),
            total_edges=total_edges,
            rss_mb=current_rss_megabytes(),
        )
        return OpenAlexCitationIndex(dict(citing_to_cited), dict(cited_to_citing))
    finally:
        connection.close()


def openalex_author_orcid_map(article_authors: pd.DataFrame) -> dict[str, str]:
    """Return unambiguous ORCID values observed for OpenAlex author IDs.

    :param article_authors: Adapted OpenAlex Work evidence rows.
    :return: Author IDs mapped to one canonical ORCID value.
    """
    evidence_by_author: dict[str, set[str]] = defaultdict(set)
    for evidence_rows in article_authors.get(
        OPENALEX_AUTHOR_ORCID_EVIDENCE, pd.Series(dtype="object")
    ):
        for evidence in evidence_rows or []:
            if not isinstance(evidence, dict):
                continue
            author_id = _string(evidence.get("author_id"))
            orcid = _normalize_orcid(evidence.get("orcid"))
            if author_id and orcid:
                evidence_by_author[author_id].add(orcid)
    return {
        author_id: next(iter(orcids))
        for author_id, orcids in evidence_by_author.items()
        if len(orcids) == 1
    }


_WORK_EVIDENCE_COLUMNS = (
    EID,
    "author_ids",
    ARTICLE_AUTHOR_NAMES,
    ARTICLE_JOURNAL,
    ARTICLE_GRANT_NUMBERS,
    OPENALEX_AUTHOR_AFFILIATION_EVIDENCE,
    OPENALEX_AUTHOR_ORCID_EVIDENCE,
    OPENALEX_WORK_FIELD_IDS,
)


def _empty_work_frame() -> pd.DataFrame:
    """Return an empty frame with the OpenAlex adapter's canonical columns."""
    return pd.DataFrame(columns=_WORK_EVIDENCE_COLUMNS)


def _read_affiliation_cities(path: Path) -> dict[str, dict[str, str]]:
    """Read the small institution dimension keyed by exact stable institution ID."""
    dimension = pd.read_parquet(path, columns=list(_AFFILIATION_COLUMNS))
    result: dict[str, dict[str, str]] = {}
    for row in dimension.to_dict(orient="records"):
        institution_id = _string(row.get("affiliation_id"))
        if institution_id:
            result[institution_id] = {
                "city": _string(row.get("city")),
                "country": _string(row.get("country")),
                "country_code": _string(row.get("country_code")),
            }
    return result


def _adapt_work(
    work: dict[str, Any],
    authorships: list[dict[str, Any]],
    author_ids: list[str],
    candidate_author_ids: set[str],
    cities: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Map one observed OpenAlex Work to the existing publication evidence shape."""
    names: set[str] = set()
    author_affiliation_evidence: list[dict[str, Any]] = []
    author_orcid_evidence: list[dict[str, str]] = []
    for authorship in authorships:
        author_id = _author_id(authorship)
        author = authorship.get("author") or {}
        if not isinstance(author, dict):
            author = {}
        orcids = {
            value
            for candidate in (
                author.get("orcid"),
                authorship.get("raw_orcid"),
                *(author.get("observed_orcids") or []),
            )
            if (value := _string(candidate))
        }
        raw_name = _string(authorship.get("raw_author_name"))
        if raw_name:
            names.add(raw_name)
        author_cities: set[str] = set()
        author_countries: set[str] = set()
        for country in authorship.get("countries") or []:
            if value := _string(country):
                author_countries.add(value)
        institutions = [
            institution
            for institution in (authorship.get("institutions") or [])
            if isinstance(institution, dict)
        ]
        for institution in institutions:
            code = _string(institution.get("country_code"))
            if code:
                author_countries.add(code)
        affiliations = [
            affiliation
            for affiliation in (authorship.get("affiliations") or [])
            if isinstance(affiliation, dict)
        ]
        institution_ids = {
            _string(identifier)
            for affiliation in affiliations
            for identifier in (affiliation.get("institution_ids") or [])
        }
        for institution_id in institution_ids:
            if values := cities.get(institution_id):
                if values["city"]:
                    author_cities.add(values["city"])
                if values["country"]:
                    author_countries.add(values["country"])
                if values["country_code"]:
                    author_countries.add(values["country_code"])
        if author_id and author_id in candidate_author_ids:
            author_affiliation_evidence.append(
                {
                    "author_id": author_id,
                    "cities": sorted(author_cities),
                    "countries": sorted(author_countries),
                }
            )
            author_orcid_evidence.extend(
                {"author_id": author_id, "orcid": orcid} for orcid in sorted(orcids)
            )

    journals: set[str] = set()
    locations = [
        (work.get("primary_location"), True),
        *((location, False) for location in (work.get("locations") or [])),
    ]
    for location, is_primary in locations:
        if not isinstance(location, dict):
            continue
        source = location.get("source") or {}
        if not isinstance(source, dict):
            continue
        source_type = _string(source.get("type")).casefold()
        if source_type != "journal":
            continue
        for value in (
            source.get("id"),
            source.get("issn_l"),
            source.get("display_name"),
            *(source.get("issn") or []),
            location.get("raw_source_name") if is_primary else None,
        ):
            if normalized := _string(value):
                journals.add(normalized)

    awards: set[str] = set()
    for award in work.get("awards") or []:
        if not isinstance(award, dict):
            continue
        # Award-specific identifiers are evidence. Funder identifiers are not
        # included because they identify an organization rather than a grant.
        for key in ("funder_award_id", "id"):
            if value := _string(award.get(key)):
                awards.add(value)

    field_ids = {
        field_id
        for value in (work.get(OPENALEX_WORK_FIELD_IDS) or [])
        if (field_id := _string(value))
    }
    if not field_ids:
        primary_field_id = _string(work.get("primary_topic_field_id"))
        if primary_field_id:
            field_ids.add(primary_field_id)

    return {
        EID: _string(work.get("id")),
        "author_ids": list(dict.fromkeys(author_ids)),
        ARTICLE_AUTHOR_NAMES: sorted(names),
        ARTICLE_JOURNAL: sorted(journals),
        ARTICLE_GRANT_NUMBERS: sorted(awards),
        OPENALEX_AUTHOR_AFFILIATION_EVIDENCE: author_affiliation_evidence,
        OPENALEX_AUTHOR_ORCID_EVIDENCE: author_orcid_evidence,
        OPENALEX_WORK_FIELD_IDS: sorted(field_ids),
    }


def _author_id(authorship: dict[str, Any]) -> str | None:
    """Return the stable OpenAlex author ID from one authorship struct."""
    author = authorship.get("author") or {}
    identifier = _string(author.get("id")) if isinstance(author, dict) else ""
    return identifier or None


def _string(value: object) -> str:
    """Return a trimmed string without converting missing values to text."""
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _normalize_orcid(value: object) -> str:
    """Canonicalize supported ORCID URL/text spellings or return empty text."""
    candidate = _string(value)
    for prefix in ("https://orcid.org/", "http://orcid.org/"):
        candidate = candidate.removeprefix(prefix)
    if _ORCID_PATTERN.fullmatch(candidate):
        return candidate.upper()
    return ""
