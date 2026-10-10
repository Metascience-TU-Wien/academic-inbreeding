# Copyright © 2026 TU Wien

"""CSV input, matching, and output workflow for the name-matching stage."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from acib.constants import (
    AFFILIATION_ID,
    AFFILIATION_ID_CURRENT,
    AFFILIATION_ID_PARENT,
    AFFILIATION_IDS_CURRENT,
    AFFILIATION_NAME,
    AUTHOR_ID,
    EID,
    RESEARCHER_ID,
    UNIVERSITY_NAME,
)
from acib.name_matching.config import NameMatchingConfig
from acib.name_matching.constants import (
    CURRENT_AFFILIATION,
    MATCH_PRIORITY,
    NAME,
    PARENT_AFFILIATION_COLUMNS,
    PERSON_FULLNAME,
    RESEARCHER_GIVEN_NAME,
    RESEARCHER_NAME,
    RESEARCHER_SURNAME,
    SCOPUS_ID_MAPPING,
)
from acib.name_matching.matching import ResearcherMatchResult, match_researchers

_logger = structlog.getLogger(__name__)


def normalize_existing_scopus_author_id(value: object) -> object:
    """Normalize a pre-existing Scopus identifier as the source runner did."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return pd.NA
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def split_researchers_with_existing_scopus_ids(
    researchers: pd.DataFrame,
    column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate rows with known Scopus IDs from names to be matched.

    :param researchers: Input researcher rows.
    :param column: Optional column containing existing author IDs.
    :return: Rows to match and pre-resolved rows, respectively.
    :raises KeyError: If a configured ID column does not exist.
    """
    if column is None:
        return researchers, researchers.iloc[0:0].copy()
    if column not in researchers:
        raise KeyError(f"Configured existing Scopus author ID column is missing: {column!r}")
    normalized = researchers[column].map(normalize_existing_scopus_author_id)
    has_id = normalized.notna()
    to_match = researchers.loc[~has_id].copy()
    resolved = researchers.loc[has_id].copy()
    resolved[AUTHOR_ID] = normalized.loc[has_id].to_numpy()
    if column != AUTHOR_ID:
        resolved = resolved.drop(columns=[column])
    resolved[MATCH_PRIORITY] = 0
    return to_match, resolved


def prepare_scopus_author_affiliations(authors: pd.DataFrame) -> pd.DataFrame:
    """Prefer mapped parent affiliation IDs and compute current-affiliation fields."""
    result = authors.copy()
    if AFFILIATION_ID not in result:
        raise KeyError("Author input must contain an 'affiliation_id' column")
    if AFFILIATION_ID_PARENT not in result:
        result[AFFILIATION_ID_PARENT] = pd.NA
    for column in PARENT_AFFILIATION_COLUMNS:
        if column not in result:
            result[column] = pd.NA
    parent_ids = result[PARENT_AFFILIATION_COLUMNS[0]]
    for column in PARENT_AFFILIATION_COLUMNS[1:]:
        parent_ids = parent_ids.combine_first(result[column])
    result[AFFILIATION_ID_PARENT] = parent_ids
    result[AFFILIATION_ID] = np.where(parent_ids.notna(), parent_ids, result[AFFILIATION_ID])
    result[AFFILIATION_ID] = result[AFFILIATION_ID].astype("int64")
    if CURRENT_AFFILIATION in result:
        current = result[CURRENT_AFFILIATION].fillna(False).astype(str).str.strip().str.lower()
        current_mask = current.isin({"true", "1", "yes", "y"})
    else:
        current_mask = pd.Series(False, index=result.index)
    if AUTHOR_ID in result:
        current_affiliations = result.loc[
            current_mask, [AUTHOR_ID, AFFILIATION_ID]
        ].drop_duplicates()
        current_affiliations = current_affiliations.sort_values(
            [AUTHOR_ID, AFFILIATION_ID], kind="mergesort"
        )
        grouped = current_affiliations.groupby(AUTHOR_ID)[AFFILIATION_ID].agg(list)
        result[AFFILIATION_IDS_CURRENT] = (
            result[AUTHOR_ID]
            .map(grouped)
            .map(lambda value: value if isinstance(value, list) else [])
        )
        result[AFFILIATION_ID_CURRENT] = result[AFFILIATION_IDS_CURRENT].map(
            lambda values: values[0] if values else -9999
        )
    return result


def sort_resolved_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Sort output stably using the source runner's available output columns."""
    columns = [
        column
        for column in (
            RESEARCHER_ID,
            MATCH_PRIORITY,
            AUTHOR_ID,
            EID,
            AFFILIATION_ID,
            AFFILIATION_ID_PARENT,
            AFFILIATION_NAME,
        )
        if column in matches
    ]
    if not columns:
        return matches.reset_index(drop=True)
    return matches.sort_values(columns, kind="mergesort", na_position="last").reset_index(drop=True)


def run_name_matching(config: NameMatchingConfig) -> pd.DataFrame:
    """Load configured CSVs, execute matching, and write resolved records.

    :param config: Settings loaded from the ``[name_matching]`` TOML table.
    :return: The rows written to the resolved-matches output.
    :raises FileNotFoundError: If either configured input does not exist.
    :raises KeyError: If required source columns are absent.
    :raises ValueError: If researcher IDs are missing or not unique, or metadata restoration
     changes row count.
    """
    started_at = time.perf_counter()
    _logger.info(
        "Loading name matching inputs",
        researcher_file_path=str(config.researcher_file_path),
        candidate_authors_file=str(config.candidate_authors_file),
    )
    config.check_paths_exist()
    output_path: Path = config.resolved_matches_output_file_path.resolve()
    source_repository = (Path(__file__).resolve().parents[4] / "name-matching-playground").resolve()
    if output_path.is_relative_to(source_repository):
        raise ValueError(
            "Refusing to write name-matching output inside the read-only source repository: "
            f"{source_repository}"
        )
    researcher_metadata = pd.read_csv(config.researcher_file_path)
    if RESEARCHER_ID not in researcher_metadata:
        raise KeyError(f"Researcher input must contain a {RESEARCHER_ID!r} column")
    if researcher_metadata[RESEARCHER_ID].isna().any():
        raise ValueError(f"Researcher input contains missing {RESEARCHER_ID!r} values")
    duplicated_ids = researcher_metadata[RESEARCHER_ID].duplicated(keep=False)
    if duplicated_ids.any():
        duplicate_values = researcher_metadata.loc[duplicated_ids, RESEARCHER_ID].drop_duplicates()
        raise ValueError(
            f"Researcher {RESEARCHER_ID!r} values must be unique; duplicates include "
            f"{duplicate_values.head(10).tolist()!r}"
        )

    matching_columns = (
        RESEARCHER_ID,
        UNIVERSITY_NAME,
        RESEARCHER_NAME,
        NAME,
        PERSON_FULLNAME,
        RESEARCHER_GIVEN_NAME,
        RESEARCHER_SURNAME,
        config.existing_scopus_author_id_column,
    )
    researchers = researcher_metadata.loc[
        :,
        [
            column
            for column in matching_columns
            if column is not None and column in researcher_metadata
        ],
    ].copy()
    authors = pd.read_csv(config.candidate_authors_file)
    _logger.info(
        "Loaded name matching inputs",
        researcher_rows=len(researcher_metadata),
        candidate_author_rows=len(authors),
    )
    pre_resolved = researchers.iloc[0:0].copy()
    researchers, pre_resolved = split_researchers_with_existing_scopus_ids(
        researchers, config.existing_scopus_author_id_column
    )
    if config.existing_scopus_author_id_column is not None:
        _logger.info(
            "Skipping researchers with existing Scopus author IDs",
            skipped_rows=len(pre_resolved),
            existing_scopus_author_id_column=config.existing_scopus_author_id_column,
        )
    authors = prepare_scopus_author_affiliations(authors)
    org_mapping: dict[str, list[int | str]] = {
        organization: list(ids) for organization, ids in SCOPUS_ID_MAPPING.items()
    }
    pre_resolved = pre_resolved.loc[
        :,
        [column for column in (RESEARCHER_ID, AUTHOR_ID, MATCH_PRIORITY) if column in pre_resolved],
    ].copy()
    matches = researchers.iloc[0:0].copy()
    if not researchers.empty:
        _logger.info(
            "Preparing researcher and candidate author names",
            researcher_rows=len(researchers),
            candidate_author_rows=len(authors),
        )
        result: ResearcherMatchResult = match_researchers(
            researchers,
            authors,
            org_mapping=org_mapping,
            enable_given_name_aliases=config.enable_given_name_aliases,
            enable_organisation_filter=config.enable_organisation_filter,
            max_workers=config.max_workers,
        )
        matches = result.matched
    resolved = pd.concat([pre_resolved, matches], ignore_index=True, sort=False)
    resolved_row_count = len(resolved)
    output = resolved.merge(
        researcher_metadata,
        on=RESEARCHER_ID,
        how="left",
        sort=False,
        validate="many_to_one",
        suffixes=("", "_researcher_metadata"),
    )
    if len(output) != resolved_row_count:
        raise RuntimeError(
            "Restoring researcher metadata changed the number of resolved match rows"
        )
    output = sort_resolved_matches(output)
    _logger.info(
        "Writing resolved name matches",
        output_rows=len(output),
        output_path=str(output_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding=config.encoding)
    _logger.info(
        "Finished name matching run",
        total_runtime_seconds=round(time.perf_counter() - started_at, 3),
    )
    return output
