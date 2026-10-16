# Copyright © 2026 TU Wien

"""CSV input, matching, and output workflow for the name-matching stage."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from acib.common.subjects import (
    ASJC_PARENT_LABELS,
    map_organization_to_asjc_codes,
    parent_code,
)
from acib.constants import (
    AFFILIATION_ID,
    AFFILIATION_ID_CURRENT,
    AFFILIATION_ID_PARENT,
    AFFILIATION_ID_PARENT_CHATGPT,
    AFFILIATION_ID_PARENT_SPLINK_CHATGPT,
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
    IS_ALTERNATIVE,
    MATCH_PRIORITY,
    MATCHING_OUTPUT_COLUMNS,
    NAME,
    PARENT_AFFILIATION_COLUMNS,
    PERSON_FULLNAME,
    RESEARCHER_GIVEN_NAME,
    RESEARCHER_NAME,
    RESEARCHER_SURNAME,
    SCOPUS_ID_MAPPING,
)
from acib.name_matching.matching import ResearcherMatchResult, match_researchers
from acib.schemas import (
    MatchingOutputSchema,
    ResearcherInputSchema,
    ScopusAuthorInputSchema,
)
from acib.schemas.affiliations import parse_nullable_affiliation_ids

_logger = structlog.getLogger(__name__)

_SOURCE_OUTPUT_COLUMN_ORDER = (
    "sc_id",
    "person_title",
    "person_fullname",
    "person_role",
    "person_link",
    "person_functions",
    "institute_link",
    "organization",
    "university_name",
    "person_grade",
    "person_email",
    "person_phone",
    "person_location_link",
    "orcid",
    "scopus",
    "researchgate",
    "organization_names",
    "organization_names_relevant",
    "inferred_asjc_evidence",
    "inferred_asjc_codes",
    "inferred_asjc_parent_codes",
    "inferred_asjc_labels",
    "inferred_asjc_parent_labels",
    "has_inferred_asjc",
    "inferred_asjc_max_score",
    "inferred_asjc_confidence",
    "organization_names_json",
    "organization_names_relevant_json",
    "inferred_asjc_evidence_json",
    "inferred_asjc_codes_json",
    "inferred_asjc_parent_codes_json",
    "inferred_asjc_labels_json",
    "inferred_asjc_parent_labels_json",
    "cv_link",
    "is_alternative",
    "person_surname",
    "person_name",
    "initials",
    "author_id",
    "priority",
    "researcher_name",
    "eid",
    "name",
    "indexed_name",
    "given_name",
    "surname",
    "publication_range",
    "name_variants",
    "subject_areas",
    "historical_identifier",
    "affiliation_id",
    "affiliation_id_parent",
    "affiliation_name",
    "affiliation_id_parent_chatgpt",
    "affiliation_id_parent_splink_chatgpt",
    "affiliation_ids_current",
    "author_name",
)

INFERRED_ASJC_CODES_COLUMN = "inferred_asjc_codes"
INFERRED_ASJC_LABELS_COLUMN = "inferred_asjc_labels"
ORGANIZATION_ASJC_SOURCE_COLUMNS = ("organization", "institute_name", "faculty", "faculty_name")


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
            AFFILIATION_ID_PARENT_CHATGPT,
            AFFILIATION_ID_PARENT_SPLINK_CHATGPT,
            AFFILIATION_NAME,
        )
        if column in matches
    ]
    if not columns:
        return matches.reset_index(drop=True)
    return matches.sort_values(columns, kind="mergesort", na_position="last").reset_index(drop=True)


def add_inferred_asjc_columns(researchers: pd.DataFrame) -> pd.DataFrame:
    """Infer ASJC codes and labels from available organization metadata.

    Existing nonblank code values are retained, matching the original workflow.

    :param researchers: Researcher metadata rows.
    :return: A copy with inferred-code and label columns populated when possible.
    """
    source_columns = [
        column for column in ORGANIZATION_ASJC_SOURCE_COLUMNS if column in researchers
    ]
    if not source_columns:
        return researchers.copy()
    result = researchers.copy()
    if INFERRED_ASJC_CODES_COLUMN not in result:
        result[INFERRED_ASJC_CODES_COLUMN] = ""
    if INFERRED_ASJC_LABELS_COLUMN not in result:
        result[INFERRED_ASJC_LABELS_COLUMN] = ""
    for index, row in result.iterrows():
        if not _is_blank_value(row[INFERRED_ASJC_CODES_COLUMN]):
            continue
        codes = map_organization_to_asjc_codes([row[column] for column in source_columns])
        if not codes:
            continue
        result.at[index, INFERRED_ASJC_CODES_COLUMN] = ",".join(map(str, codes))
        if _is_blank_value(row[INFERRED_ASJC_LABELS_COLUMN]):
            labels = sorted(
                {
                    ASJC_PARENT_LABELS[parent_code(code)]
                    for code in codes
                    if parent_code(code) in ASJC_PARENT_LABELS
                }
            )
            result.at[index, INFERRED_ASJC_LABELS_COLUMN] = ";".join(labels)
    return result


def _is_blank_value(value: object) -> bool:
    """Return whether an inferred-ASJC cell contains no meaningful value."""
    if value is None or pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "[]", "<na>"}


def run_name_matching(config: NameMatchingConfig, *, overwrite: bool = False) -> pd.DataFrame:
    """Load configured CSVs, execute matching, and write resolved records.

    :param config: Settings loaded from the ``[name_matching]`` TOML table.
    :return: The rows written to the resolved-matches output.
    :raises FileNotFoundError: If either configured input does not exist.
    :raises KeyError: If required source columns are absent.
    :raises ValueError: If researcher IDs are missing or not unique, or metadata restoration
     changes row count.
    """
    started_at = time.perf_counter()
    if config.outputs.resolved_researchers.exists() and not overwrite:
        raise FileExistsError(
            f"Name-matching output already exists: {config.outputs.resolved_researchers}. "
            "Pass overwrite=True explicitly to replace it."
        )
    _logger.info(
        "Loading name matching inputs",
        researcher_file_path=str(config.researcher_file_path),
        candidate_authors_file=str(config.candidate_authors_file),
    )
    config.check_paths_exist()
    output_path: Path = config.outputs.resolved_researchers.resolve()
    source_repository = (Path(__file__).resolve().parents[4] / "name-matching-playground").resolve()
    if output_path.is_relative_to(source_repository):
        raise ValueError(
            "Refusing to write name-matching output inside the read-only source repository: "
            f"{source_repository}"
        )
    researcher_metadata = pd.read_csv(config.researcher_file_path)
    researcher_metadata = ResearcherInputSchema.validate(researcher_metadata, lazy=True)
    if config.enable_organisation_filter and UNIVERSITY_NAME not in researcher_metadata:
        raise ValueError(
            f"Researcher input requires {UNIVERSITY_NAME!r} when organization filtering is enabled"
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
    parent_id_columns = {column: "string" for column in PARENT_AFFILIATION_COLUMNS}
    authors = pd.read_csv(
        config.candidate_authors_file,
        dtype=parent_id_columns,
    )
    for column in PARENT_AFFILIATION_COLUMNS:
        if column in authors:
            authors[column] = parse_nullable_affiliation_ids(authors[column])
    authors = ScopusAuthorInputSchema.validate(authors, lazy=True)
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
    # Keep researcher input values authoritative. The resolved matching frame
    # contains projected researcher fields used during matching, some of which
    # overlap with the original input metadata; retain only match-owned output
    # fields before joining so pandas suffix precedence cannot replace metadata.
    matching_metadata_columns = {
        RESEARCHER_ID,
        RESEARCHER_NAME,
        *MATCHING_OUTPUT_COLUMNS,
    }
    resolved = resolved.loc[
        :, [column for column in resolved.columns if column in matching_metadata_columns]
    ]
    researcher_output_metadata = pd.read_csv(
        config.researcher_file_path,
        dtype="string",
        keep_default_na=False,
    )
    researcher_output_metadata[RESEARCHER_ID] = researcher_metadata[RESEARCHER_ID].array
    researcher_output_metadata = researcher_output_metadata.drop(
        columns=(
            [config.existing_scopus_author_id_column]
            if config.existing_scopus_author_id_column is not None
            else []
        ),
        errors="ignore",
    )
    resolved_row_count = len(resolved)
    output = researcher_output_metadata.merge(
        resolved,
        on=RESEARCHER_ID,
        how="inner",
        sort=False,
        validate="one_to_many",
    )
    if len(output) != resolved_row_count:
        raise RuntimeError(
            "Restoring researcher metadata changed the number of resolved match rows"
        )
    output = sort_resolved_matches(output)
    if IS_ALTERNATIVE in output:
        output[IS_ALTERNATIVE] = output[MATCH_PRIORITY].eq(0).map({True: False, False: pd.NA})
    if "initials" not in output:
        output["initials"] = pd.NA
    output.loc[output[MATCH_PRIORITY].eq(0), "initials"] = False
    for column in MATCHING_OUTPUT_COLUMNS:
        if column not in output:
            if output.empty and column in {RESEARCHER_ID, AUTHOR_ID, MATCH_PRIORITY}:
                output[column] = pd.Series(index=output.index, dtype="int64")
            else:
                output[column] = pd.Series(pd.NA, index=output.index, dtype="object")
    output = MatchingOutputSchema.validate(output, lazy=True)
    ordered_columns = [column for column in _SOURCE_OUTPUT_COLUMN_ORDER if column in output]
    ordered_columns.extend(column for column in output.columns if column not in ordered_columns)
    output = output.loc[:, ordered_columns]
    _logger.info(
        "Writing resolved name matches",
        output_rows=len(output),
        output_path=str(output_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_output = output.copy()
    for column in (AFFILIATION_ID, AFFILIATION_ID_PARENT):
        if column in serialized_output:
            serialized_output[column] = serialized_output[column].astype("Float64")
    serialized_output.to_csv(output_path, index=False, encoding=config.encoding)
    _logger.info(
        "Finished name matching run",
        total_runtime_seconds=round(time.perf_counter() - started_at, 3),
    )
    return output
