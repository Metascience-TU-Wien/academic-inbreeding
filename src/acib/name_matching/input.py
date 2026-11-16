# Copyright © 2026 TU Wien

"""Read and validate name-matching stage inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from acib.constants import RESEARCHER_ID
from acib.name_matching.config import NameMatchingConfig
from acib.name_matching.constants import (
    OPENALEX_AUTHOR_NAME_ALTERNATIVES,
    OPENALEX_OBSERVATIONS,
    OPENALEX_RAW_AUTHOR_NAMES,
    PARENT_AFFILIATION_COLUMNS,
)
from acib.name_matching.openalex import openalex_candidate_rows
from acib.name_matching.results import NameMatchingInputs
from acib.schemas import (
    OpenAlexAuthorAffiliationInputSchema,
    ResearcherInputSchema,
    ScopusAuthorInputSchema,
)
from acib.schemas.affiliations import parse_nullable_affiliation_ids


def prepare_name_matching_inputs(
    researchers: pd.DataFrame,
    candidate_authors: pd.DataFrame,
    *,
    researcher_output_metadata: pd.DataFrame | None = None,
    source: Literal["scopus", "openalex"] = "scopus",
) -> NameMatchingInputs:
    """Validate and normalize already-loaded matching inputs.

    :param researchers: Researcher records from a caller or configured dataset.
    :param candidate_authors: Candidate-author records for the selected source.
    :param researcher_output_metadata: Optional lexical source metadata for output.
    :param source: Candidate data source.
    :return: Validated in-memory inputs ready for matching.
    """
    validated_researchers = ResearcherInputSchema.validate(researchers, lazy=True)
    if source == "openalex":
        validated_raw_authors = OpenAlexAuthorAffiliationInputSchema.validate(
            candidate_authors.copy(), lazy=True
        )
        rows = validated_raw_authors.to_dict(orient="records")
        validated_authors = openalex_candidate_rows(rows)
    else:
        authors = candidate_authors.copy()
        for column in PARENT_AFFILIATION_COLUMNS:
            if column in authors:
                authors[column] = parse_nullable_affiliation_ids(authors[column])
        validated_authors = ScopusAuthorInputSchema.validate(authors, lazy=True)
    output_metadata = researcher_output_metadata
    if output_metadata is not None:
        output_metadata = output_metadata.copy()
        output_metadata[RESEARCHER_ID] = validated_researchers[RESEARCHER_ID].array
    return NameMatchingInputs(
        researchers=validated_researchers,
        candidate_authors=validated_authors,
        researcher_output_metadata=output_metadata,
    )


def _read_tabular_file(path: Path, **csv_options: Any) -> pd.DataFrame:
    """Read a supported CSV or Parquet table.

    :param path: Input table path; the suffix selects the reader.
    :param csv_options: Options passed to the CSV reader only.
    :return: Loaded table.
    :raises ValueError: If the file suffix is unsupported.
    """
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(path, **csv_options)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input file extension: {path.suffix}")


def _parse_openalex_csv_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Strictly decode structured OpenAlex fields read from CSV.

    :param frame: OpenAlex input table with CSV-serialized structured columns.
    :return: Copy with decoded JSON values and preserved null cells.
    :raises ValueError: If a structured field contains invalid JSON.
    """
    decoded = frame.copy()
    for column in (
        OPENALEX_AUTHOR_NAME_ALTERNATIVES,
        OPENALEX_RAW_AUTHOR_NAMES,
        OPENALEX_OBSERVATIONS,
    ):
        if column not in decoded:
            continue
        decoded[column] = decoded[column].map(
            lambda value: (
                None
                if value is None or value is pd.NA or pd.isna(value)
                else json.loads(value, strict=True)
            )
        )
    return decoded


def _read_openalex_candidates(path: Path) -> pd.DataFrame:
    """Read the observed derived OpenAlex author-affiliation table.

    :param path: CSV or Parquet path to author-affiliation records.
    :return: Raw author-affiliation rows with structured fields decoded.
    """
    if path.suffix.casefold() == ".csv":
        raw = _parse_openalex_csv_fields(
            _read_tabular_file(
                path,
                dtype={
                    "author_id": "string",
                    "institution_id": "string",
                    "author_name_alternatives": "string",
                    "raw_author_names": "string",
                    "observations": "string",
                },
            )
        )
    else:
        raw = _read_tabular_file(path)
        # Arrow's arrow.json extension yields strict JSON text through pandas'
        # Arrow-backed Parquet reader; normalize it before schema/adaptation.
        if OPENALEX_AUTHOR_NAME_ALTERNATIVES in raw:
            raw[OPENALEX_AUTHOR_NAME_ALTERNATIVES] = raw[OPENALEX_AUTHOR_NAME_ALTERNATIVES].map(
                lambda value: (
                    None
                    if value is None or value is pd.NA
                    else json.loads(value, strict=True) if isinstance(value, str) else value
                )
            )
        for column in (OPENALEX_RAW_AUTHOR_NAMES, OPENALEX_OBSERVATIONS):
            if column in raw:
                raw[column] = raw[column].map(
                    lambda value: value.tolist() if hasattr(value, "tolist") else value
                )
    return raw


def load_name_matching_inputs(config: NameMatchingConfig) -> NameMatchingInputs:
    """Load configured CSV or Parquet inputs without performing file output.

    :param config: Resolved matching input paths and options.
    :return: Loaded in-memory input tables ready for validation and matching.
    :raises FileNotFoundError: If a configured input is absent.
    """
    config.check_paths_exist()
    researchers = _read_tabular_file(config.researcher_file_path)
    if config.source == "openalex":
        candidate_authors = _read_openalex_candidates(config.candidate_authors_file)
    else:
        candidate_authors = _read_tabular_file(
            config.candidate_authors_file,
            dtype={column: "string" for column in PARENT_AFFILIATION_COLUMNS},
        )
    researcher_output_metadata = (
        _read_tabular_file(
            config.researcher_file_path,
            dtype="string",
            keep_default_na=False,
        )
        if config.researcher_file_path.suffix.casefold() == ".csv"
        else researchers.copy()
    )
    return NameMatchingInputs(
        researchers=researchers,
        candidate_authors=candidate_authors,
        researcher_output_metadata=researcher_output_metadata,
    )
