# Copyright © 2026 TU Wien

"""Loading and validation for name-disambiguation inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import structlog

from acib.constants import AUTHOR_ID, RESEARCHER_ID
from acib.name_disambiguation.config import NameDisambiguationConfig
from acib.name_disambiguation.constants import (
    ARTICLE_AUTHOR_EVIDENCE_INPUT_COLUMNS,
    ARTICLE_REFERENCE_INPUT_COLUMNS,
    OPENALEX_WORK_FIELD_IDS,
)
from acib.name_disambiguation.openalex import (
    load_openalex_citation_edges,
    load_openalex_work_evidence,
)
from acib.name_disambiguation.results import NameDisambiguationInputs
from acib.schemas.matching import MatchingOutputSchema, OpenAlexMatchingOutputSchema
from acib.schemas.publications import (
    ArticleAuthorEvidenceSchema,
    ArticleReferenceSchema,
)

_logger = structlog.get_logger(__name__)


def validate_matching_artifact(
    path: str | Path, *, source: Literal["scopus", "openalex"] = "scopus"
) -> pd.DataFrame:
    """Load and validate the producer-owned name-matching artifact.

    :param path: Canonical resolved-researchers CSV path.
    :param source: Bibliographic source used to select the output contract.
    :return: Validated researcher-to-author candidate rows.
    :raises FileNotFoundError: If the required artifact is missing.
    :raises ValueError: If the file cannot be read or violates its contract.
    """
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Name-matching upstream artifact is missing: {artifact}")
    try:
        frame = pd.read_csv(artifact, low_memory=False)
        schema = OpenAlexMatchingOutputSchema if source == "openalex" else MatchingOutputSchema
        return schema.validate(frame, lazy=True)
    except Exception as exc:
        raise ValueError(f"Invalid name-matching upstream artifact {artifact}: {exc}") from exc


def read_dataframe(path: Path) -> pd.DataFrame:
    """Read one configured CSV or Parquet evidence dataset.

    :param path: External evidence file path.
    :return: Dataset rows without changing source column names.
    :raises ValueError: If the file cannot be read or has an unsupported extension.
    """
    try:
        if path.suffix.casefold() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.casefold() == ".csv":
            return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, ImportError) as exc:
        raise ValueError(f"Could not read configured input {path}: {exc}") from exc
    raise ValueError(f"Unsupported input format for {path}; expected CSV or Parquet")


def prepare_name_disambiguation_inputs(
    matched_researchers: pd.DataFrame,
    article_authors: pd.DataFrame,
    article_references: pd.DataFrame,
    *,
    source: Literal["scopus", "openalex"] = "scopus",
) -> NameDisambiguationInputs:
    """Validate caller-provided DataFrames at the stage input boundary.

    :param matched_researchers: Rows produced by name matching.
    :param article_authors: Publication-author evidence rows.
    :param article_references: Citation edges.
    :param source: Bibliographic source used to validate author identifiers.
    :return: Validated inputs suitable for in-memory disambiguation.
    """
    return NameDisambiguationInputs(
        matched_researchers=(
            OpenAlexMatchingOutputSchema.validate(matched_researchers, lazy=True)
            if source == "openalex"
            else MatchingOutputSchema.validate(matched_researchers, lazy=True)
        ),
        article_authors=_validate_article_authors(article_authors, source=source),
        article_references=_validate_article_references(article_references),
        openalex_citation_index=None,
    )


def load_name_disambiguation_inputs(
    config: NameDisambiguationConfig,
) -> NameDisambiguationInputs:
    """Load configured matcher and external evidence inputs.

    :param config: Resolved canonical input paths.
    :return: Validated in-memory input tables.
    :raises FileNotFoundError: If the matcher artifact or an external dataset is absent.
    :raises ValueError: If an input cannot be read or violates its stage contract.
    """
    config.check_external_inputs_exist()
    matching = validate_matching_artifact(config.matching_output, source=config.source)
    if config.source == "openalex":
        assert config.author_affiliations_file_path is not None
        candidate_ids = _ambiguous_openalex_author_ids(matching)
        _logger.info(
            "Selected OpenAlex authors for disambiguation evidence",
            matched_author_rows=len(matching),
            ambiguous_candidate_author_count=len(candidate_ids),
        )
        article_authors = load_openalex_work_evidence(
            config.article_author_file_path,
            candidate_author_ids=candidate_ids,
            affiliations_path=config.author_affiliations_file_path,
            max_workers=config.effective_openalex_scan_workers,
        )
        citation_index = load_openalex_citation_edges(
            config.article_reference_file_path,
            candidate_work_ids=set(article_authors["eid"].dropna().astype(str)),
        )
        article_references = pd.DataFrame(columns=ARTICLE_REFERENCE_INPUT_COLUMNS)
    else:
        citation_index = None
        article_authors = read_dataframe(config.article_author_file_path)
        article_references = read_dataframe(config.article_reference_file_path)
    return NameDisambiguationInputs(
        matched_researchers=matching,
        article_authors=_validate_article_authors(article_authors, source=config.source),
        article_references=_validate_article_references(article_references),
        openalex_citation_index=citation_index,
    )


def _ambiguous_openalex_author_ids(matching: pd.DataFrame) -> set[str]:
    """Select matcher author IDs belonging to multi-row researcher groups.

    This intentionally uses the same row-count rule as
    :func:`disambiguate_batches`; repeated author IDs are not deduplicated here.

    :param matching: Validated matcher artifact.
    :return: Stable OpenAlex author IDs from ambiguous ``sc_id`` groups.
    """
    group_sizes = matching.groupby(RESEARCHER_ID, dropna=True)[AUTHOR_ID].transform("size")
    ambiguous = matching.loc[group_sizes.gt(1), AUTHOR_ID]
    return set(ambiguous.dropna().astype(str))


def _validate_article_authors(
    frame: pd.DataFrame, *, source: Literal["scopus", "openalex"]
) -> pd.DataFrame:
    """Project article evidence fields and validate their schema.

    :param frame: Caller- or file-provided article-author data.
    :return: Validated evidence columns, preserving their source ordering.
    """
    columns = [
        column for column in ARTICLE_AUTHOR_EVIDENCE_INPUT_COLUMNS if column in frame.columns
    ]
    if source == "openalex" and OPENALEX_WORK_FIELD_IDS in frame:
        columns.append(OPENALEX_WORK_FIELD_IDS)
    projected = frame.loc[
        :,
        columns,
    ]
    return ArticleAuthorEvidenceSchema.validate(projected, lazy=True)


def _validate_article_references(frame: pd.DataFrame) -> pd.DataFrame:
    """Project citation-edge fields and validate their schema.

    :param frame: Caller- or file-provided citation edges.
    :return: Validated citation columns, preserving their source ordering.
    """
    projected = frame.loc[
        :,
        [column for column in ARTICLE_REFERENCE_INPUT_COLUMNS if column in frame.columns],
    ]
    return ArticleReferenceSchema.validate(projected, lazy=True)
