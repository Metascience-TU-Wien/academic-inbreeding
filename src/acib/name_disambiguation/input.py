# Copyright © 2026 TU Wien

"""Loading and validation for name-disambiguation inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from acib.name_disambiguation.config import NameDisambiguationConfig
from acib.name_disambiguation.constants import (
    ARTICLE_AUTHOR_EVIDENCE_INPUT_COLUMNS,
    ARTICLE_REFERENCE_INPUT_COLUMNS,
)
from acib.name_disambiguation.results import NameDisambiguationInputs
from acib.schemas.matching import MatchingOutputSchema
from acib.schemas.publications import (
    ArticleAuthorEvidenceSchema,
    ArticleReferenceSchema,
)


def validate_matching_artifact(path: str | Path) -> pd.DataFrame:
    """Load and validate the producer-owned name-matching artifact.

    :param path: Canonical resolved-researchers CSV path.
    :return: Validated researcher-to-author candidate rows.
    :raises FileNotFoundError: If the required artifact is missing.
    :raises ValueError: If the file cannot be read or violates its contract.
    """
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Name-matching upstream artifact is missing: {artifact}")
    try:
        frame = pd.read_csv(artifact, low_memory=False)
        return MatchingOutputSchema.validate(frame, lazy=True)
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
) -> NameDisambiguationInputs:
    """Validate caller-provided DataFrames at the stage input boundary.

    :param matched_researchers: Rows produced by name matching.
    :param article_authors: Publication-author evidence rows.
    :param article_references: Citation edges.
    :return: Validated inputs suitable for in-memory disambiguation.
    """
    return NameDisambiguationInputs(
        matched_researchers=MatchingOutputSchema.validate(matched_researchers, lazy=True),
        article_authors=_validate_article_authors(article_authors),
        article_references=_validate_article_references(article_references),
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
    matching = validate_matching_artifact(config.matching_output)
    article_authors = read_dataframe(config.article_author_file_path)
    article_references = read_dataframe(config.article_reference_file_path)
    return NameDisambiguationInputs(
        matched_researchers=matching,
        article_authors=_validate_article_authors(article_authors),
        article_references=_validate_article_references(article_references),
    )


def _validate_article_authors(frame: pd.DataFrame) -> pd.DataFrame:
    """Project article evidence fields and validate their schema.

    :param frame: Caller- or file-provided article-author data.
    :return: Validated evidence columns, preserving their source ordering.
    """
    projected = frame.loc[
        :,
        [column for column in ARTICLE_AUTHOR_EVIDENCE_INPUT_COLUMNS if column in frame.columns],
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
