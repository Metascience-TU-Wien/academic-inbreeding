# Copyright © 2026 TU Wien

"""Input validation and execution boundary for name disambiguation."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from acib.name_disambiguation.config import NameDisambiguationConfig
from acib.name_disambiguation.data import prepare_disambiguation_author_frame
from acib.name_disambiguation.output import write_all_outputs
from acib.name_disambiguation.reporting import render_text_report
from acib.name_disambiguation.service import disambiguate_batches
from acib.schemas.authors import DisambiguationAuthorSchema
from acib.schemas.matching import MatchingOutputSchema
from acib.schemas.publications import (
    ArticleAuthorEvidenceSchema,
    ArticleReferenceSchema,
)


def validate_matching_artifact(path: str | Path) -> pd.DataFrame:
    """Load and validate the output contract provided by name matching.

    :param path: Resolved matcher output CSV.
    :return: Validated researcher-to-author candidate rows.
    :raises FileNotFoundError: If the matcher output does not exist.
    :raises ValueError: If the file is unreadable or fails its schema contract.
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
    """Read one configured CSV or Parquet input.

    :param path: Input dataset path.
    :return: Dataset rows without altering source column names.
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


def run_name_disambiguation(
    config: NameDisambiguationConfig,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run disambiguation from the configured matcher artifact and evidence tables.

    :param config: Resolved disambiguation paths and scoring options.
    :param overwrite: Permit replacing existing configured output artifacts.
    :return: Validated final-selection rows, one per processed researcher batch.
    :raises FileNotFoundError: If an upstream or external input is absent.
    :raises ValueError: If an input fails validation.
    :raises FileExistsError: If outputs exist and overwrite is disabled.
    """
    config.check_output_paths(overwrite=overwrite)
    config.check_external_inputs_exist()

    matching = validate_matching_artifact(config.matching_output)
    authors = prepare_disambiguation_author_frame(matching)
    authors = DisambiguationAuthorSchema.validate(authors, lazy=True)

    article_authors = ArticleAuthorEvidenceSchema.validate(
        read_dataframe(config.article_author_file_path), lazy=True
    )
    article_references = ArticleReferenceSchema.validate(
        read_dataframe(config.article_reference_file_path), lazy=True
    )
    batch_results = disambiguate_batches(
        authors,
        article_author_df=article_authors,
        article_reference_df=article_references,
        same_author_score_threshold=config.same_author_score_threshold,
        possible_author_score_threshold=config.possible_author_score_threshold,
        max_workers=config.max_workers,
    )
    if config.enable_reporting:
        for result in batch_results:
            sys.stdout.write(
                render_text_report(
                    result.decisions,
                    result.clusters,
                    result.final_selection,
                    result.final_selection_scores,
                )
            )
    write_all_outputs(batch_results, config, overwrite=overwrite)
    return pd.DataFrame([asdict(result.final_selection) for result in batch_results])
