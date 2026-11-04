# Copyright © 2026 TU Wien

"""Read and validate name-matching stage inputs."""

from __future__ import annotations

import pandas as pd

from acib.constants import RESEARCHER_ID
from acib.name_matching.config import NameMatchingConfig
from acib.name_matching.constants import PARENT_AFFILIATION_COLUMNS
from acib.name_matching.results import NameMatchingInputs
from acib.schemas import ResearcherInputSchema, ScopusAuthorInputSchema
from acib.schemas.affiliations import parse_nullable_affiliation_ids


def prepare_name_matching_inputs(
    researchers: pd.DataFrame,
    candidate_authors: pd.DataFrame,
    *,
    researcher_output_metadata: pd.DataFrame | None = None,
) -> NameMatchingInputs:
    """Validate and normalize already-loaded matching inputs.

    :param researchers: Researcher records from a caller or configured dataset.
    :param candidate_authors: Scopus-author candidate records.
    :param researcher_output_metadata: Optional lexical source metadata for output.
    :return: Validated in-memory inputs ready for matching.
    """
    validated_researchers = ResearcherInputSchema.validate(researchers, lazy=True)
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


def load_name_matching_inputs(config: NameMatchingConfig) -> NameMatchingInputs:
    """Load configured CSV inputs without performing file output.

    :param config: Resolved matching input paths and options.
    :return: Loaded in-memory input tables ready for validation and matching.
    :raises FileNotFoundError: If a configured input is absent.
    """
    config.check_paths_exist()
    researchers = pd.read_csv(config.researcher_file_path)
    candidate_authors = pd.read_csv(
        config.candidate_authors_file,
        dtype={column: "string" for column in PARENT_AFFILIATION_COLUMNS},
    )
    researcher_output_metadata = pd.read_csv(
        config.researcher_file_path,
        dtype="string",
        keep_default_na=False,
    )
    return NameMatchingInputs(
        researchers=researchers,
        candidate_authors=candidate_authors,
        researcher_output_metadata=researcher_output_metadata,
    )
