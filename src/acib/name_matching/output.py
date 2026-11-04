# Copyright © 2026 TU Wien

"""Persist validated name-matching results."""

from __future__ import annotations

from pathlib import Path

from acib.constants import AFFILIATION_ID, AFFILIATION_ID_PARENT
from acib.name_matching.config import NameMatchingConfig
from acib.name_matching.results import NameMatchingResult
from acib.schemas import MatchingOutputSchema

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def save_name_matching_results(
    result: NameMatchingResult,
    config: NameMatchingConfig,
    *,
    overwrite: bool = False,
) -> None:
    """Write the canonical resolved-researchers CSV artifact.

    :param result: In-memory result produced by matching.
    :param config: Configuration owning the output directory and encoding.
    :param overwrite: Whether an existing matching artifact may be replaced.
    :raises FileExistsError: If the artifact exists and replacement was not allowed.
    :raises ValueError: If the configured output resolves inside the read-only source.
    """
    output_path = validate_name_matching_output_path(config, overwrite=overwrite)

    output = MatchingOutputSchema.validate(result.resolved_researchers, lazy=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_output = output.copy()
    for column in (AFFILIATION_ID, AFFILIATION_ID_PARENT):
        if column in serialized_output:
            serialized_output[column] = serialized_output[column].astype("Float64")
    serialized_output.to_csv(output_path, index=False, encoding=config.encoding)


def validate_name_matching_output_path(
    config: NameMatchingConfig,
    *,
    overwrite: bool = False,
) -> Path:
    """Check output overwrite permission and protect the read-only source tree.

    :param config: Configuration owning the resolved-researchers output path.
    :param overwrite: Whether an existing artifact may be replaced.
    :return: Resolved output path.
    :raises FileExistsError: If the artifact exists and replacement was not allowed.
    :raises ValueError: If the output path is inside the source repository.
    """
    output_path = config.outputs.resolved_researchers.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Name-matching output already exists: {output_path}. "
            "Pass overwrite=True explicitly to replace it."
        )
    source_repository = (_PROJECT_ROOT.parent / "name-matching-playground").resolve()
    if output_path.is_relative_to(source_repository):
        raise ValueError(
            "Refusing to write name-matching output inside the read-only source repository: "
            f"{source_repository}"
        )
    return output_path
