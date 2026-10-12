# Copyright © 2026 TU Wien

"""Configuration for the independently executable disambiguation stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from acib.name_disambiguation.constants import (
    CLUSTERS_FILENAME,
    FINAL_AUTHOR_SELECTION_FILENAME,
    FINAL_AUTHOR_SELECTION_SCORES_FILENAME,
    PAIR_DECISIONS_FILENAME,
    REPORT_JSONL_FILENAME,
    REPORT_TXT_FILENAME,
)


@dataclass(frozen=True, slots=True)
class NameDisambiguationOutputs:
    """Output directory with canonical paths for six disambiguation artifacts."""

    directory: Path

    @property
    def pair_decisions(self) -> Path:
        """Return the pairwise-decision CSV path."""
        return self.directory / PAIR_DECISIONS_FILENAME

    @property
    def clusters(self) -> Path:
        """Return the cluster-membership CSV path."""
        return self.directory / CLUSTERS_FILENAME

    @property
    def final_author_selection(self) -> Path:
        """Return the final researcher-selection CSV path."""
        return self.directory / FINAL_AUTHOR_SELECTION_FILENAME

    @property
    def final_author_selection_scores(self) -> Path:
        """Return the candidate-cluster score CSV path."""
        return self.directory / FINAL_AUTHOR_SELECTION_SCORES_FILENAME

    @property
    def report_txt(self) -> Path:
        """Return the human-readable report path."""
        return self.directory / REPORT_TXT_FILENAME

    @property
    def report_jsonl(self) -> Path:
        """Return the JSON-lines report path."""
        return self.directory / REPORT_JSONL_FILENAME

    def all_paths(self) -> tuple[Path, ...]:
        """Return output paths in write order.

        :return: Configured artifact paths in persistence order.
        """
        return (
            self.pair_decisions,
            self.clusters,
            self.final_author_selection,
            self.final_author_selection_scores,
            self.report_txt,
            self.report_jsonl,
        )


@dataclass(frozen=True, slots=True)
class NameDisambiguationConfig:
    """Resolved upstream/data inputs and scoring options for one run.

    :param matching_output: Resolved name-matching artifact.
    :param article_author_file_path: Article-author evidence dataset.
    :param article_reference_file_path: Citation-edge evidence dataset.
    :param outputs: Paths for all disambiguation artifacts.
    :param enable_reporting: Whether to print reports to standard output.
    :param same_author_score_threshold: Minimum same-author pair score.
    :param possible_author_score_threshold: Minimum possible-match pair score.
    :param max_workers: Maximum batch worker processes.
    """

    matching_output: Path
    article_author_file_path: Path
    article_reference_file_path: Path
    outputs: NameDisambiguationOutputs
    enable_reporting: bool = True
    same_author_score_threshold: float = 6.0
    possible_author_score_threshold: float = 3.0
    max_workers: int = 1

    def __post_init__(self) -> None:
        """Enforce the original scoring and worker constraints.

        :raises ValueError: If thresholds or worker count are invalid.
        """
        if self.same_author_score_threshold < 0:
            raise ValueError("same_author_score_threshold must be nonnegative")
        if self.possible_author_score_threshold < 0:
            raise ValueError("possible_author_score_threshold must be nonnegative")
        if self.possible_author_score_threshold > self.same_author_score_threshold:
            raise ValueError(
                "possible_author_score_threshold must be less than or equal to "
                "same_author_score_threshold"
            )
        if (
            not isinstance(self.max_workers, int)
            or isinstance(self.max_workers, bool)
            or self.max_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer")

    def check_external_inputs_exist(self) -> None:
        """Verify required article-level evidence inputs.

        :raises FileNotFoundError: If an article-level input is absent.
        """
        missing = [
            path
            for path in (self.article_author_file_path, self.article_reference_file_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing name-disambiguation external inputs: " + ", ".join(map(str, missing))
            )

    def check_output_paths(self, *, overwrite: bool = False) -> None:
        """Reject existing artifacts unless replacement was explicitly requested.

        :param overwrite: Permit replacement of existing stage outputs.
        :raises FileExistsError: If any output exists and overwrite is disabled.
        """
        existing = [path for path in self.outputs.all_paths() if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                "Name-disambiguation outputs already exist; use --overwrite to replace: "
                + ", ".join(map(str, existing))
            )


def load_name_disambiguation_config(path: str | Path) -> NameDisambiguationConfig:
    """Load disambiguation settings through the shared pipeline loader.

    :param path: Pipeline TOML path.
    :return: Resolved stage configuration.
    :raises KeyError: If the disambiguation stage is not configured.
    """
    from acib.pipeline.config import load_name_disambiguation_config as load_config

    return load_config(path)
