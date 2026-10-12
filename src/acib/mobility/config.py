# Copyright © 2026 TU Wien

"""Configuration models for academic mobility analysis."""

from dataclasses import dataclass
from pathlib import Path

from acib.mobility.constants import EARLY_AFFILIATION_METHODS


@dataclass(frozen=True, slots=True)
class MobilityOutputs:
    """Configured artifacts produced by the mobility stage.

    :param author_affiliations: Per-author affiliation-history CSV.
    :param author_affiliations_unique: Unique top-level affiliation CSV.
    :param author_inbreeding_proxy: Author-level inbreeding and mobility CSV.
    :param author_articles: Unique article-level CSV.
    """

    author_affiliations: Path
    author_affiliations_unique: Path
    author_inbreeding_proxy: Path
    author_articles: Path

    def all_paths(self) -> tuple[Path, ...]:
        """Return mobility artifact paths in source write order.

        :return: Configured output paths.
        """
        return (
            self.author_affiliations,
            self.author_affiliations_unique,
            self.author_inbreeding_proxy,
            self.author_articles,
        )


@dataclass(frozen=True, slots=True)
class MobilityConfig:
    """Resolved mobility dependencies, outputs, and source-compatible options.

    :param authors: Final author-selection artifact from name disambiguation.
    :param researcher_metadata: Researcher metadata artifact from name matching.
    :param article_author: External article-author evidence dataset.
    :param affiliations: External GeoPackage affiliation dataset.
    :param outputs: Explicit paths for the four mobility artifacts.
    :param enable_university_affiliation_inbreeding_detection: Use the stricter
        university/early-affiliation inbreeding rule.
    :param enable_coaffiliated_professor_merge: Merge professor rows for the
        same selected author profile in memory.
    :param professor_title_pattern: Raw title pattern for professor filtering.
    :param first_affiliation_publication_count: Number of early publications
        considered by the publication-count method.
    :param early_affiliation_method: Early-affiliation estimation method.
    :param first_affiliation_year_count: Number of years in the early window.
    :param min_publication_history_years: Optional minimum history length.
    :param max_workers: Maximum preprocessing/analysis worker processes.
    """

    authors: Path
    researcher_metadata: Path
    article_author: Path
    affiliations: Path
    outputs: MobilityOutputs
    enable_university_affiliation_inbreeding_detection: bool = False
    enable_coaffiliated_professor_merge: bool = False
    professor_title_pattern: str = "prof"
    first_affiliation_publication_count: int = 3
    early_affiliation_method: str = "first_n_publications"
    first_affiliation_year_count: int = 2
    min_publication_history_years: int | None = None
    max_workers: int = 3

    def __post_init__(self) -> None:
        """Validate mobility options without changing source defaults.

        :raises ValueError: If an option is outside the source-supported range.
        """
        if self.early_affiliation_method not in EARLY_AFFILIATION_METHODS:
            raise ValueError(
                "early_affiliation_method must be 'first_n_publications' or 'first_n_years'"
            )
        if self.first_affiliation_publication_count < 1:
            raise ValueError("first_affiliation_publication_count must be at least one")
        if self.first_affiliation_year_count < 1:
            raise ValueError("first_affiliation_year_count must be at least one")
        if (
            self.min_publication_history_years is not None
            and self.min_publication_history_years < 0
        ):
            raise ValueError("min_publication_history_years must be nonnegative or None")
        if self.max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if len(set(self.outputs.all_paths())) != len(self.outputs.all_paths()):
            raise ValueError("Mobility output paths must be distinct")

    def check_input_paths(self) -> None:
        """Verify all mobility inputs and identify each artifact's producer.

        :raises FileNotFoundError: If a required external input or upstream
            artifact is absent.
        """
        inputs = (
            (
                "[name_disambiguation.outputs].final_author_selection",
                "name_disambiguation",
                self.authors,
            ),
            (
                "[name_matching].resolved_matches_output_file_path",
                "name_matching",
                self.researcher_metadata,
            ),
            ("[data].article_author_file_path", "external dataset", self.article_author),
            (
                "[data].author_affiliations_file_path",
                "external dataset",
                self.affiliations,
            ),
        )
        missing = [
            f"{key} (producer: {producer}): {path}"
            for key, producer, path in inputs
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError("Missing required mobility inputs:\n- " + "\n- ".join(missing))
