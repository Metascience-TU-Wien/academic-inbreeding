# Copyright © 2026 TU Wien

"""TOML configuration loading for name matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class NameMatchingConfig:
    """Validated inputs and options for one name-matching run."""

    researcher_file_path: Path
    candidate_authors_file: Path
    resolved_matches_output_file_path: Path
    enable_given_name_aliases: bool = True
    enable_organisation_filter: bool = True
    existing_scopus_author_id_column: str | None = None
    max_workers: int = 3
    encoding: str = "utf-8-sig"

    def check_paths_exist(self) -> None:
        """Raise when either configured input path is absent.

        :raises FileNotFoundError: If a researcher or author input is missing.
        """
        missing = [
            path
            for path in (self.researcher_file_path, self.candidate_authors_file)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing name-matching inputs: " + ", ".join(map(str, missing)))


def load_name_matching_config(path: str | Path) -> NameMatchingConfig:
    """Load name-matching settings and shared external paths.

    :param path: Pipeline TOML file.
    :return: Validated name-matching settings.
    :raises KeyError: If required stage sections or input paths are missing.
    :raises ValueError: If centralized and legacy input paths conflict.
    """
    from acib.pipeline.config import load_name_matching_config as load_matching_config

    return load_matching_config(path)
