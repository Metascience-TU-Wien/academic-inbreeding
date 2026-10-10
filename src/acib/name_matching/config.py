# Copyright © 2026 TU Wien

"""TOML configuration loading for name matching."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class NameMatchingConfig:
    """Validated inputs and options for one name-matching run."""

    source: Literal["scopus"]
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


def _resolve_config_path(value: str, base: Path) -> Path:
    expanded = Path(os.path.expandvars(value)).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()


def load_name_matching_config(path: str | Path) -> NameMatchingConfig:
    """Load the ``[name_matching]`` table from a pipeline TOML file.

    Relative paths resolve from the project root, matching the source
    configuration loader; environment variables and ``~`` are expanded.

    :param path: Pipeline TOML file.
    :return: Validated name-matching settings.
    :raises KeyError: If the required table or path settings are missing.
    :raises ValueError: If a setting has an unsupported value or type.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    section = raw.get("name_matching")
    if not isinstance(section, dict):
        raise KeyError("Missing required config section [name_matching]")
    try:
        source = section.get("source", "scopus")
        if source != "scopus":
            raise ValueError(f"Unsupported name-matching source: {source!r}")
        max_workers = section.get("max_workers", 3)
        if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        return NameMatchingConfig(
            source=source,
            researcher_file_path=_resolve_config_path(
                section["researcher_file_path"], PROJECT_ROOT
            ),
            candidate_authors_file=_resolve_config_path(
                section["candidate_authors_file"], PROJECT_ROOT
            ),
            resolved_matches_output_file_path=_resolve_config_path(
                section["resolved_matches_output_file_path"], PROJECT_ROOT
            ),
            enable_given_name_aliases=section.get("enable_given_name_aliases", True),
            enable_organisation_filter=section.get("enable_organisation_filter", True),
            existing_scopus_author_id_column=section.get("existing_scopus_author_id_column"),
            max_workers=max_workers,
            encoding=section.get("encoding", "utf-8-sig"),
        )
    except KeyError as exc:
        raise KeyError(f"Missing required name-matching config value: {exc.args[0]}") from exc
