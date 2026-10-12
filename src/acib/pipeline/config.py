# Copyright © 2026 TU Wien

"""Shared TOML configuration for the currently migrated pipeline stages."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acib.name_disambiguation.config import (
    NameDisambiguationConfig,
    NameDisambiguationOutputs,
)
from acib.name_matching.config import NameMatchingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Paths to external datasets shared across pipeline stages."""

    researchers: Path
    scopus_authors: Path
    article_author: Path | None = None
    article_reference: Path | None = None


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configurations for the migrated matching and disambiguation stages."""

    data: DataConfig
    name_matching: NameMatchingConfig
    name_disambiguation: NameDisambiguationConfig | None = None


def resolve_config_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """Expand and resolve a config path against the project root.

    :param value: Absolute or project-relative path.
    :param base: Base directory for relative paths.
    :return: Resolved absolute path.
    """
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()


def _read_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_config_path(path)
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    return raw


def _table(raw: Mapping[str, Any], name: str, *, required: bool) -> Mapping[str, Any]:
    value = raw.get(name)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise KeyError(f"Missing or invalid config section [{name}]")
    return value


def _configured_path(
    *,
    data: Mapping[str, Any],
    data_key: str,
    legacy_section: Mapping[str, Any],
    legacy_key: str,
    data_aliases: tuple[str, ...] = (),
    required: bool,
) -> Path | None:
    shared_values = [
        (key, data[key]) for key in (data_key, *data_aliases) if data.get(key) is not None
    ]
    shared_paths = [(key, resolve_config_path(value)) for key, value in shared_values]
    if len({resolved for _, resolved in shared_paths}) > 1:
        raise ValueError(
            "Conflicting shared dataset paths: " + ", ".join(key for key, _ in shared_paths)
        )
    legacy_value = legacy_section.get(legacy_key)
    shared_path = shared_paths[0][1] if shared_paths else None
    legacy_path = resolve_config_path(legacy_value) if legacy_value is not None else None
    if shared_path is not None and legacy_path is not None and shared_path != legacy_path:
        raise ValueError(
            f"Conflicting path declarations: [data].{data_key} and "
            f"the legacy {legacy_key!r} stage path"
        )
    result = shared_path or legacy_path
    if result is None and required:
        raise KeyError(f"Missing required external dataset path: [data].{data_key}")
    return result


def _positive_integer(section: Mapping[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _nonnegative_number(section: Mapping[str, Any], key: str, default: float) -> float:
    value = section.get(key, default)
    if not isinstance(value, (float, int)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a nonnegative number")
    return float(value)


def load_name_matching_config(path: str | Path) -> NameMatchingConfig:
    """Load the matching stage and its centralized external input paths.

    :param path: Pipeline TOML file.
    :return: Name-matching configuration.
    :raises KeyError: If required sections or paths are missing.
    :raises ValueError: If duplicated legacy and shared paths conflict.
    """
    raw = _read_config(path)
    section = _table(raw, "name_matching", required=True)
    data = _table(raw, "data", required=False)
    researchers = _configured_path(
        data=data,
        data_key="researchers",
        legacy_section=section,
        legacy_key="researcher_file_path",
        data_aliases=("researcher_file_path",),
        required=True,
    )
    authors = _configured_path(
        data=data,
        data_key="scopus_authors",
        legacy_section=section,
        legacy_key="candidate_authors_file",
        data_aliases=("candidate_authors_file",),
        required=True,
    )
    assert researchers is not None and authors is not None
    aliases = section.get("enable_given_name_aliases", True)
    organization_filter = section.get("enable_organisation_filter", True)
    encoding = section.get("encoding", "utf-8-sig")
    if not isinstance(aliases, bool) or not isinstance(organization_filter, bool):
        raise TypeError("Name-matching flags must be TOML booleans")
    if not isinstance(encoding, str) or not encoding:
        raise ValueError("name_matching.encoding must be a nonempty string")
    return NameMatchingConfig(
        researcher_file_path=researchers,
        candidate_authors_file=authors,
        resolved_matches_output_file_path=resolve_config_path(
            section["resolved_matches_output_file_path"]
        ),
        enable_given_name_aliases=aliases,
        enable_organisation_filter=organization_filter,
        existing_scopus_author_id_column=section.get("existing_scopus_author_id_column"),
        max_workers=_positive_integer(section, "max_workers", 3),
        encoding=encoding,
    )


def load_name_disambiguation_config(path: str | Path) -> NameDisambiguationConfig:
    """Load disambiguation, its matching dependency, and centralized article inputs.

    :param path: Pipeline TOML file.
    :return: Disambiguation configuration.
    :raises KeyError: If required inputs, outputs, or stage sections are missing.
    :raises ValueError: If stage dependencies or legacy path declarations conflict.
    """
    raw = _read_config(path)
    section = _table(raw, "name_disambiguation", required=True)
    matching = load_name_matching_config(path)
    data = _table(raw, "data", required=False)
    legacy_author_path = section.get("author_file_path")
    if (
        legacy_author_path is not None
        and resolve_config_path(legacy_author_path) != matching.resolved_matches_output_file_path
    ):
        raise ValueError(
            "name_disambiguation.author_file_path conflicts with "
            "name_matching.resolved_matches_output_file_path"
        )
    article_author = _configured_path(
        data=data,
        data_key="article_author",
        legacy_section=section,
        legacy_key="article_author_file_path",
        data_aliases=("article_author_file_path",),
        required=True,
    )
    article_reference = _configured_path(
        data=data,
        data_key="article_reference",
        legacy_section=section,
        legacy_key="article_reference_file_path",
        data_aliases=("article_reference_file_path",),
        required=True,
    )
    assert article_author is not None and article_reference is not None
    reporting = section.get("enable_reporting", True)
    if not isinstance(reporting, bool):
        raise TypeError("name_disambiguation.enable_reporting must be a TOML boolean")
    return NameDisambiguationConfig(
        matching_output=matching.resolved_matches_output_file_path,
        article_author_file_path=article_author,
        article_reference_file_path=article_reference,
        outputs=_load_disambiguation_outputs(section),
        enable_reporting=reporting,
        same_author_score_threshold=_nonnegative_number(
            section, "same_author_score_threshold", 6.0
        ),
        possible_author_score_threshold=_nonnegative_number(
            section, "possible_author_score_threshold", 3.0
        ),
        max_workers=_positive_integer(section, "max_workers", 1),
    )


def _load_disambiguation_outputs(section: Mapping[str, Any]) -> NameDisambiguationOutputs:
    output_names = {
        "pair_decisions": "pair_decisions.csv",
        "clusters": "clusters.csv",
        "final_author_selection": "final_author_selection.csv",
        "final_author_selection_scores": "final_author_selection_scores.csv",
        "report_txt": "report.txt",
        "report_jsonl": "report.jsonl",
    }
    outputs = section.get("outputs")
    legacy_directory = section.get("disambiguated_author_path")
    if outputs is None:
        if legacy_directory is None:
            raise KeyError(
                "Missing [name_disambiguation.outputs] or legacy "
                "name_disambiguation.disambiguated_author_path"
            )
        output_root = resolve_config_path(legacy_directory)
        resolved = {key: output_root / filename for key, filename in output_names.items()}
    else:
        if not isinstance(outputs, dict):
            raise ValueError("[name_disambiguation.outputs] must be a TOML table")
        missing = set(output_names) - outputs.keys()
        if missing:
            raise KeyError("Missing output declarations: " + ", ".join(sorted(missing)))
        resolved = {key: resolve_config_path(outputs[key]) for key in output_names}
        if legacy_directory is not None:
            output_root = resolve_config_path(legacy_directory)
            if any(
                resolved[key] != output_root / filename for key, filename in output_names.items()
            ):
                raise ValueError(
                    "disambiguated_author_path conflicts with " "[name_disambiguation.outputs]"
                )
    return NameDisambiguationOutputs(**resolved)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load configuration and dependencies for the migrated pipeline stages.

    :param path: Pipeline TOML file.
    :return: Centralized data paths and configured stage settings.
    """
    raw = _read_config(path)
    data = _table(raw, "data", required=False)
    matching_section = _table(raw, "name_matching", required=True)
    disambiguation_section = _table(raw, "name_disambiguation", required=False)
    researchers = _configured_path(
        data=data,
        data_key="researchers",
        legacy_section=matching_section,
        legacy_key="researcher_file_path",
        data_aliases=("researcher_file_path",),
        required=True,
    )
    scopus_authors = _configured_path(
        data=data,
        data_key="scopus_authors",
        legacy_section=matching_section,
        legacy_key="candidate_authors_file",
        data_aliases=("candidate_authors_file",),
        required=True,
    )
    article_author = _configured_path(
        data=data,
        data_key="article_author",
        legacy_section=disambiguation_section,
        legacy_key="article_author_file_path",
        data_aliases=("article_author_file_path",),
        required=bool(disambiguation_section),
    )
    article_reference = _configured_path(
        data=data,
        data_key="article_reference",
        legacy_section=disambiguation_section,
        legacy_key="article_reference_file_path",
        data_aliases=("article_reference_file_path",),
        required=bool(disambiguation_section),
    )
    assert researchers is not None and scopus_authors is not None
    data_config = DataConfig(researchers, scopus_authors, article_author, article_reference)
    matching_config = load_name_matching_config(path)
    disambiguation_config = (
        load_name_disambiguation_config(path) if disambiguation_section else None
    )
    return PipelineConfig(data_config, matching_config, disambiguation_config)
