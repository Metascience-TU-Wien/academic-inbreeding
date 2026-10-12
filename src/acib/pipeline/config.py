# Copyright © 2026 TU Wien

"""Shared TOML configuration for the currently migrated pipeline stages."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from acib.mobility.config import MobilityConfig, MobilityOutputs
from acib.name_disambiguation.config import (
    NameDisambiguationConfig,
    NameDisambiguationOutputs,
)
from acib.name_matching.config import NameMatchingConfig
from acib.regression.config import (
    LoadedRegressionInputs,
    RegressionConfig,
    RegressionModelConfig,
    RegressionModelType,
    RegressionOutcomeMetadata,
    RegressionOutcomeType,
    RegressionOutputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Paths to external datasets shared across pipeline stages."""

    researcher_file_path: Path
    candidate_authors_file: Path
    article_author_file_path: Path | None = None
    article_reference_file_path: Path | None = None
    author_affiliations_file_path: Path | None = None
    article_impact_file_path: Path | None = None
    disruption_indicator_file_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configurations for the independently invoked migrated pipeline stages."""

    data: DataConfig
    name_matching: NameMatchingConfig
    name_disambiguation: NameDisambiguationConfig | None = None
    mobility: MobilityConfig | None = None
    regression: RegressionConfig | None = None


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


def _data_path(data: Mapping[str, Any], key: str, *, required: bool) -> Path | None:
    """Resolve one canonical external dataset declaration.

    :param data: The shared ``[data]`` TOML table.
    :param key: Canonical external dataset key.
    :param required: Whether the key must be configured.
    :return: Resolved project-root-relative or absolute path.
    :raises KeyError: If a required key is absent.
    :raises TypeError: If a configured key is not a string or path.
    """
    value = data.get(key)
    if value is None:
        if required:
            raise KeyError(f"Missing required external dataset path: [data].{key}")
        return None
    if not isinstance(value, str):
        raise TypeError(f"[data].{key} must be a path string")
    return resolve_config_path(value)


def _reject_stage_path_duplicates(
    section: Mapping[str, Any], stage: str, keys: tuple[str, ...]
) -> None:
    """Reject stage-local path declarations owned by shared or producer config.

    :param section: Stage TOML table.
    :param stage: Stage name used in the error message.
    :param keys: Forbidden redundant path keys.
    :raises ValueError: If any redundant path key is present.
    """
    duplicates = sorted(set(keys).intersection(section))
    if duplicates:
        raise ValueError(
            f"[{stage}] must not duplicate shared or upstream paths; remove: "
            + ", ".join(f"[{stage}].{key}" for key in duplicates)
        )


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
    _reject_stage_path_duplicates(
        section,
        "name_matching",
        ("researcher_file_path", "candidate_authors_file"),
    )
    data = _table(raw, "data", required=False)
    researchers = _data_path(data, "researcher_file_path", required=True)
    authors = _data_path(data, "candidate_authors_file", required=True)
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
    _reject_stage_path_duplicates(
        section,
        "name_disambiguation",
        (
            "author_file_path",
            "article_author_file_path",
            "article_reference_file_path",
            "disambiguated_author_path",
        ),
    )
    matching = load_name_matching_config(path)
    data = _table(raw, "data", required=False)
    article_author = _data_path(data, "article_author_file_path", required=True)
    article_reference = _data_path(data, "article_reference_file_path", required=True)
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
    if outputs is None:
        raise KeyError("Missing required [name_disambiguation.outputs] table")
    if not isinstance(outputs, dict):
        raise TypeError("[name_disambiguation.outputs] must be a TOML table")
    missing = set(output_names) - outputs.keys()
    if missing:
        raise KeyError("Missing output declarations: " + ", ".join(sorted(missing)))
    unexpected = outputs.keys() - output_names.keys()
    if unexpected:
        raise ValueError(
            "Unrecognized [name_disambiguation.outputs] keys: " + ", ".join(sorted(unexpected))
        )
    resolved: dict[str, Path] = {}
    for key in output_names:
        value = outputs[key]
        if not isinstance(value, str):
            raise TypeError(f"[name_disambiguation.outputs].{key} must be a path string")
        resolved[key] = resolve_config_path(value)
    return NameDisambiguationOutputs(**resolved)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load configuration and dependencies for the migrated pipeline stages.

    :param path: Pipeline TOML file.
    :return: Centralized data paths and configured stage settings.
    """
    raw = _read_config(path)
    data = _table(raw, "data", required=False)
    _table(raw, "name_matching", required=True)
    disambiguation_section = _table(raw, "name_disambiguation", required=False)
    researchers = _data_path(data, "researcher_file_path", required=True)
    scopus_authors = _data_path(data, "candidate_authors_file", required=True)
    article_author = _data_path(
        data, "article_author_file_path", required=bool(disambiguation_section)
    )
    article_reference = _data_path(
        data, "article_reference_file_path", required=bool(disambiguation_section)
    )
    author_affiliations = _data_path(data, "author_affiliations_file_path", required=False)
    article_impact = _data_path(data, "article_impact_file_path", required=False)
    disruption_indicators = _data_path(data, "disruption_indicator_file_path", required=False)
    assert researchers is not None and scopus_authors is not None
    data_config = DataConfig(
        researchers,
        scopus_authors,
        article_author,
        article_reference,
        author_affiliations,
        article_impact,
        disruption_indicators,
    )
    matching_config = load_name_matching_config(path)
    disambiguation_config = (
        load_name_disambiguation_config(path) if disambiguation_section else None
    )
    mobility_config = load_mobility_config(path) if "mobility" in raw else None
    regression_config = load_regression_config(path) if "regression" in raw else None
    return PipelineConfig(
        data_config,
        matching_config,
        disambiguation_config,
        mobility_config,
        regression_config,
    )


def load_mobility_config(path: str | Path) -> MobilityConfig:
    """Load mobility inputs from their shared or producing-stage declarations.

    :param path: Pipeline TOML file.
    :return: Mobility configuration.
    :raises KeyError: If required dependencies, datasets, or outputs are absent.
    :raises TypeError: If a setting has the wrong TOML type.
    :raises ValueError: If obsolete mobility-owned input paths are present.
    """
    raw = _read_config(path)
    section = _table(raw, "mobility", required=True)
    data = _table(raw, "data", required=False)
    duplicate_path_keys = {
        "author_file_path",
        "researcher_metadata_file_path",
        "article_author_file_path",
        "article_reference_file_path",
        "author_affiliations_file_path",
        "mobility_results_path",
    }
    duplicates = sorted(duplicate_path_keys.intersection(section))
    if duplicates:
        raise ValueError(
            "Mobility paths must use their authoritative [data] or producing-stage "
            "declarations; remove [mobility]." + ", [mobility].".join(duplicates)
        )

    matching = load_name_matching_config(path)
    disambiguation_section = _table(raw, "name_disambiguation", required=True)
    _reject_stage_path_duplicates(
        disambiguation_section,
        "name_disambiguation",
        (
            "author_file_path",
            "article_author_file_path",
            "article_reference_file_path",
            "disambiguated_author_path",
        ),
    )
    disambiguation_outputs = _load_disambiguation_outputs(disambiguation_section)
    authors = disambiguation_outputs.final_author_selection
    articles = _data_path(data, "article_author_file_path", required=True)
    assert articles is not None
    gpkg = _data_path(data, "author_affiliations_file_path", required=True)
    assert gpkg is not None
    detect = section.get("enable_university_affiliation_inbreeding_detection", False)
    merge_professors = section.get("enable_coaffiliated_professor_merge", False)
    title_pattern = section.get("professor_title_pattern", "prof")
    method = section.get("early_affiliation_method", "first_n_publications")
    if not isinstance(detect, bool) or not isinstance(merge_professors, bool):
        raise TypeError("Mobility detection and professor-merge options must be TOML booleans")
    if not isinstance(title_pattern, str) or not isinstance(method, str):
        raise TypeError("Mobility title pattern and early-affiliation method must be strings")
    outputs_section = section.get("outputs")
    if not isinstance(outputs_section, dict):
        raise KeyError("Missing required [mobility.outputs] table")
    outputs = _load_mobility_outputs(outputs_section)
    min_history = section.get("min_publication_history_years")
    if min_history is not None and (
        not isinstance(min_history, int) or isinstance(min_history, bool) or min_history < 0
    ):
        raise ValueError("min_publication_history_years must be a nonnegative integer or null")
    return MobilityConfig(
        authors=authors,
        researcher_metadata=matching.resolved_matches_output_file_path,
        article_author=articles,
        affiliations=gpkg,
        outputs=outputs,
        enable_university_affiliation_inbreeding_detection=detect,
        enable_coaffiliated_professor_merge=merge_professors,
        professor_title_pattern=title_pattern,
        first_affiliation_publication_count=_positive_integer(
            section, "first_affiliation_publication_count", 3
        ),
        early_affiliation_method=method,
        first_affiliation_year_count=_positive_integer(section, "first_affiliation_year_count", 2),
        min_publication_history_years=min_history,
        max_workers=_positive_integer(section, "max_workers", 3),
    )


def load_regression_data(config: RegressionConfig) -> LoadedRegressionInputs:
    """Load the producer-owned regression input artifact for downstream consumers.

    :param config: Resolved regression stage configuration.
    :return: Researcher-level regression input table.
    :raises FileNotFoundError: If the regression artifact is absent.
    """
    return config.load_inputs()


def _load_mobility_outputs(section: Mapping[str, Any]) -> MobilityOutputs:
    """Resolve the four explicitly owned mobility output artifacts.

    :param section: The ``[mobility.outputs]`` TOML table.
    :return: Resolved mobility output paths.
    :raises KeyError: If an output declaration is missing.
    :raises TypeError: If an output path is not a string.
    """
    output_keys = (
        "author_affiliations",
        "author_affiliations_unique",
        "author_inbreeding_proxy",
        "author_articles",
    )
    missing = [key for key in output_keys if key not in section]
    if missing:
        raise KeyError("Missing [mobility.outputs] declarations: " + ", ".join(missing))
    unexpected = set(section) - set(output_keys)
    if unexpected:
        raise ValueError("Unrecognized [mobility.outputs] keys: " + ", ".join(sorted(unexpected)))
    paths: dict[str, Path] = {}
    for key in output_keys:
        value = section[key]
        if not isinstance(value, str):
            raise TypeError(f"[mobility.outputs].{key} must be a path string")
        paths[key] = resolve_config_path(value)
    return MobilityOutputs(**paths)


def load_regression_config(path: str | Path) -> RegressionConfig:
    """Load regression paths through canonical data and mobility ownership.

    :param path: Shared pipeline TOML file.
    :return: Regression inputs, scientific options, and explicit output paths.
    :raises KeyError: If required data, mobility outputs, or regression outputs are absent.
    :raises TypeError: If a path or setting has the wrong TOML type.
    :raises ValueError: If redundant paths or invalid model settings are present.
    """
    raw = _read_config(path)
    section = _table(raw, "regression", required=True)
    _reject_stage_path_duplicates(
        section,
        "regression",
        (
            "author_article_file_path",
            "author_file_path",
            "article_impact_file_path",
            "disruption_indicator_file_path",
            "output_dir",
        ),
    )
    data = _table(raw, "data", required=False)
    mobility_section = _table(raw, "mobility", required=True)
    mobility_outputs_section = mobility_section.get("outputs")
    if not isinstance(mobility_outputs_section, dict):
        raise KeyError("Regression requires [mobility.outputs].author_inbreeding_proxy")
    mobility_outputs = _load_mobility_outputs(mobility_outputs_section)
    article_author = _data_path(data, "article_author_file_path", required=True)
    assert article_author is not None
    article_impact = _data_path(data, "article_impact_file_path", required=False)
    disruption_indicators = _data_path(data, "disruption_indicator_file_path", required=False)

    outputs_section = section.get("outputs")
    if not isinstance(outputs_section, dict):
        raise KeyError("Missing required [regression.outputs] table")
    outputs = _load_regression_outputs(outputs_section)
    dependent_variables = _load_regression_dependent_variables(section)
    outcome_metadata = _load_regression_outcome_metadata(section)
    missing_metadata = set(dependent_variables) - outcome_metadata.keys()
    if missing_metadata:
        raise KeyError(
            "Missing [regression.outcome_metadata] for dependent variables: "
            + ", ".join(sorted(missing_metadata))
        )
    model_specs = _load_regression_model_specs(section)
    inflation_formulas = section.get("zinb_inflation_formulas", {})
    if not isinstance(inflation_formulas, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in inflation_formulas.items()
    ):
        raise TypeError("regression.zinb_inflation_formulas must map outcome names to formulas")
    title_pattern = section.get("professor_title_pattern", "prof")
    if not isinstance(title_pattern, str) or not title_pattern:
        raise TypeError("regression.professor_title_pattern must be a nonempty string")
    excluded_titles = section.get("exclude_titles", ["FH-Ass. Prof."])
    if not isinstance(excluded_titles, list) or any(
        not isinstance(title, str) for title in excluded_titles
    ):
        raise TypeError("regression.exclude_titles must be an array of strings")
    rebuild_input = section.get("rebuild_input", True)
    print_summaries = section.get("print_model_summaries", False)
    drop_missing = section.get("drop_missing", True)
    if not all(isinstance(value, bool) for value in (rebuild_input, print_summaries, drop_missing)):
        raise TypeError("Regression execution flags must be TOML booleans")
    failure_policy = section.get("mle_failure_policy", "error")
    if failure_policy not in {"error", "warn"}:
        raise ValueError("regression.mle_failure_policy must be 'error' or 'warn'")
    covariance = section.get("covariance_type")
    if covariance is not None and not isinstance(covariance, str):
        raise TypeError("regression.covariance_type must be a string or null")
    return RegressionConfig(
        article_author_file_path=article_author,
        mobility_author_proxy=mobility_outputs.author_inbreeding_proxy,
        outputs=outputs,
        article_impact_file_path=article_impact,
        disruption_indicator_file_path=disruption_indicators,
        dependent_variables=dependent_variables,
        outcome_metadata=outcome_metadata,
        model_specs=model_specs,
        zinb_inflation_formulas=dict(inflation_formulas),
        zinb_regularization=_nonnegative_number(section, "zinb_regularization", 3.0),
        zinb_fit_method=_nonempty_string(section, "zinb_fit_method", "bfgs"),
        zinb_max_iterations=_positive_integer(section, "zinb_max_iterations", 1000),
        zinb_dispersion_power=_positive_integer(section, "zinb_dispersion_power", 2),
        mle_failure_policy=failure_policy,
        mle_gradient_tolerance=_positive_number(section, "mle_gradient_tolerance", 1e-4),
        covariance_type=covariance,
        drop_missing=drop_missing,
        min_publications=_positive_integer(section, "min_publications", 4),
        min_first_year=_optional_nonnegative_integer(section, "min_first_year"),
        professor_title_pattern=title_pattern,
        exclude_titles=tuple(excluded_titles),
        rebuild_input=rebuild_input,
        print_model_summaries=print_summaries,
    )


def _load_regression_outputs(section: Mapping[str, Any]) -> RegressionOutputs:
    """Resolve fixed regression artifacts and collection directories.

    :param section: The ``[regression.outputs]`` TOML table.
    :return: Explicit regression output paths.
    :raises KeyError: If a required artifact path is missing.
    :raises TypeError: If an artifact path is not a string.
    :raises ValueError: If unrecognized output keys are configured.
    """
    output_keys = (
        "regression_input",
        "professor_article_input",
        "professor_article_diagnostics",
        "asjc_field_share_diagnostics",
        "regression_input_diagnostics",
        "model_coefficients",
        "model_fit_statistics",
        "dependent_variable_descriptives",
        "aic_bic_comparison",
        "aic_bic_ranked",
        "best_models_by_aic",
        "best_models_by_bic",
        "inbreeding_discipline_spec_comparison",
        "model_formulas",
        "model_summaries",
        "latex_dir",
        "plots_dir",
    )
    missing = [key for key in output_keys if key not in section]
    if missing:
        raise KeyError("Missing [regression.outputs] declarations: " + ", ".join(missing))
    unexpected = set(section) - set(output_keys)
    if unexpected:
        raise ValueError("Unrecognized [regression.outputs] keys: " + ", ".join(sorted(unexpected)))
    paths: dict[str, Path] = {}
    for key in output_keys:
        value = section[key]
        if not isinstance(value, str):
            raise TypeError(f"[regression.outputs].{key} must be a path string")
        paths[key] = resolve_config_path(value)
    return RegressionOutputs(**paths)


def _load_regression_dependent_variables(
    section: Mapping[str, Any],
) -> dict[str, RegressionModelType]:
    """Load outcome-to-model-family settings without consulting downstream config.

    :param section: The ``[regression]`` TOML table.
    :return: Dependent variable model families.
    :raises TypeError: If outcome families are not configured as a string table.
    :raises ValueError: If any model family is unsupported.
    """
    values = section.get("dependent_variables", {})
    if not isinstance(values, dict) or any(
        not isinstance(outcome, str) or not isinstance(model, str)
        for outcome, model in values.items()
    ):
        raise TypeError("regression.dependent_variables must map outcomes to model families")
    allowed = {"ols", "pr", "nb", "zinb"}
    invalid = {model for model in values.values() if model not in allowed}
    if invalid:
        raise ValueError("Unsupported regression model family: " + ", ".join(sorted(invalid)))
    return {outcome: cast(RegressionModelType, model) for outcome, model in values.items()}


def _load_regression_outcome_metadata(
    section: Mapping[str, Any],
) -> dict[str, RegressionOutcomeMetadata]:
    """Load regression-owned outcome groupings and labels.

    :param section: The ``[regression]`` TOML table.
    :return: Outcome metadata keyed by dependent-variable name.
    :raises TypeError: If metadata values have invalid TOML types.
    :raises ValueError: If outcome groupings or aliases are invalid.
    """
    from acib.regression.constants import DEFAULT_REGRESSION_OUTCOME_METADATA

    raw = section.get("outcome_metadata")
    if raw is None:
        return {
            key: RegressionOutcomeMetadata(
                outcome_type=cast(RegressionOutcomeType, value[0]), alias=value[1]
            )
            for key, value in DEFAULT_REGRESSION_OUTCOME_METADATA.items()
        }
    if not isinstance(raw, dict):
        raise TypeError("[regression.outcome_metadata] must be a table")
    metadata: dict[str, RegressionOutcomeMetadata] = {}
    for outcome, value in raw.items():
        if not isinstance(value, dict):
            raise TypeError(f"[regression.outcome_metadata.{outcome}] must be a table")
        outcome_type = value.get("outcome_type")
        alias = value.get("alias")
        if outcome_type not in {"relative", "absolute"}:
            raise ValueError(
                f"[regression.outcome_metadata.{outcome}].outcome_type must be "
                "'relative' or 'absolute'"
            )
        if not isinstance(alias, str) or not alias:
            raise TypeError(
                f"[regression.outcome_metadata.{outcome}].alias must be a " "nonempty string"
            )
        if set(value) != {"outcome_type", "alias"}:
            raise ValueError(f"Unexpected keys in [regression.outcome_metadata.{outcome}]")
        metadata[outcome] = RegressionOutcomeMetadata(
            outcome_type=cast(RegressionOutcomeType, outcome_type),
            alias=alias,
        )
    return metadata


def _load_regression_model_specs(
    section: Mapping[str, Any],
) -> tuple[RegressionModelConfig, ...] | None:
    """Parse optional custom model specifications.

    :param section: The ``[regression]`` TOML table.
    :return: Ordered custom specs, or ``None`` to use the ten source defaults.
    :raises TypeError: If model specs have invalid TOML types.
    :raises ValueError: If model specs are empty or identifiers repeat.
    """
    raw = section.get("model_specs")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TypeError("regression.model_specs must be an array of tables")
    specs: list[RegressionModelConfig] = []
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise TypeError(f"regression.model_specs[{index}] must be a table")
        model_id = value.get("model_id")
        terms = value.get("terms")
        if not isinstance(model_id, str) or not model_id:
            raise TypeError(f"regression.model_specs[{index}].model_id must be a nonempty string")
        if (
            not isinstance(terms, list)
            or not terms
            or any(not isinstance(term, str) for term in terms)
        ):
            raise TypeError(
                f"regression.model_specs[{index}].terms must be a nonempty string array"
            )
        if set(value) != {"model_id", "terms"}:
            raise ValueError(f"Unexpected keys in regression.model_specs[{index}]")
        specs.append(RegressionModelConfig(model_id=model_id, terms=tuple(terms)))
    identifiers = [spec.model_id for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("regression.model_specs model_id values must be unique")
    return tuple(specs)


def _nonempty_string(section: Mapping[str, Any], key: str, default: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"regression.{key} must be a nonempty string")
    return value


def _positive_number(section: Mapping[str, Any], key: str, default: float) -> float:
    value = _nonnegative_number(section, key, default)
    if value <= 0:
        raise ValueError(f"regression.{key} must be positive")
    return value


def _optional_nonnegative_integer(section: Mapping[str, Any], key: str) -> int | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"regression.{key} must be a nonnegative integer or null")
    return value
