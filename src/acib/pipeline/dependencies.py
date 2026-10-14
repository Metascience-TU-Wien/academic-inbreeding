# Copyright © 2026 TU Wien

"""Ensure configured producer artifacts exist before running a pipeline stage."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import structlog

from acib.pipeline.config import (
    load_mobility_config,
    load_name_disambiguation_config,
    load_name_matching_config,
    load_regression_config,
    load_statistical_matching_config,
)

_logger = structlog.get_logger(__name__)


def ensure_stage_dependencies(
    stage: str,
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Run producers for missing generated inputs of the requested stage.

    Existing artifacts are reused; freshness is not checked. Missing external
    datasets are left to the requested stage's normal input validation.

    :param stage: Configured stage to prepare: matching, disambiguation, mobility,
        regression, or statistical matching.
    :param config_path: Shared pipeline TOML file.
    :param overwrite: Allow rebuilding partial name-disambiguation outputs.
    :raises ValueError: If the requested stage name is unsupported.
    :raises FileNotFoundError: If a producer fails to create its required artifact.
    :raises FileExistsError: If producer outputs are partial and replacement was
        not explicitly permitted.
    """
    if stage == "name_matching":
        return
    if stage == "name_disambiguation":
        _ensure_matching(config_path)
        return
    if stage == "mobility":
        _ensure_mobility(config_path)
        return
    if stage == "regression":
        _ensure_regression(config_path)
        return
    if stage == "statistical_matching":
        _ensure_statistical_matching(config_path)
        return
    raise ValueError(f"Unsupported pipeline stage: {stage}")


def _ensure_matching(config_path: str | Path) -> None:
    """Run matching only when its configured output artifact is absent."""
    from acib.name_matching.workflow import run_name_matching

    config = load_name_matching_config(config_path)
    _ensure_artifact(
        stage="name_matching",
        artifact=config.outputs.resolved_researchers,
        outputs=(config.outputs.resolved_researchers,),
        run=lambda: run_name_matching(config),
    )


def _ensure_disambiguation(config_path: str | Path, *, overwrite: bool = False) -> None:
    """Ensure disambiguation's matcher input, then create its required artifact."""
    from acib.name_disambiguation.workflow import run_name_disambiguation

    config = load_name_disambiguation_config(config_path)
    _require_external_inputs(
        (
            ("[data].article_author_file_path", config.article_author_file_path),
            ("[data].article_reference_file_path", config.article_reference_file_path),
        )
    )
    _check_partial_outputs(
        stage="name_disambiguation",
        artifact=config.outputs.final_author_selection,
        outputs=config.outputs.all_paths(),
        overwrite=overwrite,
    )
    _ensure_matching(config_path)
    _ensure_artifact(
        stage="name_disambiguation",
        artifact=config.outputs.final_author_selection,
        outputs=config.outputs.all_paths(),
        run=lambda: run_name_disambiguation(config, overwrite=overwrite),
        overwrite=overwrite,
    )


def _ensure_mobility(config_path: str | Path) -> None:
    """Ensure matching and disambiguation artifacts required by mobility."""
    from acib.mobility.workflow import run_mobility

    config = load_mobility_config(config_path)
    disambiguation = load_name_disambiguation_config(config_path)
    _require_external_inputs(
        (
            ("[data].article_author_file_path", config.article_author),
            ("[data].author_affiliations_file_path", config.affiliations),
            ("[data].article_reference_file_path", disambiguation.article_reference_file_path),
        )
    )
    _check_partial_outputs(
        stage="mobility",
        artifact=config.outputs.author_inbreeding_proxy,
        outputs=config.outputs.all_paths(),
    )
    _ensure_matching(config_path)
    _ensure_disambiguation(config_path)
    _ensure_artifact(
        stage="mobility",
        artifact=config.outputs.author_inbreeding_proxy,
        outputs=config.outputs.all_paths(),
        run=lambda: run_mobility(config),
    )


def _ensure_regression(config_path: str | Path) -> None:
    """Ensure regression's mobility input when configured to rebuild inputs."""
    config = load_regression_config(config_path)
    if not config.rebuild_input:
        # In reuse mode the regression stage requires its own existing input.
        config.check_regression_input_exists()
        return
    _require_external_inputs(
        (
            ("[data].article_author_file_path", config.article_author_file_path),
            ("[data].article_impact_file_path", config.article_impact_file_path),
            ("[data].disruption_indicator_file_path", config.disruption_indicator_file_path),
        )
    )
    _ensure_mobility(config_path)


def _ensure_statistical_matching(config_path: str | Path) -> None:
    """Build regression's required input artifact if statistical matching needs it."""
    from acib.regression.build import build_regression_inputs_from_config

    matching = load_statistical_matching_config(config_path)
    if matching.input_path.is_file():
        return

    regression = load_regression_config(config_path)
    _require_external_inputs(
        (
            ("[data].article_author_file_path", regression.article_author_file_path),
            ("[data].article_impact_file_path", regression.article_impact_file_path),
            ("[data].disruption_indicator_file_path", regression.disruption_indicator_file_path),
        )
    )
    build_outputs = (
        regression.outputs.regression_input,
        regression.outputs.professor_article_input,
        regression.outputs.professor_article_diagnostics,
        regression.outputs.asjc_field_share_diagnostics,
        regression.outputs.regression_input_diagnostics,
    )
    _check_partial_outputs(
        stage="regression input build",
        artifact=regression.outputs.regression_input,
        outputs=build_outputs,
    )
    _ensure_mobility(config_path)
    _ensure_artifact(
        stage="regression input build",
        artifact=regression.outputs.regression_input,
        outputs=build_outputs,
        run=lambda: build_regression_inputs_from_config(regression),
    )


def _require_external_inputs(inputs: Sequence[tuple[str, Path | None]]) -> None:
    """Fail early when configured external inputs are unavailable.

    :param inputs: Canonical TOML keys paired with their resolved paths.
    :raises FileNotFoundError: If any required external file is missing.
    """
    missing = [f"{key}: {path}" for key, path in inputs if path is not None and not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required external pipeline input(s):\n- " + "\n- ".join(missing)
        )


def _ensure_artifact(
    *,
    stage: str,
    artifact: Path,
    outputs: Sequence[Path],
    run: Callable[[], object],
    overwrite: bool = False,
) -> None:
    """Create one absent artifact without silently replacing partial outputs.

    :param stage: Producer stage label for logs and errors.
    :param artifact: Required producer-owned artifact.
    :param outputs: Artifacts that the producer may write.
    :param run: Callback that executes the producer.
    :param overwrite: Permit replacing existing outputs.
    :raises FileExistsError: If a partial output set would otherwise be replaced.
    :raises FileNotFoundError: If the producer finishes without its artifact.
    """
    if artifact.is_file():
        return
    _check_partial_outputs(
        stage=stage,
        artifact=artifact,
        outputs=outputs,
        overwrite=overwrite,
    )
    _logger.info(
        "Required upstream artifact is missing; starting its producer",
        producer_stage=stage,
        required_artifact=str(artifact),
    )
    run()
    if not artifact.is_file():
        raise FileNotFoundError(
            f"Producer stage {stage} completed without creating required artifact: {artifact}"
        )


def _check_partial_outputs(
    *,
    stage: str,
    artifact: Path,
    outputs: Sequence[Path],
    overwrite: bool = False,
) -> None:
    """Reject auto-runs that would replace a producer's partial output set.

    :param stage: Producer stage label for the error.
    :param artifact: Required producer-owned artifact.
    :param outputs: Artifacts that the producer may write.
    :param overwrite: Permit replacement of existing outputs.
    :raises FileExistsError: If existing outputs would be replaced implicitly.
    """
    if artifact.is_file() or overwrite:
        return
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            f"Cannot automatically run {stage}: required artifact is missing ({artifact}), "
            "but other producer outputs already exist. Review those outputs and explicitly "
            "rerun the producer with its overwrite option if replacement is intended."
        )
