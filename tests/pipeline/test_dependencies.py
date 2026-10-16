# Copyright © 2026 TU Wien

"""Tests for automatic execution of missing configured dependencies."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from acib.pipeline import dependencies


def _outputs(primary: Path, *other_paths: Path) -> SimpleNamespace:
    """Create a lightweight producer output object for dependency tests."""
    paths = (primary, *other_paths)
    return SimpleNamespace(
        resolved_researchers=primary,
        final_author_selection=primary,
        author_inbreeding_proxy=primary,
        regression_input=primary,
        professor_article_input=other_paths[0] if other_paths else primary,
        professor_article_diagnostics=other_paths[1] if len(other_paths) > 1 else primary,
        asjc_field_share_diagnostics=other_paths[2] if len(other_paths) > 2 else primary,
        regression_input_diagnostics=other_paths[3] if len(other_paths) > 3 else primary,
        all_paths=lambda: paths,
    )


def test_disambiguation_runs_matching_only_when_match_artifact_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matching_artifact = tmp_path / "matching.csv"
    selection_artifact = tmp_path / "selection.csv"
    article_author = tmp_path / "articles.parquet"
    article_reference = tmp_path / "references.parquet"
    article_author.touch()
    article_reference.touch()
    matching_config = SimpleNamespace(outputs=_outputs(matching_artifact))
    disambiguation_config = SimpleNamespace(
        matching_output=matching_artifact,
        article_author_file_path=article_author,
        article_reference_file_path=article_reference,
        outputs=_outputs(selection_artifact),
        check_external_inputs_exist=lambda: None,
    )
    monkeypatch.setattr(dependencies, "load_name_matching_config", lambda _: matching_config)
    monkeypatch.setattr(
        dependencies, "load_name_disambiguation_config", lambda _: disambiguation_config
    )
    calls: list[str] = []

    def run_matching(config: object) -> None:
        calls.append("matching")
        matching_artifact.touch()

    def run_disambiguation(config: object, *, overwrite: bool = False) -> None:
        calls.append("disambiguation")
        selection_artifact.touch()

    monkeypatch.setattr("acib.name_matching.workflow.run_name_matching", run_matching)
    monkeypatch.setattr(
        "acib.name_disambiguation.workflow.run_name_disambiguation", run_disambiguation
    )

    dependencies.ensure_stage_dependencies("name_disambiguation", "unused.toml")

    assert calls == ["matching", "disambiguation"]


def test_existing_dependency_is_reused_without_running_producer(tmp_path: Path) -> None:
    artifact = tmp_path / "present.csv"
    artifact.touch()
    called = False

    def producer() -> None:
        nonlocal called
        called = True

    dependencies._ensure_artifact(
        stage="example",
        artifact=artifact,
        outputs=(artifact,),
        run=producer,
    )

    assert not called


def test_partial_producer_outputs_are_not_overwritten_automatically(tmp_path: Path) -> None:
    required = tmp_path / "required.csv"
    partial = tmp_path / "other-output.csv"
    partial.touch()

    with pytest.raises(FileExistsError, match="other producer outputs already exist"):
        dependencies._ensure_artifact(
            stage="example",
            artifact=required,
            outputs=(required, partial),
            run=lambda: None,
        )


def test_missing_external_input_fails_before_any_producer_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matching_config = SimpleNamespace(outputs=_outputs(tmp_path / "matches.csv"))
    disambiguation_config = SimpleNamespace(
        article_author_file_path=tmp_path / "absent.parquet",
        article_reference_file_path=tmp_path / "references.parquet",
        check_external_inputs_exist=lambda: None,
        outputs=_outputs(tmp_path / "selection.csv"),
    )
    monkeypatch.setattr(dependencies, "load_name_matching_config", lambda _: matching_config)
    monkeypatch.setattr(
        dependencies, "load_name_disambiguation_config", lambda _: disambiguation_config
    )
    monkeypatch.setattr(
        "acib.name_matching.workflow.run_name_matching",
        lambda config: pytest.fail("matching must not run when external data is absent"),
    )

    with pytest.raises(FileNotFoundError, match="external pipeline input"):
        dependencies.ensure_stage_dependencies("name_disambiguation", "unused.toml")


def test_regression_ensures_mobility_only_when_rebuilding_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    article_author = tmp_path / "articles.parquet"
    article_author.touch()
    config = SimpleNamespace(
        rebuild_input=True,
        article_author_file_path=article_author,
        article_impact_file_path=None,
        disruption_indicator_file_path=None,
    )
    calls: list[str] = []
    monkeypatch.setattr(dependencies, "load_regression_config", lambda _: config)
    monkeypatch.setattr(dependencies, "_ensure_mobility", lambda _: calls.append("mobility"))

    dependencies.ensure_stage_dependencies("regression", "unused.toml")

    assert calls == ["mobility"]


def test_statistical_matching_builds_missing_regression_input_without_model_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    article_author = tmp_path / "articles.parquet"
    article_author.touch()
    regression_input = tmp_path / "regression-input.csv"
    professor_input = tmp_path / "professor-input.parquet"
    diagnostics = tuple(tmp_path / f"diagnostic-{index}.csv" for index in range(3))
    regression_outputs = SimpleNamespace(
        regression_input=regression_input,
        professor_article_input=professor_input,
        professor_article_diagnostics=diagnostics[0],
        asjc_field_share_diagnostics=diagnostics[1],
        regression_input_diagnostics=diagnostics[2],
    )
    matching_config = SimpleNamespace(input_path=regression_input)
    regression_config = SimpleNamespace(
        article_author_file_path=article_author,
        article_impact_file_path=None,
        disruption_indicator_file_path=None,
        outputs=regression_outputs,
    )
    calls: list[str] = []
    monkeypatch.setattr(dependencies, "load_statistical_matching_config", lambda _: matching_config)
    monkeypatch.setattr(dependencies, "load_regression_config", lambda _: regression_config)
    monkeypatch.setattr(dependencies, "_ensure_mobility", lambda _: calls.append("mobility"))

    def build_regression_input(config: object) -> None:
        calls.append("regression-input-build")
        regression_input.touch()

    monkeypatch.setattr(
        "acib.regression.build.build_regression_inputs_from_config", build_regression_input
    )

    dependencies.ensure_stage_dependencies("statistical_matching", "unused.toml")

    assert calls == ["mobility", "regression-input-build"]
