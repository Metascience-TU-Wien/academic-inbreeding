# Copyright © 2026 TU Wien

import json
from pathlib import Path
from typing import Any

import pytest

from acib.pipeline.dependencies import ensure_stage_dependencies


def test_missing_mobility_dependencies_run_configured_producers_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build missing matcher, selection, and mobility artifacts on demand."""
    external = {
        "researchers": tmp_path / "researchers.csv",
        "candidates": tmp_path / "candidates.csv",
        "articles": tmp_path / "articles.csv",
        "references": tmp_path / "references.csv",
        "affiliations": tmp_path / "affiliations.gpkg",
    }
    for path in external.values():
        path.touch()

    matching_dir = tmp_path / "matching"
    disambiguation_dir = tmp_path / "disambiguation"
    mobility_dir = tmp_path / "mobility"
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        "\n".join(
            (
                "[data]",
                f"researcher_file_path = {json.dumps(str(external['researchers']))}",
                f"candidate_authors_file = {json.dumps(str(external['candidates']))}",
                f"article_author_file_path = {json.dumps(str(external['articles']))}",
                f"article_reference_file_path = {json.dumps(str(external['references']))}",
                f"author_affiliations_file_path = {json.dumps(str(external['affiliations']))}",
                "",
                "[name_matching]",
                "[name_matching.outputs]",
                f"directory = {json.dumps(str(matching_dir))}",
                "",
                "[name_disambiguation]",
                "[name_disambiguation.outputs]",
                f"directory = {json.dumps(str(disambiguation_dir))}",
                "",
                "[mobility]",
                "[mobility.outputs]",
                f"directory = {json.dumps(str(mobility_dir))}",
                "",
            )
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    def run_matching(config: Any) -> None:
        calls.append("name_matching")
        config.outputs.resolved_researchers.parent.mkdir(parents=True, exist_ok=True)
        config.outputs.resolved_researchers.touch()

    def run_disambiguation(config: Any, *, overwrite: bool = False) -> None:
        del overwrite
        calls.append("name_disambiguation")
        config.outputs.final_author_selection.parent.mkdir(parents=True, exist_ok=True)
        config.outputs.final_author_selection.touch()

    def run_mobility(config: Any) -> None:
        calls.append("mobility")
        config.outputs.author_inbreeding_proxy.parent.mkdir(parents=True, exist_ok=True)
        config.outputs.author_inbreeding_proxy.touch()

    monkeypatch.setattr("acib.name_matching.workflow.run_name_matching", run_matching)
    monkeypatch.setattr(
        "acib.name_disambiguation.workflow.run_name_disambiguation", run_disambiguation
    )
    monkeypatch.setattr("acib.mobility.workflow.run_mobility", run_mobility)

    ensure_stage_dependencies("mobility", config_path)

    assert calls == ["name_matching", "name_disambiguation", "mobility"]
