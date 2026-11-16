# Copyright © 2026 TU Wien

from pathlib import Path

import pytest

from acib.mobility import cli
from acib.mobility.config import MobilityConfig, MobilityOutputs


def test_mobility_cli_explicitly_prepares_dependencies_then_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoke dependency preparation before the configured mobility runner."""
    config_path = tmp_path / "pipeline.toml"
    config = MobilityConfig(
        authors=tmp_path / "selection.csv",
        researcher_metadata=tmp_path / "matching.csv",
        article_author=tmp_path / "articles.csv",
        affiliations=tmp_path / "affiliations.gpkg",
        outputs=MobilityOutputs(tmp_path / "outputs"),
    )
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        cli,
        "ensure_stage_dependencies",
        lambda stage, path: calls.append(("dependencies", (stage, path))),
    )
    monkeypatch.setattr(cli, "load_mobility_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "run_mobility",
        lambda value: calls.append(("run", value)),
    )

    cli.main(["--config", str(config_path)])

    assert calls == [
        ("dependencies", ("mobility", config_path)),
        ("run", config),
    ]
