# Copyright © 2026 TU Wien

from pathlib import Path

import pytest

from acib.name_disambiguation import cli
from acib.name_matching.config import PROJECT_ROOT


def test_cli_defaults_to_project_pipeline_config(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []
    ensured: list[tuple[str, Path, bool]] = []
    marker = object()

    def load_config(path: str | Path) -> object:
        seen.append(Path(path))
        return marker

    def run(config: object, *, overwrite: bool = False) -> None:
        assert config is marker
        assert overwrite is False

    monkeypatch.setattr("sys.argv", ["acib-name-disambiguation"])
    monkeypatch.setattr(
        cli,
        "ensure_stage_dependencies",
        lambda stage, path, *, overwrite=False: ensured.append((stage, Path(path), overwrite)),
    )
    monkeypatch.setattr(cli, "load_name_disambiguation_config", load_config)
    monkeypatch.setattr(cli, "run_name_disambiguation", run)
    cli.main()
    assert seen == [PROJECT_ROOT / "config/pipeline.toml"]
    assert ensured == [("name_disambiguation", PROJECT_ROOT / "config/pipeline.toml", False)]
