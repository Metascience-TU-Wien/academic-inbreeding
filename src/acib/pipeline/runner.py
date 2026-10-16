# Copyright © 2026 TU Wien
"""Runner for the configured name-matching and disambiguation sequence."""
from __future__ import annotations

from pathlib import Path

from acib.name_disambiguation.config import load_name_disambiguation_config
from acib.name_disambiguation.workflow import run_name_disambiguation
from acib.pipeline.dependencies import ensure_stage_dependencies


def run_pipeline(config_path: str | Path, stage: str = "name_disambiguation") -> None:
    """Run the selected stage after ensuring its generated dependencies.

    :param config_path: Shared pipeline TOML file.
    :param stage: The only supported stage, ``name_disambiguation``.
    :raises ValueError: If another stage is requested.
    """
    if stage != "name_disambiguation":
        raise ValueError("Only stage 'name_disambiguation' is supported")
    ensure_stage_dependencies(stage, config_path)
    disambiguation = load_name_disambiguation_config(config_path)
    run_name_disambiguation(disambiguation)
