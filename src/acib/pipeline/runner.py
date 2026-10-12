# Copyright © 2026 TU Wien
"""Minimal runner for name matching followed by name disambiguation."""
from __future__ import annotations

from pathlib import Path

from acib.name_disambiguation.config import load_name_disambiguation_config
from acib.name_disambiguation.workflow import validate_matching_artifact
from acib.name_matching.config import load_name_matching_config
from acib.name_matching.workflow import run_name_matching


def run_pipeline(config_path: str | Path, stage: str = "name_disambiguation") -> None:
    """Run the selected stage, using existing matcher output when valid."""
    if stage != "name_disambiguation":
        raise ValueError("Only stage 'name_disambiguation' is supported")
    matching = load_name_matching_config(config_path)
    disambiguation = load_name_disambiguation_config(config_path)
    upstream = matching.resolved_matches_output_file_path
    if upstream.exists():
        validate_matching_artifact(upstream)
    else:
        run_name_matching(matching)
        validate_matching_artifact(upstream)
    from acib.name_disambiguation.workflow import run_name_disambiguation

    run_name_disambiguation(disambiguation)
