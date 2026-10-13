# Copyright © 2026 TU Wien
"""Command-line interface for name disambiguation."""
from __future__ import annotations

import argparse
from pathlib import Path

from acib.name_disambiguation.config import load_name_disambiguation_config
from acib.name_disambiguation.workflow import run_name_disambiguation
from acib.name_matching.config import PROJECT_ROOT
from acib.pipeline.dependencies import ensure_stage_dependencies


def main() -> None:
    """Run disambiguation after ensuring its configured matcher artifact exists."""
    parser = argparse.ArgumentParser(description="Disambiguate researcher-author matches.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/pipeline.toml")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the configured disambiguation outputs if they already exist.",
    )
    args = parser.parse_args()
    ensure_stage_dependencies("name_disambiguation", args.config, overwrite=args.overwrite)
    run_name_disambiguation(load_name_disambiguation_config(args.config), overwrite=args.overwrite)
