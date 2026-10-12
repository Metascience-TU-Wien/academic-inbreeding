# Copyright © 2026 TU Wien
"""CLI for the minimal two-stage pipeline runner."""
from __future__ import annotations

import argparse
from pathlib import Path

from acib.name_matching.config import PROJECT_ROOT
from acib.pipeline.runner import run_pipeline


def main() -> None:
    """Run name matching when needed, then disambiguation."""
    parser = argparse.ArgumentParser(description="Run the name disambiguation pipeline.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/pipeline.toml")
    parser.add_argument("--stage", default="name_disambiguation", choices=["name_disambiguation"])
    args = parser.parse_args()
    run_pipeline(args.config, args.stage)
