# Copyright © 2026 TU Wien

"""Standalone academic mobility and inbreeding analysis."""

from acib.mobility.config import MobilityConfig, MobilityOutputs
from acib.mobility.workflow import run_mobility

__all__ = ["MobilityConfig", "MobilityOutputs", "run_mobility"]
