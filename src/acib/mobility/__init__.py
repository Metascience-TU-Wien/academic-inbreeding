# Copyright © 2026 TU Wien

"""Standalone academic mobility and inbreeding analysis."""

from acib.mobility.config import MobilityConfig, MobilityOutputs
from acib.mobility.inbreeding import InbreedingResult
from acib.mobility.input import MobilityInputs, load_mobility_inputs
from acib.mobility.workflow import compute_mobility, run_mobility

__all__ = [
    "InbreedingResult",
    "MobilityConfig",
    "MobilityInputs",
    "MobilityOutputs",
    "compute_mobility",
    "load_mobility_inputs",
    "run_mobility",
]
