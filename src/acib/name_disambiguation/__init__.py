# Copyright © 2026 TU Wien
"""Researcher-to-author name disambiguation stage."""

from importlib import import_module
from typing import Any

from .config import NameDisambiguationConfig, load_name_disambiguation_config

_LAZY_EXPORTS = {
    "NameDisambiguationInputs": (".results", "NameDisambiguationInputs"),
    "NameDisambiguationResult": (".results", "NameDisambiguationResult"),
    "compute_name_disambiguation": (".workflow", "compute_name_disambiguation"),
    "load_name_disambiguation_inputs": (".input", "load_name_disambiguation_inputs"),
    "prepare_name_disambiguation_inputs": (".input", "prepare_name_disambiguation_inputs"),
    "run_name_disambiguation": (".workflow", "run_name_disambiguation"),
    "save_name_disambiguation_results": (".output", "save_name_disambiguation_results"),
}

__all__ = [
    "NameDisambiguationConfig",
    "NameDisambiguationInputs",
    "NameDisambiguationResult",
    "compute_name_disambiguation",
    "load_name_disambiguation_config",
    "load_name_disambiguation_inputs",
    "prepare_name_disambiguation_inputs",
    "run_name_disambiguation",
    "save_name_disambiguation_results",
]


def __getattr__(name: str) -> Any:
    """Load stage API exports lazily to avoid schema import cycles.

    :param name: Requested package attribute.
    :return: The requested public API object.
    :raises AttributeError: If the package does not export the requested name.
    """
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    return getattr(module, attribute_name)
