# Copyright © 2026 TU Wien

"""Researcher-to-author name matching stage."""

from importlib import import_module
from typing import Any

from .matching import ResearcherMatchResult, match_researchers, simple_name_merge

_LAZY_EXPORTS = {
    "NameMatchingInputs": (".results", "NameMatchingInputs"),
    "NameMatchingResult": (".results", "NameMatchingResult"),
    "compute_name_matching": (".workflow", "compute_name_matching"),
    "run_name_matching": (".workflow", "run_name_matching"),
    "save_name_matching_results": (".output", "save_name_matching_results"),
}

__all__ = [
    "NameMatchingInputs",
    "NameMatchingResult",
    "ResearcherMatchResult",
    "compute_name_matching",
    "match_researchers",
    "run_name_matching",
    "save_name_matching_results",
    "simple_name_merge",
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
