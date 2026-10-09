# Copyright © 2026 TU Wien

"""Researcher-to-author name matching stage."""

from .matching import ResearcherMatchResult, match_researchers, simple_name_merge

__all__ = ["ResearcherMatchResult", "match_researchers", "simple_name_merge"]
