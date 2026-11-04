# Copyright © 2026 TU Wien

"""Typed inputs and results for name matching."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from acib.name_matching.matching import ResearcherMatchResult


@dataclass(frozen=True, slots=True)
class NameMatchingInputs:
    """Loaded in-memory inputs for researcher-to-author matching.

    :param researchers: Researcher records used for matching.
    :param candidate_authors: Scopus-author candidate records.
    :param researcher_output_metadata: Optional source-spelling metadata frame used
        to preserve the existing resolved-artifact serialization.
    """

    researchers: pd.DataFrame
    candidate_authors: pd.DataFrame
    researcher_output_metadata: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class NameMatchingResult:
    """Complete analytical output of the name-matching stage.

    :param resolved_researchers: Validated researcher-to-author match records.
    :param matching_details: Matched, possible, and unmatched algorithm rows, or
        ``None`` when no rows required matching.
    """

    resolved_researchers: pd.DataFrame
    matching_details: ResearcherMatchResult | None = None
