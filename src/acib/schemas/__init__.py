# Copyright © 2026 TU Wien

"""Reusable DataFrame contracts for the academic-inbreeding pipeline."""

from acib.schemas.affiliations import AuthorAffiliationSchema
from acib.schemas.authors import ScopusAuthorInputSchema
from acib.schemas.matching import MatchingOutputSchema
from acib.schemas.researchers import ResearcherInputSchema

__all__ = [
    "AuthorAffiliationSchema",
    "MatchingOutputSchema",
    "ResearcherInputSchema",
    "ScopusAuthorInputSchema",
]
