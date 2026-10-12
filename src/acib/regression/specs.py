# Copyright © 2026 TU Wien

"""Canonical regression model specifications and formula expansion."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from acib.regression.constants import (
    BROAD_FIELD_SHARE_TERMS,
    FIELD_SHARE_TERMS,
    PAPER_REGRESSION_MODEL_TERMS,
)
from acib.regression.fields import (
    included_domain_share_terms,
    included_field_share_terms,
    omitted_reference_domain,
    omitted_reference_field,
)


@dataclass(frozen=True, slots=True)
class RegressionModelSpec:
    """Represent one named model and its ordered right-hand-side terms.

    :param model_id: Stable model identifier.
    :param terms: Ordered formula terms, including optional share-term markers.
    """

    model_id: str
    terms: tuple[str, ...]

    def formula(self, dependent_variable: str, *, data: pd.DataFrame | None = None) -> str:
        """Build a statsmodels formula for this specification.

        :param dependent_variable: Dependent variable placed on the left-hand side.
        :param data: Data used to resolve share controls and their reference category.
        :return: Complete statsmodels formula string.
        """
        return f"{dependent_variable} ~ {' + '.join(self.expanded_terms(data=data))}"

    def expanded_terms(self, *, data: pd.DataFrame | None = None) -> tuple[str, ...]:
        """Expand dynamic field-share markers into included share columns.

        :param data: Data used to resolve share controls.
        :return: Ordered formula terms.
        :raises ValueError: If share terms are requested without data.
        """
        if not self.requires_share_terms:
            return self.terms
        if data is None:
            raise ValueError("Share model terms require regression data.")
        terms: list[str] = []
        field_reference = omitted_reference_field(data)
        domain_reference = omitted_reference_domain(data)
        for term in self.terms:
            if term == FIELD_SHARE_TERMS:
                terms.extend(included_field_share_terms(data, reference_field=field_reference))
            elif term == BROAD_FIELD_SHARE_TERMS:
                terms.extend(included_domain_share_terms(data, reference_domain=domain_reference))
            else:
                terms.append(term)
        return tuple(terms)

    @property
    def requires_share_terms(self) -> bool:
        """Return whether this specification contains dynamic share controls."""
        return bool({FIELD_SHARE_TERMS, BROAD_FIELD_SHARE_TERMS}.intersection(self.terms))

    def field_share_reference(self, *, data: pd.DataFrame) -> str | None:
        """Return the omitted share reference used by this specification.

        :param data: Regression data containing share columns.
        :return: Omitted detailed/domain share field, when applicable.
        """
        if FIELD_SHARE_TERMS in self.terms:
            return omitted_reference_field(data)
        if BROAD_FIELD_SHARE_TERMS in self.terms:
            return omitted_reference_domain(data)
        return None

    def share_control_type(self) -> str | None:
        """Return the label identifying this specification's share controls."""
        if FIELD_SHARE_TERMS in self.terms:
            return "detailed_asjc_shares"
        if BROAD_FIELD_SHARE_TERMS in self.terms:
            return "broad_asjc_shares"
        return None


def regression_model_specs() -> list[RegressionModelSpec]:
    """Return the ten ordered paper regression model specifications.

    :return: Default regression model specifications in m1 through m10 order.
    """
    return [
        RegressionModelSpec(model_id=model_id, terms=terms)
        for model_id, terms in PAPER_REGRESSION_MODEL_TERMS.items()
    ]
