# Copyright © 2026 TU Wien

"""Persistence for statistical-matching tables and reports."""

from __future__ import annotations

from acib.constants import IS_INBRED
from acib.statistical_matching.config import StatisticalMatchingOutputs
from acib.statistical_matching.schemas import validate_output
from acib.statistical_matching.suite import StatisticalMatchingResult


def write_outputs(
    result: StatisticalMatchingResult,
    outputs: StatisticalMatchingOutputs,
    *,
    treatment_variable: str = IS_INBRED,
) -> None:
    """Write configured tables, formulas, and family LaTeX reports.

    :param result: Matching results to persist.
    :param outputs: Destinations for generated artifacts.
    :param treatment_variable: Treatment column name in score tables.
    """
    frames = {
        "balance_summary": result.balance_summary,
        "covariate_balance": result.covariate_balance,
        "outcome_effects_long": result.outcome_effects,
        "adjusted_outcome_means": result.adjusted_means,
        "outcome_effects_wide": result.effect_wide,
        "outcome_standard_errors_wide": result.se_wide,
        "method_diagnostics": result.diagnostics,
        "propensity_scores": result.propensity_scores,
        "propensity_scores_by_method": result.propensity_scores_by_method,
    }
    for name, frame in frames.items():
        validated = validate_output(name, frame, treatment_variable=treatment_variable)
        path = getattr(outputs, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        validated.to_csv(path, index=False)
    outputs.matching_formulas.parent.mkdir(parents=True, exist_ok=True)
    outputs.matching_formulas.write_text(
        "\n".join(f"{key}: {value}" for key, value in sorted(result.formulas.items())) + "\n",
        encoding="utf-8",
    )
    outputs.latex_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in result.latex_tables.items():
        (outputs.latex_dir / filename).write_text(content + "\n", encoding="utf-8")
