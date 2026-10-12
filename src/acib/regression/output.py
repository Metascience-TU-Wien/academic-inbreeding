# Copyright © 2026 TU Wien

"""Persist regression tables, text reports, and LaTeX collections."""

from __future__ import annotations

import pandas as pd

from acib.regression.config import RegressionOutputs
from acib.regression.schemas import (
    RegressionCoefficientOutputSchema,
    RegressionDescriptiveOutputSchema,
    RegressionFitStatisticsOutputSchema,
)


def write_regression_suite_outputs(
    *,
    coefficients: pd.DataFrame,
    fit_statistics: pd.DataFrame,
    formulas: dict[str, str],
    summaries: dict[str, str],
    descriptives: pd.DataFrame,
    aic_bic: pd.DataFrame,
    aic_bic_ranked: pd.DataFrame,
    best_by_aic: pd.DataFrame,
    best_by_bic: pd.DataFrame,
    inbreeding_spec_comparison: pd.DataFrame,
    latex_tables: dict[str, str] | None,
    author_regression_input: pd.DataFrame | None,
    professor_article_input: pd.DataFrame | None,
    outputs: RegressionOutputs,
) -> None:
    """Write analysis products to their individually configured destinations.

    :param coefficients: Long-form coefficient table.
    :param fit_statistics: Model fit-statistics table.
    :param formulas: Mapping from model key to formula string.
    :param summaries: Mapping from model key to formatted summary text.
    :param descriptives: Dependent-variable descriptive statistics.
    :param aic_bic: AIC/BIC comparison table.
    :param aic_bic_ranked: Ranked AIC/BIC comparison table.
    :param best_by_aic: Best models selected by AIC.
    :param best_by_bic: Best models selected by BIC.
    :param inbreeding_spec_comparison: Inbreeding coefficient comparison table.
    :param latex_tables: LaTeX collection keyed by filename.
    :param author_regression_input: Optional prepared researcher-level input.
    :param professor_article_input: Optional prepared professor-article input.
    :param outputs: Explicit ``RegressionOutputs`` path collection.
    :return: None.
    """
    tables: list[tuple[str, pd.DataFrame]] = [
        ("model_coefficients", coefficients),
        ("model_fit_statistics", fit_statistics),
        ("dependent_variable_descriptives", descriptives),
        ("aic_bic_comparison", aic_bic),
        ("aic_bic_ranked", aic_bic_ranked),
        ("best_models_by_aic", best_by_aic),
        ("best_models_by_bic", best_by_bic),
        ("inbreeding_discipline_spec_comparison", inbreeding_spec_comparison),
    ]
    if author_regression_input is not None:
        tables.append(("regression_input", author_regression_input))
    if professor_article_input is not None:
        tables.append(("professor_article_input", professor_article_input))
    for frame, schema in (
        (coefficients, RegressionCoefficientOutputSchema),
        (fit_statistics, RegressionFitStatisticsOutputSchema),
        (descriptives, RegressionDescriptiveOutputSchema),
    ):
        if not frame.empty or not frame.columns.empty:
            schema.validate(frame, lazy=True)
    for output_name, frame in tables:
        path = getattr(outputs, output_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    formulas_path = outputs.model_formulas
    formulas_path.parent.mkdir(parents=True, exist_ok=True)
    formulas_path.write_text(
        "\n".join(f"{key}: {formula}" for key, formula in sorted(formulas.items())) + "\n",
        encoding="utf-8",
    )
    summaries_path = outputs.model_summaries
    summaries_path.parent.mkdir(parents=True, exist_ok=True)
    summaries_path.write_text(
        "\n\n".join(summaries[key] for key in sorted(summaries)) + "\n", encoding="utf-8"
    )
    latex_dir = outputs.latex_dir
    latex_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in (latex_tables or {}).items():
        (latex_dir / filename).write_text(content + "\n", encoding="utf-8")
