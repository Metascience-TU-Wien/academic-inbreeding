# Copyright © 2026 TU Wien

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from acib.constants import (
    ABSOLUTE_P_TOP_1,
    ABSOLUTE_P_TOP_5,
    ABSOLUTE_P_TOP_10,
    DOMAIN_SHARE_COLUMNS,
    HAZEN_PERCENTILE,
    N_PUBLIC,
    PP_TOP_1,
    PP_TOP_5,
    PP_TOP_10,
)
from acib.regression.config import OLS_MODEL_TYPES
from acib.regression.constants import (
    ABSOLUTE_PERFORMANCE_OUTCOMES,
    ABSOLUTE_TOP_PERCENTILE_OUTCOMES,
    OUTCOME_DISPLAY_LABELS,
)
from acib.regression.fields import ASJC_DOMAIN_NAMES
from acib.regression.model import FittedRegressionModel
from acib.regression.models import significance_code

RESEARCH_PERFORMANCE_OUTCOMES: tuple[str, ...] = (
    HAZEN_PERCENTILE,
    PP_TOP_10,
    PP_TOP_5,
    PP_TOP_1,
)
PERCENTAGE_POINT_OUTCOMES: set[str] = {PP_TOP_10, PP_TOP_5, PP_TOP_1}
TITLE_REFERENCE = "Full Prof."
TITLE_ORDER: tuple[str, ...] = (
    "Ao. Prof.",
    "Ass. Prof.",
    "FH-Ass. Prof.",
    "FH-Prof.",
)
LATEX_ROW_END = r"\\"
MODEL_COMPARISON_MODEL_IDS: tuple[str, ...] = tuple(f"m{index}" for index in range(1, 11))
_ESTIMATOR_DISPLAY_NAMES = {
    "frac": "Fractional logit",
    "huber": "Huber",
    "nb": "NB",
    "ols": "OLS",
    "olshc3": "OLS",
    "ppml": "PPML",
    "pr": "Poisson",
    "zinb": "ZINB",
}


def outcome_label(outcome: str, aliases: Mapping[str, str] | None = None) -> str:
    if aliases is not None and outcome in aliases:
        return aliases[outcome]
    return OUTCOME_DISPLAY_LABELS.get(outcome, latex_escape(outcome))


@dataclass(frozen=True, slots=True)
class LatexCoefficient:
    estimate: float
    std_error: float
    p_value: float


@dataclass(frozen=True, slots=True)
class LatexModelColumn:
    dependent_variable: str
    model: FittedRegressionModel

    @property
    def scale(self) -> float:
        return 100.0 if self.dependent_variable in PERCENTAGE_POINT_OUTCOMES else 1.0

    def term(self, term_name: str) -> LatexCoefficient | None:
        if self.model.model_type in {"frac", "ppml", "nb"}:
            row = self.model.reportable_effect(term_name)
            if row is None:
                return None
            return LatexCoefficient(
                estimate=float(row["estimate"]),
                std_error=float(row["std_error"]),
                p_value=float(row["p_value"]),
            )
        fit = self.model.fit
        if term_name not in fit.params.index:
            return None
        return LatexCoefficient(
            estimate=float(fit.params[term_name]) * self.scale,
            std_error=float(fit.bse[term_name]) * self.scale,
            p_value=float(fit.pvalues[term_name]),
        )

    def title_contrast(
        self, title: str, *, reference: str = TITLE_REFERENCE
    ) -> LatexCoefficient | None:
        if title == reference:
            return None
        categories = title_categories(self.model.model_data)
        if title not in categories or reference not in categories:
            return None
        if self.model.model_type in {"frac", "ppml", "nb"}:
            row = self.model.categorical_contrast(
                variable="title", level=title, reference=reference
            )
            if row is None:
                return None
            return LatexCoefficient(
                estimate=float(row["estimate"]),
                std_error=float(row["std_error"]),
                p_value=float(row["p_value"]),
            )
        weights = title_contrast_weights(self.model.fit, title=title, reference=reference)
        if not np.any(weights):
            return LatexCoefficient(estimate=0.0, std_error=0.0, p_value=1.0)
        test = self.model.fit.t_test(weights)
        return LatexCoefficient(
            estimate=float(np.ravel(test.effect)[0]) * self.scale,
            std_error=float(np.ravel(test.sd)[0]) * self.scale,
            p_value=float(np.ravel(test.pvalue)[0]),
        )


def research_performance_latex_table(
    models: dict[tuple[str, str], FittedRegressionModel],
    *,
    model_id: str = "m9",
    outcomes: tuple[str, ...] = RESEARCH_PERFORMANCE_OUTCOMES,
    outcome_aliases: Mapping[str, str] | None = None,
) -> str:
    """
    Render the paper research-performance table for one model specification.

    :param models: Fitted models keyed by ``(dependent_variable, model_id)``.
    :param model_id: Model specification to render, usually ``m9``.
    :return: Complete LaTeX table.
    """
    return performance_latex_table(
        models,
        model_id=model_id,
        outcomes=outcomes,
        performance_type="relative",
        outcome_aliases=outcome_aliases,
    )


def absolute_performance_latex_table(
    models: dict[tuple[str, str], FittedRegressionModel],
    *,
    model_id: str = "m9",
    outcomes: tuple[str, ...] = ABSOLUTE_PERFORMANCE_OUTCOMES,
    outcome_aliases: Mapping[str, str] | None = None,
) -> str:
    """Render the absolute-performance analogue of the relative-performance table."""
    return performance_latex_table(
        models,
        model_id=model_id,
        outcomes=outcomes,
        performance_type="absolute",
        outcome_aliases=outcome_aliases,
    )


def performance_latex_table(
    models: dict[tuple[str, str], FittedRegressionModel],
    *,
    model_id: str,
    outcomes: tuple[str, ...],
    performance_type: str,
    outcome_aliases: Mapping[str, str] | None = None,
) -> str:
    columns = [
        LatexModelColumn(dependent_variable=outcome, model=models[(outcome, model_id)])
        for outcome in outcomes
    ]
    lines = table_preamble(
        model_id=model_id,
        outcomes=outcomes,
        performance_type=performance_type,
        sample_sizes=tuple(int(column.model.fit.nobs) for column in columns),
    )
    lines.extend(
        outcome_header(
            outcomes,
            models=models,
            model_id=model_id,
            outcome_aliases=outcome_aliases,
        )
    )
    lines.append(r"\midrule")
    lines.append("")
    lines.append(coefficient_row("Inbreeding", columns, term_name="C(is_inbred)[T.True]"))
    lines.append("")
    lines.append(coefficient_row("Academic age", columns, term_name="academic_age"))
    lines.append("")
    lines.append(r"\addlinespace[3pt]")
    lines.append("")
    lines.append(
        rf"\multicolumn{{{1 + 3 * len(columns)}}}"
        r"{@{}l}{\textit{Academic title (ref.: Full Prof.)}} \\[2pt]"
    )
    lines.append("")
    for title in title_rows_for_columns(columns):
        lines.append(title_row(title, columns))
        lines.append("")
    lines.append(
        coefficient_row(
            "Co-affiliation publication share",
            columns,
            term_name="coaffiliation_publication_share",
        )
    )
    lines.append("")
    lines.append(
        coefficient_row(
            "Mean coauthors",
            columns,
            term_name="mean_coauthors_per_publication",
        )
    )
    lines.append("")
    lines.append(coefficient_row("Domestic mobility", columns, term_name="n_away_years_domestic"))
    lines.append("")
    lines.append(
        coefficient_row("International mobility", columns, term_name="n_away_years_abroad")
    )
    lines.append("")
    lines.append(coefficient_row("First publication year", columns, term_name="first_year"))
    lines.append("")
    lines.append(r"\addlinespace[3pt]")
    lines.append("")
    lines.extend(discipline_control_rows(columns))
    lines.append("")
    lines.append(r"\midrule")
    lines.append("")
    sample_sizes = {int(column.model.fit.nobs) for column in columns}
    if len(sample_sizes) > 1:
        lines.append(observations_row(columns))
    lines.append("")
    if all(column.model.model_type in OLS_MODEL_TYPES for column in columns):
        lines.append(adjusted_r_squared_row(columns))
    lines.extend(
        table_postamble(
            model_id=model_id,
            outcomes=outcomes,
            performance_type=performance_type,
            outcome_aliases=outcome_aliases,
            model_types={column.dependent_variable: column.model.model_type for column in columns},
            columns=columns,
        )
    )
    return "\n".join(lines)


def can_render_model10_without_international_mobility_table(
    models: dict[tuple[str, str], FittedRegressionModel],
    *,
    relative_outcomes: tuple[str, ...],
    absolute_outcomes: tuple[str, ...],
) -> bool:
    outcomes = (*relative_outcomes, *absolute_outcomes)
    return bool(relative_outcomes and absolute_outcomes) and all(
        (outcome, "m10") in models for outcome in outcomes
    )


def model10_without_international_mobility_latex_table(
    models: dict[tuple[str, str], FittedRegressionModel],
    *,
    relative_outcomes: tuple[str, ...],
    absolute_outcomes: tuple[str, ...],
    outcome_aliases: Mapping[str, str] | None = None,
) -> str:
    """Render Model 10 in the compact relative/absolute outcome layout."""
    outcomes = (*relative_outcomes, *absolute_outcomes)
    columns = [
        LatexModelColumn(dependent_variable=outcome, model=models[(outcome, "m10")])
        for outcome in outcomes
    ]
    sample_sizes = tuple(int(column.model.fit.nobs) for column in columns)
    caption = (
        r"\caption{Association between academic inbreeding and research performance excluding "
        rf"international mobility{latex_sample_size_caption(sample_sizes)}}}"
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        caption,
        r"\label{tab:without_international_mobility}",
        "",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{5.5pt}",
        "",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{",
        "@{}",
        "l",
        *["S[table-format=-2.2,table-space-text-post={***}]" for _ in outcomes],
        "@{}",
        "}",
        "",
        r"\toprule",
        latex_row(
            [
                "{}",
                (
                    rf"\multicolumn{{{len(relative_outcomes)}}}"
                    r"{c}{\textbf{Relative citation impact}}"
                ),
                (
                    rf"\multicolumn{{{len(absolute_outcomes)}}}"
                    r"{c}{\textbf{Absolute publication output}}"
                ),
            ]
        ),
        "",
        rf"\cmidrule(lr){{2-{1 + len(relative_outcomes)}}}",
        rf"\cmidrule(lr){{{2 + len(relative_outcomes)}-{1 + len(outcomes)}}}",
        "",
        latex_row(
            [
                "{}",
                *[
                    model10_outcome_header_cell(outcome, aliases=outcome_aliases)
                    for outcome in outcomes
                ],
            ]
        ),
        "",
        "",
        r"\midrule",
        "",
    ]

    estimates = [column.term("C(is_inbred)[T.True]") for column in columns]
    lines.extend(
        [
            latex_row(["Inbreeding", *[format_model10_estimate(value) for value in estimates]]),
            "",
            latex_row(
                [
                    (
                        "Robust SE"
                        if all(column_uses_robust_standard_errors(column) for column in columns)
                        else "SE"
                    ),
                    *[format_model10_standard_error(value) for value in estimates],
                ]
            ),
            "",
            latex_row([r"$P$", *[format_model10_p_value(value) for value in estimates]]),
        ]
    )
    if len(set(sample_sizes)) > 1:
        lines.append(
            latex_row(
                [
                    "Observations",
                    *[latex_integer(sample_size) for sample_size in sample_sizes],
                ]
            )
        )
    lines.extend(
        [
            "",
            r"\midrule",
            latex_row(["Estimator", *[model10_estimator_cell(column) for column in columns]]),
            latex_row(
                [
                    "Effect",
                    *[model10_effect_cell(column) for column in columns],
                ]
            ),
            "",
            r"\bottomrule",
            r"\end{tabular}%",
            "}",
            "",
            r"\vspace{2mm}",
            "",
            r"\begin{minipage}{\textwidth}",
            r"\footnotesize",
            model10_without_international_mobility_note(
                columns=columns,
                outcomes=outcomes,
                outcome_aliases=outcome_aliases,
            ),
            r"\end{minipage}",
            "",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def format_model10_estimate(coefficient: LatexCoefficient | None) -> str:
    return f"{coefficient.estimate:.2f}" if coefficient is not None else "--"


def model10_without_international_mobility_note(
    *,
    columns: list[LatexModelColumn],
    outcomes: tuple[str, ...],
    outcome_aliases: Mapping[str, str] | None,
) -> str:
    """Return estimator-appropriate notes for the Model 10 robustness table.

    :param columns: Fitted outcome/model columns in the table.
    :param outcomes: Outcome keys in display order.
    :param outcome_aliases: Optional dependent-variable labels.
    :return: LaTeX note describing estimators, outcomes, inference, and controls.
    """
    models_by_outcome = {column.dependent_variable: column.model for column in columns}
    canonical_model_10 = all(
        models_by_outcome[outcome].model_type == ("nb" if outcome == N_PUBLIC else "olshc3")
        for outcome in outcomes
    )
    if not canonical_model_10:
        return performance_table_note(
            columns=columns,
            outcomes=outcomes,
            performance_type="mixed",
            model_id="m10",
            outcome_aliases=outcome_aliases,
            has_p_column=True,
        )
    return (
        r"\textit{Notes.} Reported are regression coefficients (Coef.) or average marginal "
        r"effects (AMEs); estimators are ordinary least squares (OLS) and negative binomial "
        r"regression (NB). AME denotes average marginal effect; P denotes p-value. "
        r"The dependent variables are the mean weighted percentile "
        r"rank (mwPR) and the proportions of publications in the respective citation-impact "
        r"percentiles (PP(top 10\%), PP(top 5\%), and PP(top 1\%)) and the number of "
        r"publications (Publications) and fractionally counted numbers of highly cited "
        r"publications (P(top 10\%), P(top 5\%), and P(top 1\%)). mwPR effects are expressed "
        r"in percentile points; PP denotes proportions of publications in the respective "
        r"citation-impact percentiles; their effects are expressed in percentage points. AMEs "
        r"for Publications represent changes in the expected number of publications, whereas "
        r"coefficients for P(top 10\%), P(top 5\%), and P(top 1\%) represent changes in "
        r"fractionally counted numbers of highly cited publications. HC3 heteroskedasticity-"
        r"robust standard errors (Robust SE); OLS $p$-values are based on $t$-tests, and NB "
        r"AME $p$-values are based on $z$-tests. AMEs are average derivatives for continuous "
        r"predictors and average discrete differences for categorical predictors. Full Prof. "
        r"(Full Professor) is the reference category for academic title; all academic-title "
        r"abbreviations are defined in Table~\ref{tab:prof_titles}. Research-field composition "
        r"is represented by fractional publication shares across 27 major fields of the All "
        r"Science Journal Classification (ASJC), with Medicine as the reference field because "
        r"it has the largest overall publication share. All models include the explanatory "
        r"variables of the full specification except international mobility. Significance "
        r"levels: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. Definitions and calculations "
        r"of the analytical variables are provided in Table~\ref{tab:analytical_variables}."
    )


def format_model10_standard_error(coefficient: LatexCoefficient | None) -> str:
    return f"{coefficient.std_error:.2f}" if coefficient is not None else "--"


def format_model10_p_value(coefficient: LatexCoefficient | None) -> str:
    if coefficient is None:
        return "--"
    value = coefficient.p_value
    stars = significance_code(value)
    if stars == ".":
        stars = ""
    return f"{value:.2f}\\textsuperscript{{{stars}}}" if stars else f"{value:.2f}"


def model10_outcome_header_cell(
    outcome: str,
    *,
    aliases: Mapping[str, str] | None = None,
) -> str:
    label = outcome_label(outcome, aliases)
    return rf"\multicolumn{{1}}{{c}}{{\textbf{{{label}}}}}"


def model10_estimator_cell(column: LatexModelColumn) -> str:
    """Render the configured estimator name for one Model 10 outcome.

    :param column: Outcome/model column in the supplementary table.
    :return: Centered LaTeX cell containing the estimator label.
    """
    model_type = column.model.model_type
    label = _ESTIMATOR_DISPLAY_NAMES.get(model_type, model_type)
    return rf"\multicolumn{{1}}{{c}}{{{label}}}"


def model10_effect_cell(column: LatexModelColumn) -> str:
    """Render whether one Model 10 outcome reports an AME or coefficient.

    :param column: Outcome/model column in the supplementary table.
    :return: Centered LaTeX cell containing the effect label.
    """
    effect = "AME" if column.model.model_type in {"frac", "ppml", "nb"} else "Coef."
    return rf"\multicolumn{{1}}{{c}}{{{effect}}}"


def column_uses_robust_standard_errors(column: LatexModelColumn) -> bool:
    """Return whether a fitted model uses a robust covariance estimator.

    Huber RLM results expose their H1 covariance type on the model metadata
    rather than as ``fit.cov_type``.

    :param column: Fitted model column in the Model 10 table.
    :return: Whether the model uses a covariance estimator other than nonrobust.
    """
    fit = column.model.fit
    covariance_estimator = getattr(fit.model, "_acib_covariance_estimator", None)
    if covariance_estimator is None:
        covariance_estimator = getattr(fit, "cov_type", "nonrobust")
    return str(covariance_estimator).upper() != "NONROBUST"


def join_labels(labels: tuple[str, ...]) -> str:
    if len(labels) < 2:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return " and ".join(labels)
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def absolute_top_percentile_latex_table(
    models: dict[tuple[str, str], FittedRegressionModel],
    *,
    model_id: str = "m9",
    outcome_aliases: Mapping[str, str] | None = None,
) -> str:
    """Render the Model 9 robustness table for fractional absolute top-x% counts."""
    columns = [
        LatexModelColumn(dependent_variable=outcome, model=models[(outcome, model_id)])
        for outcome in ABSOLUTE_TOP_PERCENTILE_OUTCOMES
    ]
    sample_sizes = tuple(int(column.model.fit.nobs) for column in columns)
    field_share_description = shared_share_control_value(
        columns,
        control_type="detailed_asjc_shares",
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Robustness regressions for fractional counts of highly cited "
            rf"publications (Model {model_id.removeprefix('m')})"
            rf"{latex_sample_size_caption(sample_sizes)}}}"
        ),
        r"\label{tab:absolute_top_percentile_model9}",
        "",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{",
        "@{}",
        "l",
        "S[table-format=-3.2]",
        "S[table-format=2.2]",
        "S[table-format=1.3,table-space-text-post={***}]",
        "S[table-format=-3.2]",
        "S[table-format=2.2]",
        "S[table-format=1.3,table-space-text-post={***}]",
        "S[table-format=-3.2]",
        "S[table-format=2.2]",
        "S[table-format=1.3,table-space-text-post={***}]",
        "@{}",
        "}",
        r"\toprule",
        absolute_top_percentile_outcome_header(
            columns,
            outcome_aliases=outcome_aliases,
        ),
        r"\midrule",
        "",
        coefficient_row("Inbreeding", columns, term_name="C(is_inbred)[T.True]"),
        "",
        coefficient_row("Academic age", columns, term_name="academic_age"),
        "",
        r"\addlinespace[3pt]",
        "",
        r"\multicolumn{10}{@{}l}{\textit{Academic title (ref.: Full Prof.)}} \\[2pt]",
        "",
    ]
    for title in title_rows_for_columns(columns):
        lines.extend([title_row(title, columns), ""])
    lines.extend(
        [
            coefficient_row(
                "Co-affiliation publication share",
                columns,
                term_name="coaffiliation_publication_share",
            ),
            "",
            coefficient_row("Mean coauthors", columns, term_name="mean_coauthors_per_publication"),
            "",
            coefficient_row("Domestic mobility", columns, term_name="n_away_years_domestic"),
            "",
            coefficient_row("International mobility", columns, term_name="n_away_years_abroad"),
            "",
            coefficient_row("First publication year", columns, term_name="first_year"),
            "",
            r"\addlinespace[3pt]",
            "",
            latex_row(
                [
                    "ASJC field shares",
                    (rf"\multicolumn{{9}}{{c}}{{" f"{field_share_description}}}"),
                ]
            ),
            "",
            r"\midrule",
            "",
        ]
    )
    if len(set(sample_sizes)) > 1:
        lines.extend([observations_row(columns), ""])
    if all(column.model.model_type in OLS_MODEL_TYPES for column in columns):
        lines.append(adjusted_r_squared_row(columns))
    lines.extend(
        absolute_top_percentile_table_postamble(
            model_id=model_id,
            columns=columns,
            outcome_aliases=outcome_aliases,
        )
    )
    return "\n".join(lines)


def absolute_top_percentile_outcome_header(
    columns: list[LatexModelColumn],
    *,
    outcome_aliases: Mapping[str, str] | None = None,
) -> str:
    labels = [
        outcome_label(outcome, outcome_aliases) for outcome in ABSOLUTE_TOP_PERCENTILE_OUTCOMES
    ]
    groups = [rf"\multicolumn{{3}}{{c}}{{\textbf{{{label}}}}}" for label in labels]
    header = latex_row(["{}", *groups])
    subheader_cells = [r"\textbf{Variable}"]
    for column in columns:
        effect_heading = "AME" if column.model.model_type in {"frac", "ppml", "nb"} else "Coef."
        robust = column.model.model_type in {"olshc3", "huber", "frac", "ppml", "nb"}
        error_heading = (
            r"\shortstack{\textbf{Robust}\\\textbf{std. err.}}" if robust else r"\textbf{SE}"
        )
        subheader_cells.extend(
            [
                rf"\multicolumn{{1}}{{c}}{{\textbf{{{effect_heading}}}}}",
                rf"\multicolumn{{1}}{{c}}{{{error_heading}}}",
                r"\multicolumn{1}{c}{\textbf{$P$}}",
            ]
        )
    subheader = latex_row(subheader_cells)
    return (
        f"{header}\n"
        r"\cmidrule(lr){2-4}"
        "\n"
        r"\cmidrule(lr){5-7}"
        "\n"
        r"\cmidrule(lr){8-10}"
        "\n\n"
        f"{subheader}"
    )


def absolute_top_percentile_table_postamble(
    *,
    model_id: str,
    columns: list[LatexModelColumn],
    outcome_aliases: Mapping[str, str] | None = None,
) -> list[str]:
    notes = performance_table_note(
        columns=columns,
        outcomes=ABSOLUTE_TOP_PERCENTILE_OUTCOMES,
        performance_type="absolute",
        model_id=model_id,
        outcome_aliases=outcome_aliases,
    )
    return [
        "",
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        "",
        r"\vspace{2mm}",
        r"\begin{minipage}{\textwidth}",
        r"\scriptsize",
        notes,
        r"\end{minipage}",
        "",
        r"\end{table}",
    ]


def can_render_absolute_top_percentile_latex_table(
    models: dict[tuple[str, str], FittedRegressionModel],
    *,
    model_id: str = "m9",
) -> bool:
    return can_render_performance_latex_table(
        models,
        outcomes=ABSOLUTE_TOP_PERCENTILE_OUTCOMES,
        model_id=model_id,
    )


def can_render_research_performance_latex_table(
    models: dict[tuple[str, str], FittedRegressionModel],
    *,
    model_id: str = "m9",
    outcomes: tuple[str, ...] = RESEARCH_PERFORMANCE_OUTCOMES,
) -> bool:
    """
    Check whether all outcome models needed for the paper table are available.

    :param models: Fitted models keyed by ``(dependent_variable, model_id)``.
    :param model_id: Model specification to render.
    :return: Whether the complete table can be rendered.
    """
    return can_render_performance_latex_table(
        models,
        outcomes=outcomes,
        model_id=model_id,
    )


def can_render_performance_latex_table(
    models: dict[tuple[str, str], FittedRegressionModel],
    *,
    outcomes: tuple[str, ...],
    model_id: str,
) -> bool:
    return bool(outcomes) and all((outcome, model_id) in models for outcome in outcomes)


def renderable_research_performance_model_ids(
    models: dict[tuple[str, str], FittedRegressionModel],
    *,
    outcomes: tuple[str, ...] = RESEARCH_PERFORMANCE_OUTCOMES,
) -> tuple[str, ...]:
    """
    Return model ids with all outcome models needed for the paper table.

    :param models: Fitted models keyed by ``(dependent_variable, model_id)``.
    :return: Sorted model ids that can be rendered as complete performance tables.
    """
    model_ids = sorted({model_id for _, model_id in models})
    return tuple(
        model_id
        for model_id in model_ids
        if can_render_research_performance_latex_table(
            models,
            model_id=model_id,
            outcomes=outcomes,
        )
    )


def renderable_absolute_performance_model_ids(
    models: dict[tuple[str, str], FittedRegressionModel],
    *,
    outcomes: tuple[str, ...] = ABSOLUTE_PERFORMANCE_OUTCOMES,
) -> tuple[str, ...]:
    model_ids = sorted({model_id for _, model_id in models})
    return tuple(
        model_id
        for model_id in model_ids
        if can_render_performance_latex_table(
            models,
            model_id=model_id,
            outcomes=outcomes,
        )
    )


def table_preamble(
    *,
    model_id: str,
    outcomes: tuple[str, ...] = RESEARCH_PERFORMANCE_OUTCOMES,
    performance_type: str = "relative",
    sample_sizes: tuple[int, ...] | None = None,
) -> list[str]:
    label_prefix = (
        "research_performance" if performance_type == "relative" else "absolute_performance"
    )
    caption_suffix = (
        "relative citation impact"
        if performance_type == "relative"
        else "absolute research performance"
    )
    estimate_format = "-2.2" if performance_type == "absolute" else "-1.2"
    standard_error_format = "2.2" if performance_type == "absolute" else "1.2"
    columns = [
        value
        for _ in outcomes
        for value in (
            f"S[table-format={estimate_format}]",
            f"S[table-format={standard_error_format}]",
            "S[table-format=1.2,table-space-text-post={***}]",
        )
    ]
    sample_caption = latex_sample_size_caption(sample_sizes)
    return [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            rf"\caption{{Association between academic inbreeding and {caption_suffix}"
            rf"{sample_caption}}}"
        ),
        rf"\label{{tab:{label_prefix}_{model_id}}}",
        "",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{5.5pt}",
        "",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{",
        "@{}",
        "l",
        *columns,
        "@{}",
        "}",
        "",
        r"\toprule",
    ]


def latex_sample_size_caption(sample_sizes: tuple[int, ...] | None) -> str:
    """Format sample size information for a table caption.

    :param sample_sizes: Sample size for each outcome or model column.
    :return: LaTeX caption suffix, empty when no sizes are available.
    """
    if not sample_sizes:
        return ""
    unique_sizes = sorted(set(sample_sizes))
    if len(unique_sizes) == 1:
        return rf" (\(n={latex_integer(unique_sizes[0])}\))"
    low, high = unique_sizes[0], unique_sizes[-1]
    return rf" (\(n={latex_integer(low)}\)–\(n={latex_integer(high)}\))"


def latex_integer(value: int) -> str:
    """Format an integer with LaTeX-safe thousands separators."""
    return f"{value:,}".replace(",", "{,}")


def outcome_header(
    outcomes: tuple[str, ...],
    *,
    models: Mapping[tuple[str, str], FittedRegressionModel],
    model_id: str,
    outcome_aliases: Mapping[str, str] | None = None,
) -> list[str]:
    header = latex_row(
        [
            "{}",
            *[
                outcome_header_cell(outcome, outcome_aliases=outcome_aliases)
                for outcome in outcomes
            ],
        ]
    )
    cmidrules = [
        rf"\cmidrule(lr){{{2 + 3 * index}-{4 + 3 * index}}}" for index in range(len(outcomes))
    ]
    columns = [r"\textbf{Variable}"]
    for outcome in outcomes:
        model_type = models[(outcome, model_id)].model_type
        effect_heading = "AME" if model_type in {"frac", "ppml", "nb"} else "Coef."
        robust = model_type in {"olshc3", "huber", "frac", "ppml", "nb"}
        error_heading = r"\shortstack{\textbf{Robust}\\\textbf{SE}}" if robust else r"\textbf{SE}"
        columns.extend(
            [
                rf"\multicolumn{{1}}{{c}}{{\textbf{{{effect_heading}}}}}",
                rf"\multicolumn{{1}}{{c}}{{{error_heading}}}",
                r"\multicolumn{1}{c}{\textbf{$P$}}",
            ]
        )
    return [header, "", *cmidrules, "", latex_row(columns), ""]


def outcome_header_cell(
    outcome: str,
    *,
    outcome_aliases: Mapping[str, str] | None = None,
) -> str:
    label = outcome_label(outcome, outcome_aliases)
    return rf"\multicolumn{{3}}{{c}}{{\textbf{{{label}}}}}"


def coefficient_row(label: str, columns: list[LatexModelColumn], *, term_name: str) -> str:
    cells = [label]
    for column in columns:
        cells.extend(format_coefficient_cells(column.term(term_name)))
    return latex_row(cells)


def title_row(title: str, columns: list[LatexModelColumn]) -> str:
    cells = [rf"\quad {latex_title_label(title)}"]
    for column in columns:
        cells.extend(format_coefficient_cells(column.title_contrast(title)))
    return latex_row(cells)


def included_row(label: str) -> str:
    return "\n".join(
        [
            label,
            r"& \multicolumn{3}{c}{Included}",
            r"& \multicolumn{3}{c}{Included}",
            r"& \multicolumn{3}{c}{Included}",
            rf"& \multicolumn{{3}}{{c}}{{Included}} {LATEX_ROW_END}",
        ]
    )


def discipline_control_rows(columns: list[LatexModelColumn]) -> list[str]:
    rows = []
    if any(column.model.share_control_type == "detailed_asjc_shares" for column in columns):
        rows.append(major_field_share_row(columns))
    return rows


def major_field_share_row(columns: list[LatexModelColumn]) -> str:
    value = shared_share_control_value(columns, control_type="detailed_asjc_shares")
    return latex_row(["ASJC field shares", rf"\multicolumn{{{3 * len(columns)}}}{{c}}{{{value}}}"])


def shared_share_control_value(
    columns: list[LatexModelColumn],
    *,
    control_type: str,
) -> str:
    values = {share_control_value(column.model, control_type=control_type) for column in columns}
    return values.pop() if len(values) == 1 else "Included"


def broad_field_share_rows(columns: list[LatexModelColumn]) -> list[str]:
    if not any(column.model.share_control_type == "broad_asjc_shares" for column in columns):
        return []
    reference = next(
        (
            column.model.field_share_reference
            for column in columns
            if column.model.share_control_type == "broad_asjc_shares"
            and column.model.field_share_reference
        ),
        None,
    )
    reference_label = latex_domain_label(reference)
    rows = [
        broad_field_share_header(reference_label, column_count=1 + 3 * len(columns)),
        "",
    ]
    for share_column in DOMAIN_SHARE_COLUMNS:
        if share_column == reference:
            continue
        if any(column.term(share_column) is not None for column in columns):
            label = rf"\quad {latex_domain_label(share_column)}"
            rows.append(
                coefficient_row(
                    label,
                    columns,
                    term_name=share_column,
                )
            )
            rows.append("")
    return rows


def broad_field_share_header(reference_label: str, *, column_count: int = 13) -> str:
    return (
        rf"\multicolumn{{{column_count}}}{{@{{}}l}}{{\textit{{ASJC field "
        rf"(ref.: {reference_label})}}}} \\[2pt]"
    )


def broad_field_share_summary_row(columns: list[LatexModelColumn]) -> str:
    cells = ["Broad ASJC shares"]
    for column in columns:
        value = share_control_value(column.model, control_type="broad_asjc_shares")
        cells.append(rf"\multicolumn{{3}}{{c}}{{{value}}}")
    return latex_row(cells)


def share_control_value(model: FittedRegressionModel, *, control_type: str) -> str:
    if model.share_control_type != control_type:
        return "Not included"
    reference = model.field_share_reference
    if control_type == "detailed_asjc_shares" and reference == "med_share":
        return "27 major fields; Medicine as reference"
    return f"Included; ref.: {latex_field_label(reference)}" if reference else "Included"


def latex_field_label(value: str | None) -> str:
    if not value:
        return ""
    if value == "med_share":
        return "Medicine"
    return value.replace("_", r"\_")


def latex_domain_label(value: str | None) -> str:
    if not value:
        return ""
    domain = value.removesuffix("_share")
    label = ASJC_DOMAIN_NAMES.get(domain, value)
    return label.replace("&", r"\&").replace("_", r"\_")


def observations_row(columns: list[LatexModelColumn]) -> str:
    cells = ["Observations"]
    for column in columns:
        cells.append(rf"\multicolumn{{3}}{{c}}{{{latex_integer(int(column.model.fit.nobs))}}}")
    return latex_row(cells)


def adjusted_r_squared_row(columns: list[LatexModelColumn]) -> str:
    cells = [r"Adjusted $R^2$"]
    for column in columns:
        value = (
            getattr(column.model.fit, "rsquared_adj", np.nan)
            if column.model.model_type in OLS_MODEL_TYPES | {"huber"}
            else np.nan
        )
        formatted = f"{float(value):.2f}" if np.isfinite(value) else "---"
        cells.append(rf"\multicolumn{{3}}{{c}}{{{formatted}}}")
    return latex_row(cells)


def outcome_description(
    outcomes: tuple[str, ...],
    *,
    performance_type: str,
    outcome_aliases: Mapping[str, str] | None = None,
) -> str:
    """Describe the dependent variables actually represented in a table.

    :param outcomes: Outcome keys included in the table.
    :param performance_type: Either relative or absolute performance.
    :return: Publication-ready dependent-variable description.
    """
    labels = {outcome: outcome_label(outcome, outcome_aliases) for outcome in outcomes}
    if performance_type == "mixed":
        relative = tuple(
            outcome
            for outcome in outcomes
            if outcome == HAZEN_PERCENTILE or outcome in PERCENTAGE_POINT_OUTCOMES
        )
        absolute = tuple(outcome for outcome in outcomes if outcome not in relative)
        parts = []
        if relative:
            parts.append(
                outcome_description(
                    relative,
                    performance_type="relative",
                    outcome_aliases=outcome_aliases,
                )
            )
        if absolute:
            parts.append(
                outcome_description(
                    absolute,
                    performance_type="absolute",
                    outcome_aliases=outcome_aliases,
                )
            )
        return " and ".join(parts)
    descriptions = []
    if performance_type == "relative":
        if HAZEN_PERCENTILE in outcomes:
            descriptions.append("the mean weighted percentile rank (mwPR)")
        percentile_outcomes = [
            outcome for outcome in outcomes if outcome in PERCENTAGE_POINT_OUTCOMES
        ]
        if percentile_outcomes:
            names = join_labels(tuple(labels[outcome] for outcome in percentile_outcomes))
            descriptions.append(
                f"the proportions of publications in the respective citation-impact percentiles "
                f"({names})"
            )
    else:
        if N_PUBLIC in outcomes:
            descriptions.append("the number of publications (Publications)")
        count_outcomes = [
            outcome for outcome in outcomes if outcome in ABSOLUTE_TOP_PERCENTILE_OUTCOMES
        ]
        if count_outcomes:
            names = join_labels(tuple(labels[outcome] for outcome in count_outcomes))
            descriptions.append(
                f"fractionally counted numbers of highly cited publications ({names})"
            )
    return " and ".join(descriptions)


def estimator_description(model_type: str) -> str:
    """Expand an estimator identifier into its publication-ready name."""
    descriptions = {
        "frac": "fractional logit",
        "huber": "Huber robust regression",
        "nb": "negative binomial regression (NB)",
        "ols": "ordinary least squares (OLS)",
        "olshc3": "ordinary least squares (OLS)",
        "ppml": "Poisson pseudo-maximum likelihood (PPML)",
        "pr": "Poisson regression",
        "zinb": "zero-inflated negative binomial regression",
    }
    return descriptions.get(model_type, model_type)


def table_estimator_note(columns: list[LatexModelColumn], *, has_p_column: bool = True) -> str:
    """Describe fitted estimators and whether tables show coefficients or AMEs.

    :param columns: Fitted outcome/model columns in the table.
    :param has_p_column: Whether the table labels a p-value column as ``P``.
    :return: Estimator and reported-effect description.
    """
    model_types = tuple(dict.fromkeys(column.model.model_type for column in columns))
    estimators = join_labels(tuple(estimator_description(value) for value in model_types))
    has_ames = any(value in {"frac", "ppml", "nb"} for value in model_types)
    if len(model_types) == 1 and has_ames:
        reported = f"Reported are average marginal effects (AMEs) from {estimators}. "
    elif len(model_types) == 1:
        reported = f"Reported are regression coefficients (Coef.) from {estimators}. "
    elif has_ames:
        reported = (
            "Reported are regression coefficients (Coef.) or average marginal effects (AMEs); "
            f"estimators are {estimators}. "
        )
    else:
        reported = f"Reported are regression coefficients (Coef.) from {estimators}. "
    abbreviations = ""
    if has_ames:
        abbreviations += "AME denotes average marginal effect; "
    if has_p_column:
        abbreviations += "P denotes p-value. "
    return reported + abbreviations


def table_inference_note(columns: list[LatexModelColumn]) -> str:
    """Describe the covariance estimator and tests represented in a table.

    :param columns: Fitted outcome/model columns in the table.
    :return: Estimator-specific standard-error and p-value explanation.
    """
    covariance_by_type: dict[str, set[str]] = {}
    for column in columns:
        covariance = str(
            getattr(
                column.model.fit.model,
                "_acib_covariance_estimator",
                getattr(column.model.fit, "cov_type", "nonrobust"),
            )
        )
        covariance_by_type.setdefault(column.model.model_type, set()).add(covariance)
    unique_covariances = set().union(*covariance_by_type.values()) if covariance_by_type else set()
    if unique_covariances == {"HC3"}:
        standard_errors = "HC3 heteroskedasticity-robust standard errors (Robust SE)"
    elif unique_covariances == {"nonrobust"}:
        standard_errors = "model-based standard errors (SE)"
    else:
        covariance_details = []
        for model_type, estimators in covariance_by_type.items():
            estimator_name = estimator_description(model_type)
            labels = join_labels(tuple(sorted(estimators)))
            covariance_details.append(f"{estimator_name} uses {labels} covariance")
        standard_errors = "standard errors use " + "; ".join(covariance_details)

    model_types = {column.model.model_type for column in columns}
    t_types = model_types.intersection(OLS_MODEL_TYPES)
    z_types = model_types - OLS_MODEL_TYPES
    if t_types and z_types:
        tests = "OLS p-values are based on t-tests; other estimator p-values use z-tests"
    elif t_types:
        tests = "p-values are based on t-tests"
    else:
        tests = "p-values are based on z-tests"
    return f"{standard_errors}; {tests}. "


def table_units_note(columns: list[LatexModelColumn], *, performance_type: str) -> str:
    """Describe units for the outcomes and effects shown in a table.

    :param columns: Fitted outcome/model columns in table order.
    :param performance_type: Relative or absolute performance table.
    :return: Outcome-specific effect-unit explanation.
    """
    model_by_outcome = {column.dependent_variable: column.model.model_type for column in columns}
    if performance_type == "mixed":
        relative_outcomes = {
            column.dependent_variable
            for column in columns
            if column.dependent_variable == HAZEN_PERCENTILE
            or column.dependent_variable in PERCENTAGE_POINT_OUTCOMES
        }
        relative = tuple(
            column for column in columns if column.dependent_variable in relative_outcomes
        )
        absolute = [
            column for column in columns if column.dependent_variable not in relative_outcomes
        ]
        notes = []
        if relative:
            notes.append(table_units_note(list(relative), performance_type="relative").strip())
        if absolute:
            notes.append(table_units_note(absolute, performance_type="absolute").strip())
        return " ".join(notes) + (" " if notes else "")
    if performance_type == "relative":
        units = []
        if HAZEN_PERCENTILE in model_by_outcome:
            units.append("mwPR effects are expressed in percentile points")
        if any(outcome in PERCENTAGE_POINT_OUTCOMES for outcome in model_by_outcome):
            units.append(
                "PP denotes proportions of publications in the respective citation-impact "
                "percentiles; their effects are expressed in percentage points"
            )
        return "; ".join(units) + (". " if units else "")

    units = []
    if N_PUBLIC in model_by_outcome:
        model_type = model_by_outcome[N_PUBLIC]
        if model_type in {"nb", "ppml"}:
            units.append("Publications AMEs are expressed in publication-count units")
        elif model_type == "pr":
            units.append("Publications coefficients are on the log expected-count scale")
        elif model_type == "zinb":
            units.append(
                "zero-inflated negative binomial count coefficients are on the log expected-count "
                "scale and inflation coefficients are on the logit scale"
            )
        else:
            units.append("Publications effects are expressed in publication counts")
    count_outcomes = [
        outcome for outcome in model_by_outcome if outcome in ABSOLUTE_TOP_PERCENTILE_OUTCOMES
    ]
    if count_outcomes:
        units_by_description: dict[str, list[str]] = {}
        for outcome in count_outcomes:
            model_type = model_by_outcome[outcome]
            if model_type == "nb":
                description = "AMEs in publication counts after nearest-integer half-up rounding"
            elif model_type in {"ppml", "ols", "olshc3", "huber"}:
                quantity = "AMEs" if model_type == "ppml" else "coefficients"
                description = f"{quantity} in fractional publication-count units"
            elif model_type == "pr":
                description = "coefficients on the log expected-count scale"
            elif model_type == "zinb":
                description = "count-component coefficients on the log expected-count scale"
            else:
                description = "effects in the configured outcome units"
            units_by_description.setdefault(description, []).append(outcome_label(outcome))
        units.extend(
            f"{join_labels(tuple(labels))} {description}"
            for description, labels in units_by_description.items()
        )
    return "; ".join(units) + (". " if units else "")


def table_context_note(columns: list[LatexModelColumn]) -> str:
    """Add only reference-category and research-field context present in a table.

    :param columns: Fitted outcome/model columns in the table.
    :return: Applicable academic-title and research-field notes.
    """
    notes = []
    if any("title" in column.model.formula for column in columns):
        notes.append(
            r"Full Prof. (Full Professor) is the reference category for academic title; "
            r"all academic-title "
            r"abbreviations are defined in Table~\ref{tab:prof_titles}."
        )
    share_types = {column.model.share_control_type for column in columns}
    if "detailed_asjc_shares" in share_types:
        notes.append(
            "Research-field composition is represented by fractional publication "
            "shares across 27 major fields of the All Science Journal Classification (ASJC), "
            "with Medicine as the reference field because it has the largest overall publication "
            "share."
        )
    elif "broad_asjc_shares" in share_types:
        notes.append(
            "Where included, research-field composition is represented by fractional publication "
            "shares across broad domains of the All Science Journal Classification (ASJC)."
        )
    return " ".join(notes) + (" " if notes else "")


def performance_table_note(
    *,
    columns: list[LatexModelColumn],
    outcomes: tuple[str, ...],
    performance_type: str,
    model_id: str,
    outcome_aliases: Mapping[str, str] | None = None,
    has_p_column: bool = True,
) -> str:
    """Build an estimator- and outcome-specific note for a performance table.

    :param columns: Fitted outcome/model columns represented in the table.
    :param outcomes: Outcome keys included in the table.
    :param performance_type: Relative or absolute performance.
    :param model_id: Single model specification shown in the table.
    :param outcome_aliases: Optional labels for dependent variables.
    :param has_p_column: Whether a p-value column is headed ``P``.
    :return: Complete table-note paragraph.
    """
    model_types = {column.model.model_type for column in columns}
    if is_nb_publications_with_ols_percentile_counts(columns, performance_type):
        return negative_binomial_publication_note(columns)
    note = table_estimator_note(columns, has_p_column=has_p_column)
    dependent_variable_description = outcome_description(
        outcomes,
        performance_type=performance_type,
        outcome_aliases=outcome_aliases,
    )
    note += f"The dependent variables are {dependent_variable_description}. "
    if len(model_types) > 1 and performance_type != "mixed":
        mapping = []
        for column in columns:
            family = estimator_description(column.model.model_type)
            effect = "AMEs" if column.model.model_type in {"frac", "ppml", "nb"} else "coefficients"
            if len(outcomes) == 1:
                mapping.append(f"{column.model.model_id.upper()} uses {family} ({effect})")
            else:
                label = outcome_label(column.dependent_variable, outcome_aliases)
                mapping.append(f"{label} uses {family} ({effect})")
        note += "Estimator by outcome: " + "; ".join(dict.fromkeys(mapping)) + ". "
    note += table_units_note(columns, performance_type=performance_type)
    note += table_inference_note(columns)
    if len(model_types.intersection({"frac", "ppml", "nb"})):
        note += (
            "AMEs are average derivatives for continuous predictors and average discrete "
            "differences for categorical predictors. "
        )
    note += table_context_note(columns)
    if model_id == "m10" and performance_type == "mixed":
        note += (
            "All models include the explanatory variables of the full specification except "
            "international mobility. "
        )
    note += r"Significance levels: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. "
    note += (
        r"Definitions and calculations of the analytical variables are provided in "
        r"Table~\ref{tab:analytical_variables}."
    )
    return r"\textit{Notes.} " + note


def is_nb_publications_with_ols_percentile_counts(
    columns: list[LatexModelColumn],
    performance_type: str,
) -> bool:
    """Identify the absolute-performance table with NB publications and OLS counts.

    :param columns: Fitted models represented in the table.
    :param performance_type: Performance-table outcome family.
    :return: Whether the table has the requested mixed NB/OLS structure.
    """
    models = {column.dependent_variable: column.model.model_type for column in columns}
    return (
        performance_type == "absolute"
        and models.get(N_PUBLIC) == "nb"
        and all(
            models.get(outcome) in OLS_MODEL_TYPES for outcome in ABSOLUTE_TOP_PERCENTILE_OUTCOMES
        )
        and all(
            str(getattr(getattr(column.model, "fit", None), "cov_type", "")).upper() == "HC3"
            for column in columns
        )
    )


def negative_binomial_publication_note(columns: list[LatexModelColumn]) -> str:
    """Build the specific note for NB publication counts and OLS percentile counts.

    :param columns: Fitted models represented in the table.
    :return: Estimator- and outcome-specific note.
    """
    note = (
        r"\textit{Notes.} "
        r"Reported are average marginal effects (AMEs) from negative binomial regression (NB) "
        r"for Publications and ordinary least squares (OLS) regression coefficients (Coef.) "
        r"for P(top 10\%), P(top 5\%), and P(top 1\%), with HC3 "
        r"heteroskedasticity-robust standard errors (Robust SE) and $p$-values ($P$). "
        r"The dependent variables are the number of publications per professor (Publications) "
        r"and the "
        r"fractionally counted numbers of publications in the top 10\%, top 5\%, and top 1\% "
        r"of citation impact, respectively. Publications AMEs are expressed in publication-count "
        r"units and represent changes in the "
        r"expected publication count, whereas OLS coefficients for the P indicators represent "
        r"changes in fractionally counted numbers of highly cited publications. AMEs represent "
        r"average derivatives for continuous predictors and average discrete differences for "
        r"categorical predictors. OLS $p$-values are based on $t$-tests, whereas negative "
        r"binomial AME $p$-values are based on $z$-tests. "
    )
    note += table_context_note(columns)
    note += r"Significance levels: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. "
    note += (
        r"Definitions and calculations of the analytical variables are provided in "
        r"Table~\ref{tab:analytical_variables}."
    )
    return note


def table_postamble(
    *,
    model_id: str,
    outcomes: tuple[str, ...] = RESEARCH_PERFORMANCE_OUTCOMES,
    performance_type: str = "relative",
    model_types: dict[str, str] | None = None,
    outcome_aliases: Mapping[str, str] | None = None,
    columns: list[LatexModelColumn] | None = None,
) -> list[str]:
    if columns is not None:
        notes = performance_table_note(
            columns=columns,
            outcomes=outcomes,
            performance_type=performance_type,
            model_id=model_id,
            outcome_aliases=outcome_aliases,
        )
        return [
            "",
            r"\bottomrule",
            r"\end{tabular}%",
            "}",
            "",
            r"\vspace{2mm}",
            "",
            r"\begin{minipage}{\textwidth}",
            r"\footnotesize",
            notes,
            r"\end{minipage}",
            "",
            r"\end{table}",
        ]
    model_number = model_id[1:] if model_id.lower().startswith("m") else model_id
    if performance_type == "relative":
        outcome_note = relative_performance_note(
            outcomes=outcomes,
            outcome_aliases=outcome_aliases,
        )
        if set((model_types or {}).values()).intersection({"frac", "ppml", "nb"}):
            outcome_note = outcome_note.replace(
                "Coefficients and standard errors", "AMEs and robust standard errors"
            )
    else:
        outcome_note = absolute_performance_note(
            outcomes=outcomes,
            model_types=model_types or {},
            outcome_aliases=outcome_aliases,
        )
    inference_note = model_inference_note(
        {outcome: (model_types or {}).get(outcome, "ols") for outcome in outcomes}
    )
    outcome_labels = ", ".join(outcome_label(outcome, outcome_aliases) for outcome in outcomes)
    absolute_nb_ols_hc3_specification = (
        performance_type == "absolute"
        and (model_types or {}).get(N_PUBLIC) == "nb"
        and set(ABSOLUTE_TOP_PERCENTILE_OUTCOMES).issubset(outcomes)
        and all(
            (model_types or {}).get(outcome) == "olshc3"
            for outcome in ABSOLUTE_TOP_PERCENTILE_OUTCOMES
        )
    )
    if performance_type == "relative" and set((model_types or {}).values()) == {"olshc3"}:
        notes = (
            r"\textit{Notes.} Reported are coefficients (Coef.), robust standard errors "
            r"(Robust SE), and $p$-values ($P$). The dependent variables are the mean "
            r"weighted percentile rank (mwPR) and the proportions of publications in the "
            r"top 10\%, top 5\%, and top 1\% (PP(top 10\%), PP(top 5\%), and "
            r"PP(top 1\%), respectively). Coefficients and standard errors for mwPR are "
            r"reported in percentile points; those for the PP indicators are reported in "
            r"percentage points. Significance levels: $^{*}p<0.05$, $^{**}p<0.01$, "
            r"$^{***}p<0.001$. Academic-title abbreviations are defined in "
            r"Table~\ref{tab:prof_titles}. Definitions and calculations of the "
            r"analytical variables are provided in Table~\ref{tab:analytical_variables}. "
            r"Full Prof.\ is the reference category for academic title. OLS $p$-values are "
            r"based on $t$-tests; HC3 standard errors are heteroskedasticity-robust. ASJC "
            r"field composition is represented by fractional publication shares across 27 "
            r"ASJC major fields, with Medicine as the reference field because it has the "
            r"largest overall publication share."
        )
    elif absolute_nb_ols_hc3_specification:
        notes = (
            r"\textit{Notes.} Reported are average marginal effects (AMEs) or coefficients "
            r"(Coef.), robust standard errors (Robust SE), and $p$-values ($P$). The dependent "
            r"variables are the total number of publications (Publications) and the fractionally "
            r"counted numbers of publications in the top 10\%, top 5\%, and top 1\% "
            r"(P(top 10\%), P(top 5\%), and P(top 1\%), respectively). Total publication "
            r"output is "
            r"estimated using negative binomial regression, with AMEs expressed in publication-"
            r"count units. AMEs represent average changes in predicted publication counts, "
            r"calculated as discrete differences for categorical variables and derivatives for "
            r"continuous variables. The three highly cited publication indicators are estimated "
            r"using ordinary least squares (OLS), with coefficients expressed in fractional "
            r"publication-count units. All models use HC3 heteroskedasticity-robust standard "
            r"errors. Negative binomial AME standard errors are calculated using the delta method, "
            r"with $p$-values based on $z$-tests; OLS $p$-values are based on $t$-tests. "
            r"Significance levels: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. Academic-title "
            r"abbreviations are defined in Table~\ref{tab:prof_titles}. Definitions and "
            r"calculations of the analytical variables are provided in "
            r"Table~\ref{tab:analytical_variables}. Full Prof.\ is the reference category for "
            r"academic title. ASJC field composition is represented by fractional publication "
            r"shares across 27 ASJC major fields, with Medicine as the reference field because it "
            r"has the largest overall publication share."
        )
    else:
        selected_types = {(model_types or {}).get(outcome, "ols") for outcome in outcomes}
        nonlinear_types = selected_types.intersection({"frac", "ppml", "nb"})
        linear_types = selected_types - {"frac", "ppml", "nb"}
        if nonlinear_types and linear_types:
            effect_description = (
                "estimator-specific coefficients or average marginal effects (AMEs), their "
                "standard errors, and $p$-values"
            )
        elif nonlinear_types:
            effect_description = (
                "average marginal effects (AMEs), robust standard errors, and $p$-values"
            )
        else:
            effect_description = "estimated coefficients, standard errors, and $p$-values"
        nonlinear_note = ""
        if nonlinear_types:
            nonlinear_note = (
                " For nonlinear estimators, continuous effects are average derivatives and "
                "categorical effects are average discrete changes versus the reference category. "
                "Fractional-logit AMEs are restored to the original outcome scale; PPML and NB "
                "AMEs use outcome units. HC3 robust standard errors and delta-method inference "
                "are used for these AMEs."
            )
        if nonlinear_types and linear_types:
            estimator_descriptions = []
            for outcome in outcomes:
                model_type = (model_types or {}).get(outcome, "ols")
                effect_name = "AMEs" if model_type in {"frac", "ppml", "nb"} else "coefficients"
                estimator_descriptions.append(
                    f"{outcome_label(outcome, outcome_aliases)} uses {model_type} ({effect_name})"
                )
            nonlinear_note += " Estimators by outcome: " + "; ".join(estimator_descriptions) + "."
        if "nb" in selected_types and performance_type == "absolute":
            nonlinear_note += (
                " Negative-binomial estimation of fractional P(top x%) counts uses nearest-integer "
                "half-up rounding where such fractional outcomes are configured."
            )
        notes = (
            r"\textit{Notes.} Reported are "
            f"{effect_description}. {outcome_note}Outcomes: {outcome_labels}."
            f"{nonlinear_note} "
            r"Significance levels: $^{*}p<0.05$, $^{**}p<0.01$, "
            r"$^{***}p<0.001$. Academic-title abbreviations are defined in "
            r"Table~\ref{tab:prof_titles}. Definitions and calculations of the "
            r"analytical variables are provided in Table~\ref{tab:analytical_variables}. "
            r"Full Prof.\ is the reference category for academic "
            f"title. {inference_note} "
            r"ASJC field composition is represented by "
            r"fractional publication shares across 27 ASJC major fields, with Medicine "
            r"as the reference field because it has the largest overall publication share. "
            f"{'Effects' if nonlinear_types else 'Coefficients'} "
            f"shown are from Model~{model_number}."
        )
    return [
        "",
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        "",
        r"\vspace{2mm}",
        "",
        r"\begin{minipage}{\textwidth}",
        r"\footnotesize",
        notes,
        r"\end{minipage}",
        "",
        r"\end{table}",
    ]


def absolute_performance_note(
    *,
    outcomes: tuple[str, ...],
    model_types: dict[str, str],
    outcome_aliases: Mapping[str, str] | None = None,
) -> str:
    notes = []
    if N_PUBLIC in outcomes:
        count_model_type = model_types.get(N_PUBLIC, "ols")
        if count_model_type == "nb":
            notes.append(
                "The number of publications is estimated using negative binomial regression "
                "and average marginal effects are reported in publication-count units."
            )
        elif count_model_type == "zinb":
            notes.append(
                "The number of publications is estimated using zero-inflated negative "
                "binomial regression; count-component coefficients are on the log "
                "expected-count scale."
            )
        elif count_model_type == "pr":
            notes.append(
                "The number of publications is estimated using Poisson regression; its "
                "coefficients are expressed on the log expected-count scale."
            )
        else:
            notes.append(
                "The number of publications is estimated using OLS; its coefficients are "
                "expressed on the original outcome scale."
            )
    percentile_outcomes = tuple(
        outcome
        for outcome in outcomes
        if outcome in {ABSOLUTE_P_TOP_10, ABSOLUTE_P_TOP_5, ABSOLUTE_P_TOP_1}
    )
    if percentile_outcomes:
        labels = ", ".join(
            outcome_label(outcome, outcome_aliases) for outcome in percentile_outcomes
        )
        if all(
            model_types.get(outcome, "ols") in OLS_MODEL_TYPES for outcome in percentile_outcomes
        ):
            notes.append(
                f"{labels} are fractional counts of highly cited publications and are "
                "estimated using OLS; their coefficients are expressed on the original "
                "outcome scale."
            )
        else:
            for outcome in percentile_outcomes:
                model_type = model_types.get(outcome, "ols")
                label = outcome_label(outcome, outcome_aliases)
                if model_type == "nb":
                    notes.append(
                        f"{label} is reported as an average marginal effect after nearest-integer "
                        "half-up rounding of the fractional outcome for negative-binomial "
                        "estimation."
                    )
                elif model_type == "ppml":
                    notes.append(
                        f"{label} is reported as an average marginal effect in original fractional "
                        "publication-count units; values are not rounded."
                    )
                else:
                    notes.append(f"{label} is estimated using its configured model family.")
    return " ".join(notes) + (" " if notes else "")


def relative_performance_note(
    *,
    outcomes: tuple[str, ...],
    outcome_aliases: Mapping[str, str] | None,
) -> str:
    outcome_labels = {outcome: outcome_label(outcome, outcome_aliases) for outcome in outcomes}
    if HAZEN_PERCENTILE in outcomes:
        other_outcomes = [outcome for outcome in outcomes if outcome in PERCENTAGE_POINT_OUTCOMES]
        if other_outcomes:
            labels = join_labels(tuple(outcome_labels[outcome] for outcome in other_outcomes))
            return (
                f"Coefficients and standard errors for {outcome_labels[HAZEN_PERCENTILE]} "
                f"are reported in percentile points; those for {labels} are reported in "
                "percentage points. "
            )
        return (
            f"Coefficients and standard errors for {outcome_labels[HAZEN_PERCENTILE]} "
            "are reported in percentile points. "
        )
    percentage_outcomes = [outcome for outcome in outcomes if outcome in PERCENTAGE_POINT_OUTCOMES]
    if percentage_outcomes:
        labels = join_labels(tuple(outcome_labels[outcome] for outcome in percentage_outcomes))
        return f"Coefficients and standard errors for {labels} are reported in percentage points. "
    return "Coefficients and standard errors are reported on the configured outcome scales. "


def model_inference_note(model_types: dict[str, str]) -> str:
    model_families = set(model_types.values())
    notes = []
    if model_families.intersection(OLS_MODEL_TYPES):
        if "olshc3" in model_families:
            notes.append(
                "OLS uses conventional covariance and OLS-HC3 uses HC3 heteroskedasticity-robust "
                "standard errors; OLS $p$-values are based on $t$-tests"
            )
        else:
            notes.append("OLS uses conventional standard errors and $t$-test inference")
    if "nb" in model_families:
        notes.append("negative-binomial inference uses $z$-tests and HC3 robust standard errors")
    if model_families.intersection({"pr", "zinb"}):
        notes.append("count-model $p$-values are based on $z$-tests")
    if "huber" in model_families:
        notes.append(
            "Huber RLM uses HuberT with the H1 covariance estimator; $p$-values use the existing "
            "$z$-test inference"
        )
    if model_families.intersection({"frac", "ppml"}):
        notes.append("GLM AME inference uses delta-method $z$-tests and HC3 robust standard errors")
    if "frac" in model_families:
        notes.append(
            "fractional-logit AMEs are average derivatives for continuous predictors and "
            "average discrete changes for categorical predictors, on the original outcome scale"
        )
    if "ppml" in model_families:
        notes.append(
            "PPML AMEs use original outcome units; fractional publication counts are preserved"
        )
    if "nb" in model_families:
        notes.append("negative-binomial tables report average marginal effects")
    if not notes:
        return ""
    return "; ".join(notes) + "."


def latex_row(cells: list[str]) -> str:
    return "\n".join([cells[0], *[f"& {cell}" for cell in cells[1:]]]) + f" {LATEX_ROW_END}"


def format_coefficient_cells(coefficient: LatexCoefficient | None) -> list[str]:
    if coefficient is None:
        return ["--", "--", "--"]
    return [
        f"{coefficient.estimate:.2f}",
        f"{coefficient.std_error:.2f}",
        format_p_value_cell(coefficient.p_value),
    ]


def format_p_value_cell(value: float) -> str:
    formatted = f"{value:.2f}"
    stars = significance_code(value)
    if stars == ".":
        stars = ""
    return f"{formatted}\\textsuperscript{{{stars}}}" if stars else formatted


def title_rows_for_columns(columns: list[LatexModelColumn]) -> list[str]:
    available = set().union(*(set(title_categories(column.model.model_data)) for column in columns))
    return [title for title in TITLE_ORDER if title in available]


def title_categories(data: pd.DataFrame) -> tuple[str, ...]:
    if "title" not in data:
        return ()
    series = data["title"]
    if hasattr(series.dtype, "categories"):
        return tuple(str(category) for category in series.cat.categories)
    return tuple(str(value) for value in sorted(series.dropna().unique()))


def title_contrast_weights(fit: Any, *, title: str, reference: str) -> np.ndarray:
    params = list(fit.params.index)
    weights = np.zeros(len(params), dtype=float)
    add_title_weight(weights=weights, params=params, title=title, weight=1.0)
    add_title_weight(weights=weights, params=params, title=reference, weight=-1.0)
    return weights


def add_title_weight(
    *,
    weights: np.ndarray,
    params: list[str],
    title: str,
    weight: float,
) -> None:
    term = f"C(title)[T.{title}]"
    if term in params:
        weights[params.index(term)] += weight


def latex_title_label(title: str) -> str:
    return title.replace(". ", r".\ ")


def renderable_model_comparison_outcomes(
    models: dict[tuple[str, str], FittedRegressionModel],
    *,
    model_ids: tuple[str, ...] = MODEL_COMPARISON_MODEL_IDS,
    outcomes: tuple[str, ...] = (
        *RESEARCH_PERFORMANCE_OUTCOMES,
        *ABSOLUTE_PERFORMANCE_OUTCOMES,
    ),
) -> tuple[str, ...]:
    """Return outcomes with all model specifications needed for comparison tables."""
    available_outcomes = {outcome for outcome, _ in models}
    return tuple(
        outcome
        for outcome in dict.fromkeys(outcomes)
        if outcome in available_outcomes
        and all((outcome, model_id) in models for model_id in model_ids)
    )


def split_decimal(value: float) -> tuple[str, str]:
    formatted = f"{value:.2f}"
    whole, decimal = formatted.split(".", maxsplit=1)
    return whole, decimal


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def latex_label_slug(value: str) -> str:
    return value.replace("_", "")


def model_comparison_preamble(
    *,
    dependent_variable: str,
    model_ids: tuple[str, ...],
    sample_size: int | None = None,
    outcome_aliases: Mapping[str, str] | None = None,
) -> list[str]:
    label = outcome_label(dependent_variable, outcome_aliases)
    sample_note = f" (\\(n={latex_integer(sample_size)}\\))" if sample_size is not None else ""
    return [
        r"\providecommand{\coef}[3]{#1 & .#2\textsuperscript{#3}}",
        r"\providecommand{\coefn}[2]{#1 & .#2}",
        r"\providecommand{\se}[2]{(#1 & .#2)}",
        r"\providecommand{\na}{\multicolumn{2}{c}{--}}",
        r"\providecommand{\txt}[1]{\multicolumn{2}{c}{#1}}",
        r"\providecommand{\fit}[1]{\multicolumn{2}{c}{\scriptsize #1}}",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        (
            rf"\caption{{Association between academic inbreeding and {label} "
            rf"across regression model specifications{sample_note}}}"
        ),
        rf"\label{{tab:supp_{latex_label_slug(dependent_variable)}_models}}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{@{{}}p{{2.8cm}}*{{{len(model_ids)}}}{{r@{{}}l}}@{{}}}}",
        r"\toprule",
        model_comparison_header(model_ids),
    ]


def model_comparison_header(model_ids: tuple[str, ...]) -> str:
    cells = ["{}"] + [
        rf"\multicolumn{{2}}{{c}}{{\textbf{{{model_id.upper()}}}}}" for model_id in model_ids
    ]
    return model_comparison_row(cells)


def model_comparison_effect_header(columns: list[LatexModelColumn]) -> str:
    """Label each model column with its reported coefficient or AME scale.

    :param columns: Outcome-specific model columns in table order.
    :return: LaTeX row identifying coefficient versus AME entries per model.
    """
    cells = ["{}"]
    for column in columns:
        effect_heading = "AME" if column.model.model_type in {"frac", "ppml", "nb"} else "Coef."
        cells.append(rf"\multicolumn{{2}}{{c}}{{\textit{{{effect_heading}}}}}")
    return model_comparison_row(cells)


def model_comparison_row(cells: list[str], *, row_spacing: str = "") -> str:
    return " & ".join(cells) + f" {LATEX_ROW_END}{row_spacing}"


def append_comparison_coefficient_rows(
    lines: list[str],
    label: str,
    columns: list[LatexModelColumn],
    *,
    term_name: str,
) -> None:
    coefficients = [column.term(term_name) for column in columns]
    append_comparison_rows(lines, label, coefficients)


def append_comparison_title_rows(
    lines: list[str],
    title: str,
    columns: list[LatexModelColumn],
) -> None:
    coefficients = [column.title_contrast(title) for column in columns]
    append_comparison_rows(lines, rf"\quad {latex_title_label(title)}", coefficients)


def append_comparison_rows(
    lines: list[str],
    label: str,
    coefficients: list[LatexCoefficient | None],
) -> None:
    lines.append(
        model_comparison_row([label, *[format_comparison_estimate(coef) for coef in coefficients]])
    )
    lines.append(
        model_comparison_row(
            ["{}", *[format_comparison_std_error(coef) for coef in coefficients]],
            row_spacing=r"[2pt]",
        )
    )


def format_comparison_estimate(coefficient: LatexCoefficient | None) -> str:
    if coefficient is None:
        return r"\na"
    stars = significance_code(coefficient.p_value)
    if stars == ".":
        stars = ""
    whole, decimal = split_decimal(coefficient.estimate)
    return (
        rf"\coef{{{whole}}}{{{decimal}}}{{{stars}}}"
        if stars
        else rf"\coefn{{{whole}}}{{{decimal}}}"
    )


def format_comparison_std_error(coefficient: LatexCoefficient | None) -> str:
    if coefficient is None:
        return r"\na"
    whole, decimal = split_decimal(coefficient.std_error)
    return rf"\se{{{whole}}}{{{decimal}}}"


def model_comparison_inclusion_row(label: str, columns: list[LatexModelColumn]) -> str:
    cells = [label]
    for column in columns:
        value = "Yes" if column.model.share_control_type else "No"
        cells.append(rf"\txt{{{value}}}")
    return model_comparison_row(cells)


def model_comparison_adjusted_r_squared_row(columns: list[LatexModelColumn]) -> str:
    cells = [r"Adjusted $R^2$"]
    for column in columns:
        value = getattr(column.model.fit, "rsquared_adj", np.nan)
        formatted = f"{float(value):.2f}" if np.isfinite(value) else "---"
        cells.append(rf"\txt{{{formatted}}}")
    return model_comparison_row(cells)


def model_comparison_aic_row(columns: list[LatexModelColumn]) -> str:
    cells = ["AIC"]
    for column in columns:
        cells.append(rf"\fit{{{float(column.model.fit_statistics['aic']):,.0f}}}")
    return model_comparison_row(cells)


def model_comparison_bic_row(columns: list[LatexModelColumn]) -> str:
    cells = ["BIC"]
    for column in columns:
        cells.append(rf"\fit{{{float(column.model.fit_statistics['bic']):,.0f}}}")
    return model_comparison_row(cells)


def model_comparison_observations_row(columns: list[LatexModelColumn]) -> str:
    """Render per-specification sample sizes when they differ.

    :param columns: Fitted model columns ordered as they appear in the table.
    :return: LaTeX observations row.
    """
    cells = ["Observations"]
    cells.extend(
        rf"\multicolumn{{2}}{{c}}{{{latex_integer(int(column.model.fit.nobs))}}}"
        for column in columns
    )
    return model_comparison_row(cells)


def model_comparison_estimator_row(columns: list[LatexModelColumn]) -> str:
    """Render the estimator used by each specification.

    :param columns: Fitted model columns ordered as they appear in the table.
    :return: LaTeX estimator row.
    """
    cells = ["Estimator"]
    cells.extend(
        rf"\multicolumn{{2}}{{c}}{{{estimator_description(column.model.model_type)}}}"
        for column in columns
    )
    return model_comparison_row(cells)


def model_comparison_postamble(
    *,
    dependent_variable: str,
    columns: list[LatexModelColumn],
    outcome_aliases: Mapping[str, str] | None = None,
) -> list[str]:
    performance_type = (
        "relative"
        if dependent_variable == HAZEN_PERCENTILE or dependent_variable in PERCENTAGE_POINT_OUTCOMES
        else "absolute"
    )
    requested_ols_note = dependent_variable in {
        HAZEN_PERCENTILE,
        *PERCENTAGE_POINT_OUTCOMES,
        *ABSOLUTE_TOP_PERCENTILE_OUTCOMES,
    } and is_hc3_ols_comparison(columns)
    requested_nb_note = dependent_variable == N_PUBLIC and is_nb_comparison(columns)
    if requested_ols_note:
        notes = model_comparison_relative_ols_note(
            columns=columns,
            dependent_variable=dependent_variable,
            outcome_aliases=outcome_aliases,
        )
    elif requested_nb_note:
        notes = model_comparison_n_public_nb_note(columns)
    else:
        notes = performance_table_note(
            columns=columns,
            outcomes=(dependent_variable,),
            performance_type=performance_type,
            model_id="comparison",
            outcome_aliases=outcome_aliases,
            has_p_column=False,
        )
    if not requested_ols_note and not requested_nb_note:
        spec_labels = [column.model.model_id.upper() for column in columns]
        spec_description = (
            f"{spec_labels[0]}–{spec_labels[-1]}"
            if len(spec_labels) > 1 and spec_labels == [f"M{index}" for index in range(1, 11)]
            else join_labels(tuple(spec_labels))
        )
        notes += (
            f"{spec_description} denote different regression specifications. "
            "Dashes indicate terms omitted from a specification. "
        )
        notes += (
            "AIC denotes the Akaike information criterion; BIC denotes the Bayesian information "
            "criterion. Lower AIC and BIC indicate a preferable balance between model fit and "
            "complexity for comparable models fitted to the same outcome and sample. "
        )
    return [
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        r"\vspace{2mm}",
        r"\begin{minipage}{\textwidth}",
        r"\scriptsize",
        (notes),
        r"\end{minipage}",
        r"\end{table}",
    ]


def is_hc3_ols_comparison(columns: list[LatexModelColumn]) -> bool:
    """Return whether every comparison model is OLS with HC3 covariance.

    :param columns: Fitted models displayed in the comparison table.
    :return: Whether the requested OLS/HC3 note accurately describes the models.
    """
    return bool(columns) and all(
        column.model.model_type in OLS_MODEL_TYPES
        and str(getattr(column.model.fit, "cov_type", "")).upper() == "HC3"
        for column in columns
    )


def is_nb_comparison(columns: list[LatexModelColumn]) -> bool:
    """Return whether every comparison model uses negative binomial regression.

    :param columns: Fitted models displayed in the comparison table.
    :return: Whether all models use the NB estimator.
    """
    return bool(columns) and all(column.model.model_type == "nb" for column in columns)


def model_comparison_n_public_nb_note(columns: list[LatexModelColumn]) -> str:
    """Render notes for total-publication negative-binomial comparisons.

    :param columns: Fitted models displayed in the comparison table.
    :return: LaTeX note describing NB average marginal effects and inference.
    """
    sample_sizes = {int(column.model.fit.nobs) for column in columns}
    if len(sample_sizes) == 1:
        sample_note = f"$N={latex_integer(next(iter(sample_sizes)))}$ in all models. "
    else:
        sample_note = "Sample sizes differ across models; see the Observations row. "
    spec_labels = [column.model.model_id.upper() for column in columns]
    spec_description = (
        f"{spec_labels[0]}--{spec_labels[-1]}"
        if len(spec_labels) > 1 and spec_labels == [f"M{index}" for index in range(1, 11)]
        else join_labels(tuple(spec_labels))
    )
    return (
        r"\textit{Notes.} "
        + sample_note
        + r"Reported are average marginal effects (AMEs) from negative binomial regression "
        r"(NB). The dependent variable is the number of publications per professor "
        r"(Publications). Publications AMEs are expressed in publication-count units and "
        r"represent changes in the expected number of publications. "
        r"HC3 heteroskedasticity-robust standard errors (Robust SE) are calculated using the "
        r"delta method; p-values are based on $z$-tests. Full Professor (Full Prof.) is the "
        r"reference category for academic title; academic-title abbreviations are defined in "
        r"Table~1. Research-field composition is represented by fractional publication shares "
        r"across 27 major fields of the All Science Journal Classification (ASJC), with "
        r"Medicine as the reference field because it has the largest overall publication share. "
        r"Significance levels: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. Definitions and "
        rf"calculations of the analytical variables are provided in Table~3. {spec_description} "
        r"denote different regression specifications. Dashes indicate terms omitted from a "
        r"specification. AIC denotes the Akaike information criterion; BIC denotes the "
        r"Bayesian information criterion. Lower AIC and BIC indicate a preferable balance "
        r"between model fit and complexity for comparable models fitted to the same outcome "
        r"and sample."
    )


def model_comparison_relative_ols_note(
    *,
    columns: list[LatexModelColumn],
    dependent_variable: str,
    outcome_aliases: Mapping[str, str] | None,
) -> str:
    """Render OLS model-comparison notes for publication-impact outcomes.

    :param columns: Fitted models displayed in the comparison table.
    :param dependent_variable: Outcome represented by the table.
    :param outcome_aliases: Optional display labels for outcomes.
    :return: LaTeX note with the sample size and model interpretation.
    """
    sample_sizes = {int(column.model.fit.nobs) for column in columns}
    if len(sample_sizes) == 1:
        sample_note = f"$N={latex_integer(next(iter(sample_sizes)))}$ in all models. "
    else:
        sample_note = "Sample sizes differ across models; see the Observations row. "
    if dependent_variable == HAZEN_PERCENTILE:
        dependent_variable_note = (
            r"The dependent variable is the mean weighted percentile rank (mwPR). "
            r"mwPR effects are expressed in percentile points. "
        )
    elif dependent_variable in ABSOLUTE_TOP_PERCENTILE_OUTCOMES:
        label = outcome_label(dependent_variable, outcome_aliases)
        percentile = {
            ABSOLUTE_P_TOP_1: "1",
            ABSOLUTE_P_TOP_5: "5",
            ABSOLUTE_P_TOP_10: "10",
        }[dependent_variable]
        dependent_variable_note = (
            rf"The dependent variable, {label}, is the sum of publication-level fractional "
            rf"contributions to the top {percentile}\% most cited publications in their "
            r"respective fields and publication years, calculated for each professor. "
            rf"{label} coefficients are expressed in fractional publication-count units. "
        )
    else:
        label = outcome_label(dependent_variable, outcome_aliases)
        dependent_variable_note = (
            rf"The dependent variable is the proportion of publications in the respective "
            rf"citation-impact percentile ({label}). PP denotes proportions of publications "
            r"in the respective citation-impact percentiles; their effects are expressed in "
            r"percentage points. "
        )
    spec_labels = [column.model.model_id.upper() for column in columns]
    spec_description = (
        f"{spec_labels[0]}--{spec_labels[-1]}"
        if len(spec_labels) > 1 and spec_labels == [f"M{index}" for index in range(1, 11)]
        else join_labels(tuple(spec_labels))
    )
    return (
        r"\textit{Notes.} "
        + sample_note
        + r"Reported are regression coefficients (Coef.) from ordinary least squares (OLS). "
        + dependent_variable_note
        + r"HC3 heteroskedasticity-robust standard errors (Robust SE); p-values are based on "
        r"t-tests. Full "
        r"Professor (Full Prof.) is the reference category for academic title; academic-title "
        r"abbreviations are defined in Table~1. Research-field composition is represented by "
        r"fractional publication shares across 27 major fields of the All Science Journal "
        r"Classification (ASJC), with Medicine as the reference field because it has the "
        r"largest overall publication share. Significance levels: $^{*}p<0.05$, "
        r"$^{**}p<0.01$, $^{***}p<0.001$. Definitions and calculations of the analytical "
        rf"variables are provided in Table~3. {spec_description} denote different regression "
        r"specifications. Dashes indicate terms omitted from a specification. "
        + r"AIC denotes the Akaike information criterion; BIC denotes the Bayesian information "
        r"criterion. Lower AIC and BIC indicate a preferable balance between model fit and "
        r"complexity for comparable models fitted to the same outcome and sample."
    )


def model_comparison_latex_table(
    models: dict[tuple[str, str], FittedRegressionModel],
    *,
    dependent_variable: str,
    model_ids: tuple[str, ...] = MODEL_COMPARISON_MODEL_IDS,
    outcome_aliases: Mapping[str, str] | None = None,
) -> str:
    """Render one supplementary table comparing model specifications for one outcome."""
    columns = [
        LatexModelColumn(
            dependent_variable=dependent_variable,
            model=models[(dependent_variable, model_id)],
        )
        for model_id in model_ids
    ]
    lines = model_comparison_preamble(
        dependent_variable=dependent_variable,
        model_ids=model_ids,
        sample_size=int(columns[0].model.fit.nobs) if columns else None,
        outcome_aliases=outcome_aliases,
    )
    if any(column.model.model_type in {"frac", "ppml", "nb"} for column in columns) or (
        dependent_variable
        not in {
            HAZEN_PERCENTILE,
            *PERCENTAGE_POINT_OUTCOMES,
            *ABSOLUTE_TOP_PERCENTILE_OUTCOMES,
            N_PUBLIC,
        }
    ):
        lines.append(model_comparison_effect_header(columns))
    lines.append(r"\midrule")
    append_comparison_coefficient_rows(
        lines,
        "Inbreeding",
        columns,
        term_name="C(is_inbred)[T.True]",
    )
    append_comparison_coefficient_rows(
        lines,
        "Academic age",
        columns,
        term_name="academic_age",
    )
    append_comparison_coefficient_rows(
        lines,
        r"Academic age$^2$",
        columns,
        term_name="I(academic_age ** 2)",
    )
    lines.append(r"\addlinespace[4pt]")
    lines.append(
        rf"\multicolumn{{{len(columns) * 2 + 1}}}{{@{{}}l}}"
        r"{\textit{Academic title (ref.: Full Prof.)}} \\[1pt]"
    )
    for title in title_rows_for_columns(columns):
        append_comparison_title_rows(lines, title, columns)
    lines.append(r"\addlinespace[3pt]")
    append_comparison_coefficient_rows(
        lines,
        "Total mobility",
        columns,
        term_name="n_away_years",
    )
    append_comparison_coefficient_rows(
        lines,
        "Domestic mobility",
        columns,
        term_name="n_away_years_domestic",
    )
    append_comparison_coefficient_rows(
        lines,
        "International mobility",
        columns,
        term_name="n_away_years_abroad",
    )
    append_comparison_coefficient_rows(
        lines,
        "Co-affiliation publication share",
        columns,
        term_name="coaffiliation_publication_share",
    )
    append_comparison_coefficient_rows(
        lines,
        "Mean coauthors per publication",
        columns,
        term_name="mean_coauthors_per_publication",
    )
    append_comparison_coefficient_rows(
        lines,
        "First publication year",
        columns,
        term_name="first_year",
    )
    lines.append(r"\addlinespace[4pt]")
    lines.append(model_comparison_inclusion_row("ASJC field shares", columns))
    lines.append(r"\midrule")
    if all(column.model.model_type in OLS_MODEL_TYPES for column in columns):
        lines.append(model_comparison_adjusted_r_squared_row(columns))
    lines.append(model_comparison_aic_row(columns))
    lines.append(model_comparison_bic_row(columns))
    if len({int(column.model.fit.nobs) for column in columns}) > 1:
        lines.append(model_comparison_observations_row(columns))
    lines.extend(
        model_comparison_postamble(
            dependent_variable=dependent_variable,
            columns=columns,
            outcome_aliases=outcome_aliases,
        )
    )
    return "\n".join(lines)
