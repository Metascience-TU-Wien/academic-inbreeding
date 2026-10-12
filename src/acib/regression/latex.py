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
    )
    lines.extend(outcome_header(outcomes, outcome_aliases=outcome_aliases))
    lines.append(r"\midrule")
    lines.append("")
    lines.append(coefficient_row("Inbreeding", columns, term_name="C(is_inbred)[T.True]"))
    lines.append("")
    lines.append(coefficient_row(r"Acad.\ age", columns, term_name="academic_age"))
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
            r"Co-affil.\ pub.\ share",
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
    lines.append(coefficient_row(r"Int.\ mobility", columns, term_name="n_away_years_abroad"))
    lines.append("")
    lines.append(coefficient_row(r"First pub.\ year", columns, term_name="first_year"))
    lines.append("")
    lines.append(r"\addlinespace[3pt]")
    lines.append("")
    lines.extend(discipline_control_rows(columns))
    lines.append("")
    lines.append(r"\midrule")
    lines.append("")
    lines.append(observations_row(columns))
    lines.append("")
    lines.append(adjusted_r_squared_row(columns))
    lines.extend(
        table_postamble(
            model_id=model_id,
            outcomes=outcomes,
            performance_type=performance_type,
            outcome_aliases=outcome_aliases,
            model_types={column.dependent_variable: column.model.model_type for column in columns},
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
    sample_size = int(columns[0].model.fit.nobs)
    caption = (
        r"\caption{Association between academic inbreeding and research performance in the "
        rf"full specification excluding international mobility (\(n={sample_size:,}\))}}"
    ).replace(",", "{,}")
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
        *["S[table-format=-1.2,table-space-text-post={***}]" for _ in outcomes],
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
        r"\midrule",
        "",
    ]

    estimates = [column.term("C(is_inbred)[T.True]") for column in columns]
    lines.extend(
        [
            latex_row(["Estimate", *[format_model10_estimate(value) for value in estimates]]),
            "",
            latex_row(
                ["Std.\\ err.", *[format_model10_standard_error(value) for value in estimates]]
            ),
            "",
            latex_row([r"$P$", *[format_model10_p_value(value) for value in estimates]]),
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
                relative_outcomes=relative_outcomes,
                absolute_outcomes=absolute_outcomes,
                model_types={
                    column.dependent_variable: column.model.model_type for column in columns
                },
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


def model10_without_international_mobility_note(
    *,
    relative_outcomes: tuple[str, ...],
    absolute_outcomes: tuple[str, ...],
    model_types: dict[str, str],
    outcome_aliases: Mapping[str, str] | None = None,
) -> str:
    fractional_absolute_labels = join_labels(
        tuple(
            outcome_label(outcome, outcome_aliases)
            for outcome in absolute_outcomes
            if outcome != N_PUBLIC
        )
    )
    has_publication_count = N_PUBLIC in absolute_outcomes
    is_negative_binomial = model_types.get(N_PUBLIC) == "nb"
    other_outcomes_are_ols = all(
        model_types.get(outcome, "ols") == "ols"
        for outcome in (*relative_outcomes, *absolute_outcomes)
        if outcome != N_PUBLIC
    )
    if has_publication_count and is_negative_binomial and other_outcomes_are_ols:
        publication_model_note = (
            "The number of publications is estimated using negative binomial regression and "
            "its estimate is expressed on the log expected-count scale; all other outcomes "
            "are estimated using OLS."
        )
    elif has_publication_count and is_negative_binomial:
        publication_model_note = (
            "The number of publications is estimated using negative binomial regression and "
            "its estimate is expressed on the log expected-count scale; other outcomes use "
            "their configured model families."
        )
    else:
        publication_model_note = "Outcomes use their configured model families."
    percentage_outcomes = [
        outcome for outcome in relative_outcomes if outcome in {"pp_top_10", "pp_top_5", "pp_top_1"}
    ]
    has_mwpr = HAZEN_PERCENTILE in relative_outcomes
    percentage_labels = join_labels(
        tuple(outcome_label(outcome, outcome_aliases) for outcome in percentage_outcomes)
    )
    if has_mwpr and percentage_outcomes:
        relative_note_sentence = (
            r"Estimates and standard errors for mwPR are reported in percentile points; "
            f"those for {percentage_labels} are reported in percentage points. "
        )
    elif has_mwpr:
        relative_note_sentence = (
            "Estimates and standard errors for mwPR are reported in percentile points. "
        )
    elif percentage_outcomes:
        relative_note_sentence = (
            f"Estimates and standard errors for {percentage_labels} are reported in "
            "percentage points. "
        )
    else:
        relative_note_sentence = ""
    fractional_count_note = (
        f"{fractional_absolute_labels} are fractional publication counts. "
        if fractional_absolute_labels
        else ""
    )
    return (
        r"\textit{Notes.} Reported are the estimated associations between academic inbreeding "
        r"and each research-performance outcome from the full specification excluding "
        r"international mobility. The specification adjusts for academic age, academic title, "
        r"domestic mobility, coaffiliation publication share, mean number of coauthors per "
        r"publication, first publication year, and fractional publication shares across 27 "
        r"ASJC major fields. "
        + relative_note_sentence
        + fractional_count_note
        + publication_model_note
        + " "
        r"Significance levels: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$."
    )


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
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Robustness regressions for fractional counts of top-cited "
            r"publications (Model 9)}"
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
        absolute_top_percentile_outcome_header(outcome_aliases=outcome_aliases),
        r"\midrule",
        "",
        coefficient_row("Inbreeding", columns, term_name="C(is_inbred)[T.True]"),
        "",
        coefficient_row(r"Acad.\ age", columns, term_name="academic_age"),
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
                r"Co-affil.\ pub.\ share",
                columns,
                term_name="coaffiliation_publication_share",
            ),
            "",
            coefficient_row("Mean coauthors", columns, term_name="mean_coauthors_per_publication"),
            "",
            coefficient_row("Domestic mobility", columns, term_name="n_away_years_domestic"),
            "",
            coefficient_row(r"Int.\ mobility", columns, term_name="n_away_years_abroad"),
            "",
            coefficient_row(r"First pub.\ year", columns, term_name="first_year"),
            "",
            r"\addlinespace[3pt]",
            "",
            latex_row(
                [
                    "ASJC field shares",
                    r"\multicolumn{9}{c}{27 major fields; Medicine as reference}",
                ]
            ),
            "",
            r"\midrule",
            "",
            observations_row(columns),
            "",
            adjusted_r_squared_row(columns),
            *absolute_top_percentile_table_postamble(model_id=model_id),
        ]
    )
    return "\n".join(lines)


def absolute_top_percentile_outcome_header(
    *,
    outcome_aliases: Mapping[str, str] | None = None,
) -> str:
    labels = [
        outcome_label(outcome, outcome_aliases) for outcome in ABSOLUTE_TOP_PERCENTILE_OUTCOMES
    ]
    groups = [rf"\multicolumn{{3}}{{c}}{{\textbf{{{label}}}}}" for label in labels]
    header = latex_row(["{}", *groups])
    subheader = latex_row(
        [
            r"\textbf{Variable}",
            *[
                header_cell
                for _ in labels
                for header_cell in (
                    r"\multicolumn{1}{c}{\textbf{Coef.}}",
                    r"\multicolumn{1}{c}{\shortstack{\textbf{Robust}\\\textbf{std. err.}}}",
                    r"\multicolumn{1}{c}{\textbf{$P$}}",
                )
            ],
        ]
    )
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


def absolute_top_percentile_table_postamble(*, model_id: str) -> list[str]:
    model_number = model_id[1:] if model_id.lower().startswith("m") else model_id
    return [
        "",
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        "",
        r"\vspace{2mm}",
        r"\begin{minipage}{\textwidth}",
        r"\scriptsize",
        (
            r"\textit{Notes.} Outcomes are sums of publication-level fractional indicators "
            r"(\texttt{ptop10\_frac}, \texttt{ptop5\_frac}, \texttt{ptop1\_frac}) for "
            r"each professor; fractional contributions are retained without rounding, and "
            r"each professor-publication pair is counted once using the existing regression "
            r"sample filters. Entries are coefficients with heteroskedasticity-robust "
            r"standard errors in parentheses. Significance levels: $^{*}p<0.05$, "
            r"$^{**}p<0.01$, $^{***}p<0.001$. Full Prof.\ is the reference title; "
            f"estimates are from Model~{model_number}."
        ),
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
) -> list[str]:
    label_prefix = (
        "research_performance" if performance_type == "relative" else "absolute_performance"
    )
    caption_suffix = (
        "research performance" if performance_type == "relative" else "absolute performance"
    )
    columns = [
        value
        for _ in outcomes
        for value in (
            "S[table-format=-1.2]",
            "S[table-format=1.2]",
            "S[table-format=1.2,table-space-text-post={***}]",
        )
    ]
    return [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Association between academic inbreeding and {caption_suffix}}}",
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


def outcome_header(
    outcomes: tuple[str, ...],
    *,
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
    for _ in outcomes:
        columns.extend(
            [
                r"\multicolumn{1}{c}{\textbf{Coef.}}",
                r"\multicolumn{1}{c}{\shortstack{\textbf{Std.}\\\textbf{err.}}}",
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
        cells.append(rf"\multicolumn{{3}}{{c}}{{{int(column.model.fit.nobs):,}}}")
    return latex_row(cells)


def adjusted_r_squared_row(columns: list[LatexModelColumn]) -> str:
    cells = [r"Adj.\ $R^2$"]
    for column in columns:
        value = getattr(column.model.fit, "rsquared_adj", np.nan)
        formatted = f"{float(value):.2f}" if np.isfinite(value) else r"\na"
        cells.append(rf"\multicolumn{{3}}{{c}}{{{formatted}}}")
    return latex_row(cells)


def table_postamble(
    *,
    model_id: str,
    outcomes: tuple[str, ...] = RESEARCH_PERFORMANCE_OUTCOMES,
    performance_type: str = "relative",
    model_types: dict[str, str] | None = None,
    outcome_aliases: Mapping[str, str] | None = None,
) -> list[str]:
    model_number = model_id[1:] if model_id.lower().startswith("m") else model_id
    if performance_type == "relative":
        outcome_note = relative_performance_note(
            outcomes=outcomes,
            outcome_aliases=outcome_aliases,
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
        (
            r"\textit{Notes.} Reported are coefficients (Coef.), standard errors "
            r"(Std.\ err.), and $p$-values ($P$). "
            f"{outcome_note}Outcomes: {outcome_labels}. "
            r"Significance levels: $^{*}p<0.05$, $^{**}p<0.01$, "
            r"$^{***}p<0.001$. Full Prof.\ is the reference category for academic "
            f"title. {inference_note} "
            r"ASJC field composition is represented by "
            r"fractional publication shares across 27 ASJC major fields, with Medicine "
            r"as the reference field because it has the largest overall publication share. "
            f"Coefficients shown are from Model~{model_number}."
        ),
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
                "because publication output is a non-negative count variable and exhibits "
                "substantial overdispersion. Its coefficients are therefore expressed on "
                "the log expected-count scale."
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
        if all(model_types.get(outcome, "ols") == "ols" for outcome in percentile_outcomes):
            notes.append(
                f"{labels} are fractional counts of highly cited publications and are "
                "estimated using OLS; their coefficients are expressed on the original "
                "outcome scale."
            )
        else:
            notes.append(
                f"{labels} are fractional counts of highly cited publications; each "
                "outcome is estimated using its configured model family."
            )
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
    if "ols" in model_families:
        notes.append("OLS $p$-values are based on $t$-tests")
    if model_families.intersection({"pr", "nb", "zinb"}):
        notes.append("count-model $p$-values are based on $z$-tests")
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
    formatted = "0.00" if value < 0.001 else f"{value:.3f}"
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
    outcome_aliases: Mapping[str, str] | None = None,
) -> list[str]:
    label = outcome_label(dependent_variable, outcome_aliases)
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
            r"across regression model specifications}"
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
        value = "Yes" if column.model.share_control_type == "detailed_asjc_shares" else "No"
        cells.append(rf"\txt{{{value}}}")
    return model_comparison_row(cells)


def model_comparison_adjusted_r_squared_row(columns: list[LatexModelColumn]) -> str:
    cells = [r"Adj.\ $R^2$"]
    for column in columns:
        value = getattr(column.model.fit, "rsquared_adj", np.nan)
        formatted = f"{float(value):.2f}" if np.isfinite(value) else r"\na"
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


def model_comparison_postamble(
    *,
    dependent_variable: str,
    columns: list[LatexModelColumn],
    outcome_aliases: Mapping[str, str] | None = None,
) -> list[str]:
    sample_size = f"{int(columns[0].model.fit.nobs):,}".replace(",", "{,}")
    model_types = {column.model.model_type for column in columns}
    covariance_types = {
        str(getattr(column.model.fit, "cov_type", "nonrobust")) for column in columns
    }
    if dependent_variable == N_PUBLIC and model_types == {"nb"}:
        label = outcome_label(dependent_variable, outcome_aliases)
        outcome_note = f"{label} coefficients are on the log expected-count scale."
    elif dependent_variable in ABSOLUTE_TOP_PERCENTILE_OUTCOMES:
        outcome_note = (
            r"Outcomes are unrounded sums of publication-level fractional top-x\% "
            r"indicators for each professor."
        )
    elif dependent_variable == HAZEN_PERCENTILE:
        outcome_note = "Coefficients and standard errors are reported in percentile points."
    else:
        outcome_note = "Coefficients and standard errors are reported in percentage points."
    if len(covariance_types) == 1 and "nonrobust" in covariance_types:
        standard_error_note = "model-based standard errors in parentheses. "
    else:
        standard_error_note = "heteroskedasticity-robust standard errors in parentheses. "
    inference_note = model_inference_note(
        {column.model.model_id: column.model.model_type for column in columns}
    )
    return [
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        r"\vspace{2mm}",
        r"\begin{minipage}{\textwidth}",
        r"\scriptsize",
        (
            rf"\textit{{Notes.}} $N={sample_size}$ in all models. Entries are coefficients "
            f"with {standard_error_note}"
            rf"{outcome_note} Full Prof.\ is the reference category for academic title. "
            f"{inference_note} "
            r"ASJC field composition is represented by fractional publication shares across "
            r"27 ASJC major fields, with Medicine as the reference field. Dashes indicate "
            r"that a term was not included in the respective model specification. "
            r"AIC = Akaike Information Criterion; BIC = Bayesian Information Criterion. "
            r"Significance levels: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$."
        ),
        r"\end{minipage}",
        r"\end{table}",
    ]


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
        outcome_aliases=outcome_aliases,
    )
    lines.append(r"\midrule")
    append_comparison_coefficient_rows(
        lines,
        "Inbreeding",
        columns,
        term_name="C(is_inbred)[T.True]",
    )
    append_comparison_coefficient_rows(
        lines,
        r"Acad.\ age",
        columns,
        term_name="academic_age",
    )
    append_comparison_coefficient_rows(
        lines,
        r"Acad.\ age$^2$",
        columns,
        term_name="I(academic_age ** 2)",
    )
    lines.append(r"\addlinespace[4pt]")
    lines.append(
        rf"\multicolumn{{{len(columns) * 2 + 1}}}{{@{{}}l}}"
        r"{\textit{Academic title (ref.: Full Prof.)}} \[1pt]"
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
        r"Int.\ mobility",
        columns,
        term_name="n_away_years_abroad",
    )
    append_comparison_coefficient_rows(
        lines,
        r"Co-affil.\ pub.\ share",
        columns,
        term_name="coaffiliation_publication_share",
    )
    append_comparison_coefficient_rows(
        lines,
        "Mean coauthors",
        columns,
        term_name="mean_coauthors_per_publication",
    )
    append_comparison_coefficient_rows(
        lines,
        r"First pub.\ year",
        columns,
        term_name="first_year",
    )
    lines.append(r"\addlinespace[4pt]")
    lines.append(model_comparison_inclusion_row("ASJC field shares", columns))
    lines.append(r"\midrule")
    if all(column.model.model_type == "ols" for column in columns):
        lines.append(model_comparison_adjusted_r_squared_row(columns))
    lines.append(model_comparison_aic_row(columns))
    lines.append(model_comparison_bic_row(columns))
    lines.extend(
        model_comparison_postamble(
            dependent_variable=dependent_variable,
            columns=columns,
            outcome_aliases=outcome_aliases,
        )
    )
    return "\n".join(lines)
