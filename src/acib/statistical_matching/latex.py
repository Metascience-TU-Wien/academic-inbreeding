# Copyright © 2026 TU Wien

"""LaTeX tables for configured relative and absolute matching outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from acib.constants import N_PUBLIC, PRIMARY_FIELD
from acib.regression.constants import DEFAULT_REGRESSION_OUTCOME_METADATA
from acib.statistical_matching.config import StatisticalMatchingConfig
from acib.statistical_matching.constants import (
    ABSOLUTE_P_OUTCOMES,
    DEFAULT_METHODS,
    DEFAULT_OUTCOMES,
    LATEX_FILENAME_PREFIX,
    LATEX_FILENAME_SUFFIX,
    PP_OUTCOMES,
)

LATEX_ROW_END = r"\\"
COVARIATE_LABELS = {
    "academic_age": "academic age",
    "title": "academic title",
    PRIMARY_FIELD: "primary field",
    "first_year": "first publication year",
    "coaffiliation_publication_share": "co-affil. pub. share",
    "mean_coauthors_per_publication": "mean coauthors per publication",
    "n_away_years_domestic": "domestic mobility",
    "n_away_years_abroad": "international mobility",
}
METHOD_LABELS = {"PSM": "PSM", "IPTW": "IPTW", "CEM": "CEM", "EB": "EB"}
METHOD_NOTES = {
    "PSM": "PSM = propensity score matching",
    "IPTW": "IPTW = inverse probability of treatment weighting",
    "CEM": "CEM = coarsened exact matching",
    "EB": "EB = entropy balancing",
}


@dataclass(frozen=True, slots=True)
class MatchingLatexCell:
    """Rendered model statistics for one outcome/method cell."""

    estimate: float
    std_error: float
    conf_low: float
    conf_high: float
    p_value: float
    stars: str


def latex_tables(effects: pd.DataFrame, config: StatisticalMatchingConfig) -> dict[str, str]:
    """Render relative/absolute family tables in selected outcome order."""
    tables = {}
    for family in ("relative", "absolute"):
        outcomes = tuple(
            outcome for outcome in config.outcomes if config.outcome_types.get(outcome) == family
        )
        if outcomes:
            filename = f"{LATEX_FILENAME_PREFIX}{family}{LATEX_FILENAME_SUFFIX}"
            tables[filename] = statistical_matching_latex_table(
                effects,
                covariates=config.covariates,
                covariance_type=config.covariance_type,
                outcome_order=outcomes,
                outcome_aliases=config.outcome_aliases,
                outcome_type=family,
            )
    return tables


def statistical_matching_latex_table(
    outcome_effects: pd.DataFrame,
    *,
    covariates: tuple[str, ...],
    covariance_type: str,
    method_order: tuple[str, ...] = DEFAULT_METHODS,
    outcome_order: tuple[str, ...] = DEFAULT_OUTCOMES,
    outcome_aliases: dict[str, str] | None = None,
    outcome_type: str | None = None,
) -> str:
    """Build one source-format family table with ATT, inference, and notes."""
    methods = methods_for_table(outcome_effects, method_order)
    outcomes = outcomes_for_table(outcome_effects, outcome_order=outcome_order)
    table_family = outcome_type or "relative"
    aliases = outcome_aliases or {
        outcome: metadata[1] for outcome, metadata in DEFAULT_REGRESSION_OUTCOME_METADATA.items()
    }
    lines = table_preamble(outcome_type=table_family)
    lines.extend(method_header(methods))
    lines.extend((r"\midrule", ""))
    rows = outcome_blocks(outcome_effects, methods=methods, outcomes=outcomes, aliases=aliases)
    for index, row in enumerate(rows):
        lines.append(row)
        if index < len(rows) - 1 and row.startswith(r"\addlinespace"):
            lines.append("")
    lines.append("")
    lines.extend(
        table_postamble(
            covariates=covariates,
            covariance_type=covariance_type,
            outcomes=outcomes,
            outcome_type=table_family,
        )
    )
    return "\n".join(lines)


def table_preamble(*, outcome_type: str) -> list[str]:
    """Return source caption, label, and table-layout opening."""
    if outcome_type == "relative":
        family_label = "relative citation impact"
        table_label = "tab:matching_results_relative"
    else:
        family_label = "absolute publication output"
        table_label = f"tab:matching_results_{outcome_type}"
    return [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            rf"\caption{{Association between academic inbreeding and {family_label} "
            r"after statistical matching and weighting}"
        ),
        rf"\label{{{table_label}}}",
        "",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{8pt}",
        "",
        r"\begin{tabular*}{0.84\textwidth}{",
        r"@{\extracolsep{\fill}}",
        "l",
        *[r"S[table-format=-1.3,table-space-text-post={***}]"] * 4,
        r"@{}",
        "}",
        "",
        r"\toprule",
    ]


def method_header(methods: list[str]) -> list[str]:
    """Format ordered method headings."""
    cells = [
        rf"\multicolumn{{1}}{{c}}{{\textbf{{{METHOD_LABELS.get(m, latex_escape(m))}}}}}"
        for m in methods
    ]
    return [" &\n".join(["", *cells]) + f" {LATEX_ROW_END}", ""]


def outcome_blocks(
    effects: pd.DataFrame, *, methods: list[str], outcomes: list[str], aliases: dict[str, str]
) -> list[str]:
    """Render all statistics for each configured outcome."""
    rows = []
    statistics = (r"\quad ATT", r"\quad Std.\ err.", r"\quad \(P{>}|t|\)", r"\quad 95\% CI")
    for index, outcome in enumerate(outcomes):
        if index:
            rows.append(r"\addlinespace[5pt]")
        rows.append(outcome_heading_row(outcome, methods=methods, aliases=aliases))
        rows.extend(
            statistic_row(effects, outcome=outcome, methods=methods, statistic=stat)
            for stat in statistics
        )
    return rows


def outcome_heading_row(outcome: str, *, methods: list[str], aliases: dict[str, str]) -> str:
    """Format a bold outcome label."""
    alias = aliases.get(outcome, latex_escape(outcome))
    return (
        rf"\multicolumn{{{len(methods) + 1}}}{{@{{}}l}}{{\textbf{{{alias}}}}}"
        rf" {LATEX_ROW_END}[1pt]"
    )


def statistic_row(
    effects: pd.DataFrame, *, outcome: str, methods: list[str], statistic: str
) -> str:
    """Format a single model statistic across ordered methods."""
    cells = [statistic]
    for method in methods:
        cell = latex_cell(effects, method=method, outcome=outcome)
        cells.append("" if cell is None else format_statistic_cell(cell, statistic=statistic))
    return "\n& ".join(cells) + f" {LATEX_ROW_END}"


def format_statistic_cell(cell: MatchingLatexCell, *, statistic: str) -> str:
    """Format estimate, robust SE, p-value, or confidence interval."""
    if statistic == r"\quad ATT":
        return format_number(cell.estimate)
    if statistic == r"\quad Std.\ err.":
        return format_number(cell.std_error)
    if statistic == r"\quad 95\% CI":
        return format_centered_cell(format_ci_cell(cell.conf_low, cell.conf_high))
    return format_p_value_cell(cell.p_value, cell.stars)


def latex_cell(effects: pd.DataFrame, *, method: str, outcome: str) -> MatchingLatexCell | None:
    """Read one method/outcome row into a typed display record."""
    rows = effects.loc[effects["method"].eq(method) & effects["outcome"].eq(outcome)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return MatchingLatexCell(
        float(row["estimate_display"]),
        float(row["std_error_display"]),
        float(row["conf_low_display"]),
        float(row["conf_high_display"]),
        float(row["p_value"]),
        str(row.get("stars", "")),
    )


def table_postamble(
    *,
    covariates: tuple[str, ...],
    covariance_type: str,
    outcomes: list[str],
    outcome_type: str,
) -> list[str]:
    """Close the table with source explanatory notes."""
    return [
        r"\bottomrule",
        r"\end{tabular*}",
        "",
        r"\vspace{2mm}",
        "",
        r"\begin{minipage}{\textwidth}",
        r"\footnotesize",
        notes(
            covariates=covariates,
            covariance_type=covariance_type,
            outcomes=outcomes,
            outcome_type=outcome_type,
        ),
        r"\end{minipage}",
        "",
        r"\end{table}",
    ]


def notes(
    *,
    covariates: tuple[str, ...],
    covariance_type: str,
    outcomes: list[str],
    outcome_type: str,
) -> str:
    """Render source family, unit, estimator, and significance notes."""
    units = []
    if any(value in PP_OUTCOMES for value in outcomes):
        units.append(r"PP outcomes are reported in percentage points.")
    if N_PUBLIC in outcomes:
        units.append("Publication output is reported as publication counts.")
    if any(value in ABSOLUTE_P_OUTCOMES for value in outcomes):
        units.append(r"P outcomes are fractional publication counts.")
    family_note = (
        "Negative ATT estimates indicate lower research performance among academically "
        "inbred professors than among non-inbred professors. "
        if outcome_type == "relative"
        else "Negative ATT estimates indicate lower absolute publication output among "
        "academically inbred professors than among non-inbred professors. "
    )
    return (
        rf"\textit{{Notes.}} Reported are average treatment effects on the treated (ATTs), "
        rf"heteroskedasticity-robust {covariance_type} standard errors (Std.\ err.), "
        r"\(p\)-values (\(P\)), and 95\% confidence intervals (CIs). "
        f"{family_note}{' '.join(units)} "
        r"Professors were balanced on academic age, academic title, first "
        r"publication year, and fractional publication shares across the five broad ASJC "
        r"areas (Physical Sciences, Life Sciences, Health Sciences, Social Sciences and "
        r"Humanities, and Multidisciplinary). Physical Sciences was omitted as the "
        r"reference area from the propensity-score specification but was included in the "
        r"balance diagnostics. PSM = propensity score matching; IPTW = inverse probability "
        r"of treatment weighting; CEM = coarsened exact matching; EB = entropy balancing. "
        r"Significance levels: \(^{*}p<0.05\), \(^{**}p<0.01\), "
        r"\(^{***}p<0.001\)."
    )


def methods_for_table(effects: pd.DataFrame, order: tuple[str, ...]) -> list[str]:
    """Return present method identifiers in configured-first order."""
    available = list(dict.fromkeys(effects["method"].astype(str)))
    return [
        *(method for method in order if method in available),
        *(method for method in available if method not in order),
    ]


def outcomes_for_table(effects: pd.DataFrame, *, outcome_order: tuple[str, ...]) -> list[str]:
    """Return present outcomes in configured order."""
    available = set(effects["outcome"].astype(str))
    return [outcome for outcome in outcome_order if outcome in available]


def covariate_phrase(covariates: tuple[str, ...]) -> str:
    """Format configured covariate labels as a phrase."""
    labels = [COVARIATE_LABELS.get(value, latex_escape(value)) for value in covariates]
    if not labels:
        return "the configured covariates"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def method_notes() -> str:
    """Return the source method glossary."""
    return "; ".join(METHOD_NOTES[method] for method in DEFAULT_METHODS)


def format_number(value: float) -> str:
    """Format finite values at two decimal places."""
    return "" if math.isnan(value) else f"{value:.2f}"


def format_ci_cell(conf_low: float, conf_high: float) -> str:
    """Format confidence limits at two decimal places."""
    if math.isnan(conf_low) or math.isnan(conf_high):
        return ""
    return rf"[{conf_low:.2f}, {conf_high:.2f}]"


def format_centered_cell(value: str) -> str:
    """Center a nonempty value in one LaTeX table cell."""
    return "" if not value else rf"\multicolumn{{1}}{{c}}{{{value}}}"


def format_p_value_cell(p_value: float, stars: str) -> str:
    """Keep source two-decimal p-value display and significance stars."""
    value = f"{p_value:.2f}"
    return rf"{value}\textsuperscript{{{stars}}}" if stars else value


def latex_escape(value: str) -> str:
    """Escape common LaTeX-special characters."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)
