# Copyright © 2026 TU Wien

"""Descriptive-statistics LaTeX report for the regression analytical sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from acib.constants import (
    ABSOLUTE_P_TOP_1,
    ABSOLUTE_P_TOP_5,
    ABSOLUTE_P_TOP_10,
    ACADEMIC_AGE,
    FIRST_YEAR,
    HAZEN_PERCENTILE,
    N_PUBLIC,
    PP_TOP_1,
    PP_TOP_5,
    PP_TOP_10,
)
from acib.regression.constants import ABSOLUTE_PERFORMANCE_OUTCOMES
from acib.regression.model import FittedRegressionModel

LATEX_ROW_END = r"\\"


@dataclass(frozen=True, slots=True)
class ContinuousVariable:
    """Describe a continuous table row and its displayed scale."""

    column: str
    label: str
    scale: float = 1.0
    digits: int = 2
    mean_sd_digits: int | None = None
    range_digits: int | None = None


DESCRIPTIVE_OUTCOME_VARIABLES = (
    ContinuousVariable(
        HAZEN_PERCENTILE,
        "Mean weighted percentile rank (mwPR) [percentile points]",
        digits=1,
    ),
    ContinuousVariable(PP_TOP_10, r"PP(top 10\%) [percentage points]", 100.0, 1),
    ContinuousVariable(PP_TOP_5, r"PP(top 5\%) [percentage points]", 100.0, 1),
    ContinuousVariable(PP_TOP_1, r"PP(top 1\%) [percentage points]", 100.0, 1),
)
DESCRIPTIVE_ABSOLUTE_OUTCOME_VARIABLES = (
    ContinuousVariable(N_PUBLIC, "Publications [count]", digits=0),
    ContinuousVariable(ABSOLUTE_P_TOP_10, r"P(top 10\%) [fractional publication counts]", digits=1),
    ContinuousVariable(ABSOLUTE_P_TOP_5, r"P(top 5\%) [fractional publication counts]", digits=1),
    ContinuousVariable(ABSOLUTE_P_TOP_1, r"P(top 1\%) [fractional publication counts]", digits=1),
)
DESCRIPTIVE_EXPLANATORY_VARIABLES = (
    ContinuousVariable(ACADEMIC_AGE, "Academic age [years]", digits=0),
    ContinuousVariable(
        "coaffiliation_publication_share",
        "Co-affiliation publication share [percentage points]",
        100.0,
        1,
    ),
    ContinuousVariable("mean_coauthors_per_publication", "Mean coauthors", digits=1),
    ContinuousVariable(
        "n_away_years_domestic", "Domestic mobility [years]", digits=0, mean_sd_digits=1
    ),
    ContinuousVariable(
        "n_away_years_abroad",
        "International mobility [years]",
        digits=0,
        mean_sd_digits=1,
    ),
    ContinuousVariable(FIRST_YEAR, "First publication year", digits=0),
)
TITLE_ORDER = ("Full Prof.", "Ao. Prof.", "Ass. Prof.", "FH-Prof.", "FH-Ass. Prof.")


def descriptive_statistics_sample(
    *,
    data: pd.DataFrame,
    models: dict[tuple[str, str], FittedRegressionModel] | None,
    model_id: str,
    absolute_performance_outcomes: tuple[str, ...] = ABSOLUTE_PERFORMANCE_OUTCOMES,
) -> pd.DataFrame:
    """Use complete selected-model cases for a common comparison sample.

    :param data: Prepared input frame.
    :param models: Fitted suite models, if available.
    :param model_id: Model identifier defining the analytical cases.
    :param absolute_performance_outcomes: Absolute outcome columns in scope.
    :return: Input rows complete for all eligible selected-model outcomes.
    """
    if not models:
        return data.copy()
    outcomes = {item.column for item in DESCRIPTIVE_OUTCOME_VARIABLES} | set(
        absolute_performance_outcomes
    )
    frames = [
        model.model_data
        for (outcome, key), model in models.items()
        if key == model_id and outcome in outcomes
    ]
    if not frames:
        return data.copy()
    index = frames[0].index
    for frame in frames[1:]:
        index = index.intersection(frame.index)
    return data.loc[index].copy()


def descriptive_statistics_latex_table(
    data: pd.DataFrame,
    *,
    models: dict[tuple[str, str], FittedRegressionModel] | None = None,
    model_id: str = "m9",
    absolute_performance_outcomes: tuple[str, ...] = ABSOLUTE_PERFORMANCE_OUTCOMES,
) -> str:
    """Render descriptive statistics by academic-inbreeding status.

    :param data: Prepared regression data.
    :param models: Optional fitted models used to determine the analytical sample.
    :param model_id: Model identifying complete analytical cases.
    :param absolute_performance_outcomes: Absolute outcome columns to render.
    :return: Complete LaTeX table source.
    """
    sample = descriptive_statistics_sample(
        data=data,
        models=models,
        model_id=model_id,
        absolute_performance_outcomes=absolute_performance_outcomes,
    )
    variables = (
        *DESCRIPTIVE_OUTCOME_VARIABLES,
        *(
            v
            for v in DESCRIPTIVE_ABSOLUTE_OUTCOME_VARIABLES
            if v.column in absolute_performance_outcomes and v.column in sample
        ),
        *DESCRIPTIVE_EXPLANATORY_VARIABLES,
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            rf"\caption{{Descriptive statistics of the professor-level analytical "
            rf"sample (\(n={latex_count(len(sample))}\))}}"
        ),
        r"\label{tab:descriptive_statistics}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.15}",
        (
            r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X "
            r"*{8}{>{\raggedleft\arraybackslash}p{0.07\textwidth}}@{}}"
        ),
        r"\toprule",
        (
            rf"\textbf{{Variable}} & \multicolumn{{4}}{{c}}{{\textbf{{Non-inbred}}}} "
            rf"& \multicolumn{{4}}{{c}}{{\textbf{{Inbred}}}} {LATEX_ROW_END}"
        ),
        (
            r" & \textbf{Mean} & \textbf{SD} & \textbf{Min} & \textbf{Max} "
            r"& \textbf{Mean} & \textbf{SD} & \textbf{Min} & \textbf{Max} \\"
        ),
        r"\midrule",
        grouped_value_row(
            r"Professors, \(n\)",
            format_integer(len(group_sample(sample, False))),
            format_integer(len(group_sample(sample, True))),
        ),
    ]
    for variable in variables:
        if variable.column in sample:
            lines.append(
                latex_row(
                    [
                        variable.label,
                        *format_continuous_stats(
                            group_sample(sample, False)[variable.column], variable
                        ),
                        *format_continuous_stats(
                            group_sample(sample, True)[variable.column], variable
                        ),
                    ]
                )
            )
    if "title" in sample:
        lines.append(grouped_value_row(r"Academic title, \(n\) (\%)", "", "", row_end=r"\\[2pt]"))
        titles = [title for title in TITLE_ORDER if (sample["title"] == title).any()]
        titles.extend(
            sorted(title for title in sample.title.dropna().unique() if title not in TITLE_ORDER)
        )
        for title in titles:
            lines.append(
                grouped_value_row(
                    rf"\quad {latex_escape(title)}",
                    format_count_percent(group_sample(sample, False), title),
                    format_count_percent(group_sample(sample, True), title),
                )
            )
    note_parts = [
        (
            r"\textit{Notes.} Continuous variables are summarized by mean, standard deviation "
            r"(SD), minimum, and maximum."
        )
    ]
    if HAZEN_PERCENTILE in sample:
        note_parts.append(
            "The mean weighted percentile rank (mwPR) is reported in percentile points."
        )
    if any(column in sample for column in (PP_TOP_10, PP_TOP_5, PP_TOP_1)):
        note_parts.append(
            r"PP(top 10\%), PP(top 5\%), and PP(top 1\%) are proportions of publications "
            "in the respective citation-impact percentiles, displayed in percentage points."
        )
    if any(column in sample for column in ABSOLUTE_PERFORMANCE_OUTCOMES):
        note_parts.append(
            r"Publications is the publication count; P(top 10\%), P(top 5\%), and "
            r"P(top 1\%) are fractionally counted highly cited publication counts."
        )
    if "title" in sample:
        note_parts.append(
            r"Full Prof. is the reference category for academic title; all academic-title "
            r"abbreviations are defined in Table~\ref{tab:prof_titles}."
        )
    note_parts.append(
        r"Definitions and calculations of the analytical variables are provided in "
        r"Table~\ref{tab:analytical_variables}."
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\begin{minipage}{\textwidth}",
            r"\footnotesize",
            " ".join(note_parts),
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def group_sample(sample: pd.DataFrame, is_inbred: bool) -> pd.DataFrame:
    """Select one inbreeding group, returning empty when unavailable."""
    if "is_inbred" not in sample:
        return sample.iloc[0:0]
    return sample.loc[sample.is_inbred.map(normalize_bool) == is_inbred]


def normalize_bool(value: Any) -> bool | None:
    """Parse the boolean spellings used by regression input data."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    return True if text in {"true", "1", "yes"} else False if text in {"false", "0", "no"} else None


def format_continuous_stats(series: pd.Series, variable: ContinuousVariable) -> list[str]:
    """Format mean, sample SD, minimum, and maximum for one variable."""
    values = pd.to_numeric(series, errors="coerce").dropna() * variable.scale
    if values.empty:
        return ["--"] * 4
    digits = variable.mean_sd_digits if variable.mean_sd_digits is not None else variable.digits
    range_digits = variable.range_digits if variable.range_digits is not None else variable.digits
    return [
        format_number(values.mean(), digits),
        format_number(values.std(), digits) if len(values) > 1 else "--",
        format_number(values.min(), range_digits),
        format_number(values.max(), range_digits),
    ]


def format_count_percent(sample: pd.DataFrame, title: str) -> str:
    """Format the count and within-group percentage for an academic title."""
    count = int((sample.title == title).sum()) if not sample.empty and "title" in sample else 0
    return (
        f"{format_integer(count)} ({format_number(count / len(sample) * 100, 0)})"
        if len(sample)
        else "0 (0)"
    )


def latex_row(cells: list[str], *, row_end: str = LATEX_ROW_END) -> str:
    """Join cells into a LaTeX table row."""
    return " & ".join(cells) + f" {row_end}"


def grouped_value_row(
    label: str,
    non_inbred_value: str,
    inbred_value: str,
    *,
    row_end: str = LATEX_ROW_END,
) -> str:
    """Render values spanning each four-column inbreeding group."""
    return (
        f"{label} & \\multicolumn{{4}}{{l}}{{{non_inbred_value}}} & "
        f"\\multicolumn{{4}}{{l}}{{{inbred_value}}} {row_end}"
    )


def format_number(value: float, digits: int) -> str:
    """Format a number with the configured decimal precision."""
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if digits == 0 and "." in text else text


def format_integer(value: int) -> str:
    """Format an integer with thousands separators."""
    return f"{value:,}"


def latex_count(value: int) -> str:
    """Format a count using LaTeX-safe thousands separators."""
    return format_integer(value).replace(",", r"{,}")


def latex_escape(value: object) -> str:
    """Escape ampersand and percent characters in LaTeX text."""
    return str(value).replace("&", r"\&").replace("%", r"\%")
