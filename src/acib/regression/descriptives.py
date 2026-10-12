# Copyright © 2026 TU Wien

"""Descriptive-statistics LaTeX report for the regression analytical sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

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
    ContinuousVariable("hazen_perc_med", r"mwPR [\%]", digits=1),
    ContinuousVariable("pp_top_10", r"PP(top 10\%) [\%]", 100.0, 1),
    ContinuousVariable("pp_top_5", r"PP(top 5\%) [\%]", 100.0, 1),
    ContinuousVariable("pp_top_1", r"PP(top 1\%) [\%]", 100.0, 1),
)
DESCRIPTIVE_ABSOLUTE_OUTCOME_VARIABLES = (
    ContinuousVariable("n_public", "Publications [count]", digits=0),
    ContinuousVariable("P_top10", r"P(top 10\%) [count]", digits=1),
    ContinuousVariable("P_top5", r"P(top 5\%) [count]", digits=1),
    ContinuousVariable("P_top1", r"P(top 1\%) [count]", digits=1),
)
DESCRIPTIVE_EXPLANATORY_VARIABLES = (
    ContinuousVariable("academic_age", "Academic age [years]", digits=0),
    ContinuousVariable("coaffiliation_publication_share", r"Co-affil. pub. share [\%]", 100.0, 1),
    ContinuousVariable("mean_coauthors_per_publication", "Mean coauthors per pub.", digits=1),
    ContinuousVariable(
        "n_away_years_domestic", "Domestic mobility [years]", digits=0, mean_sd_digits=1
    ),
    ContinuousVariable(
        "n_away_years_abroad", "International mobility [years]", digits=0, mean_sd_digits=1
    ),
    ContinuousVariable("first_year", "First publication year", digits=0),
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
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\begin{minipage}{\textwidth}",
            r"\footnotesize",
            (
                r"\textit{Notes.} Continuous variables are reported as mean, standard deviation "
                r"(SD), minimum, and maximum."
            ),
            (
                r"mwPR is reported in percentile points; PP and co-affiliation shares are "
                r"reported as percentages."
            ),
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
    label: str, non_inbred_value: str, inbred_value: str, *, row_end: str = LATEX_ROW_END
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
