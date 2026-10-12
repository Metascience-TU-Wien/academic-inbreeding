# Copyright © 2026 TU Wien

"""Fixed-width console reporting for statistical-matching effects."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from acib.constants import N_PUBLIC
from acib.regression.constants import DEFAULT_REGRESSION_OUTCOME_METADATA
from acib.statistical_matching.constants import (
    ABSOLUTE_P_OUTCOMES,
    DEFAULT_METHODS,
    DEFAULT_OUTCOMES,
    PP_OUTCOMES,
)


@dataclass(frozen=True, slots=True)
class ConsoleColumn:
    """One heading and width in the result table."""

    header: str
    width: int


def statistical_matching_console_table(
    outcome_effects: pd.DataFrame,
    *,
    covariance_type: str,
    method_order: tuple[str, ...] = DEFAULT_METHODS,
    outcomes: tuple[str, ...] = DEFAULT_OUTCOMES,
    aliases: dict[str, str] | None = None,
) -> str:
    """Format the source fixed-width ATT table, notes, and significance key."""
    selected_outcomes = outcomes_for_table(outcome_effects, outcomes=outcomes)
    selected_aliases = aliases or {
        outcome: metadata[1] for outcome, metadata in DEFAULT_REGRESSION_OUTCOME_METADATA.items()
    }
    columns = table_columns(selected_outcomes, aliases=selected_aliases)
    lines = [
        f"Statistical matching / weighting results (ATT, {covariance_type} SEs)",
        "",
        separator(columns),
        header_row(columns),
        separator(columns),
    ]
    for method in methods_for_table(outcome_effects, method_order):
        lines.append(
            method_row(outcome_effects, method=method, columns=columns, outcomes=selected_outcomes)
        )
    notes = []
    if any(outcome in PP_OUTCOMES for outcome in selected_outcomes):
        notes.append("PP outcomes are reported in percentage points.")
    if N_PUBLIC in selected_outcomes:
        notes.append("Publications are reported as publication counts.")
    if any(outcome in ABSOLUTE_P_OUTCOMES for outcome in selected_outcomes):
        notes.append("P outcomes are unrounded sums of fractional publication counts.")
    lines.extend(
        [separator(columns), " ".join(notes) + " Stars: * p<0.05, ** p<0.01, *** p<0.001."]
    )
    return "\n".join(lines)


def table_columns(outcomes: list[str], *, aliases: dict[str, str]) -> list[ConsoleColumn]:
    """Create method/outcome columns with source widths."""
    columns = [ConsoleColumn("Method", 6)]
    for outcome in outcomes:
        label = aliases.get(outcome, outcome).replace(r"\%", "%")
        columns.extend(
            [
                ConsoleColumn(f"{label} ATT", max(len(label) + 4, 10)),
                ConsoleColumn("SE", 6),
                ConsoleColumn("p", 10),
            ]
        )
    return columns


def header_row(columns: list[ConsoleColumn]) -> str:
    """Render aligned headings."""
    return "| " + " | ".join(column.header.ljust(column.width) for column in columns) + " |"


def method_row(
    outcome_effects: pd.DataFrame,
    *,
    method: str,
    columns: list[ConsoleColumn],
    outcomes: list[str],
) -> str:
    """Render one method's estimates in configured outcome order."""
    values = [method]
    for outcome in outcomes:
        rows = outcome_effects[
            outcome_effects["method"].eq(method) & outcome_effects["outcome"].eq(outcome)
        ]
        values.extend(["", "", ""] if rows.empty else formatted_cells(rows.iloc[0]))
    return (
        "| "
        + " | ".join(format_cell(value, column) for value, column in zip(values, columns))
        + " |"
    )


def formatted_cells(row: pd.Series) -> list[str]:
    """Format estimate, standard error, and p-value cells."""
    return [
        format_number(float(row["estimate_display"])),
        format_number(float(row["std_error_display"])),
        format_p_value(float(row["p_value"]), str(row.get("stars", ""))),
    ]


def separator(columns: list[ConsoleColumn]) -> str:
    """Render a table-width separator."""
    return "+-" + "-+-".join("-" * column.width for column in columns) + "-+"


def methods_for_table(effects: pd.DataFrame, order: tuple[str, ...]) -> list[str]:
    """Order configured methods first, then any observed methods."""
    available = list(dict.fromkeys(effects["method"].astype(str)))
    return [
        *(method for method in order if method in available),
        *(method for method in available if method not in order),
    ]


def outcomes_for_table(effects: pd.DataFrame, *, outcomes: tuple[str, ...]) -> list[str]:
    """Keep only present outcomes in configured order."""
    available = set(effects["outcome"].astype(str))
    return [outcome for outcome in outcomes if outcome in available]


def format_cell(value: str, column: ConsoleColumn) -> str:
    """Apply source left/right alignment rules."""
    if column.header in {"Method", "p"}:
        return value.ljust(column.width)
    return value.rjust(column.width)


def format_number(value: float) -> str:
    """Format a numeric value at source precision."""
    return "" if math.isnan(value) else f"{value:.2f}"


def format_p_value(p_value: float, stars: str) -> str:
    """Format p-values to two decimals and retain significance stars."""
    return f"{p_value:.2f}{stars}"
