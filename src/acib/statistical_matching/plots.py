# Copyright © 2026 TU Wien

"""Source-style plots for statistical-matching diagnostics and effects."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MultipleLocator

_BACKGROUND = "#FFFFFF"
_GRID = "#E4E0D8"
_BALANCED = "#2F5D7C"
_IMBALANCED = "#B23A48"
_UNADJUSTED = "#B8B3AA"
_ZERO = "#6F6A60"
_THRESHOLD = "#8E887F"
_CONTROL = "#2E86C1"
_TREATED = "#E95A0C"
_METHOD_COLORS = {"PSM": "#2F5D7C", "IPTW": "#B23A48", "CEM": "#5B8E7D", "EB": "#C08A35"}
_COVARIATE_LABELS = {
    "academic_age": "Academic age",
    "first_year": "First pub. year",
    "title[Ao. Prof.]": "Ao. Prof.",
    "title[Ass. Prof.]": "Ass. Prof.",
    "title[FH-Prof.]": "FH-Prof.",
    "title[Full Prof.]": "Full Prof.",
    "multi_area_share": "Multidiscip.",
    "life_sci_share": "Life Sci.",
    "soc_hum_share": "Social Sci.",
    "health_sci_share": "Health Sci.",
    "phys_sci_share": "Physical Sci.",
}
_COVARIATE_ORDER = (
    "title[Full Prof.]",
    "title[Ao. Prof.]",
    "academic_age",
    "multi_area_share",
    "life_sci_share",
    "soc_hum_share",
    "health_sci_share",
    "first_year",
    "phys_sci_share",
    "title[Ass. Prof.]",
    "title[FH-Prof.]",
)


def write_matching_plots(
    balance: pd.DataFrame,
    balance_summary: pd.DataFrame,
    propensity: pd.DataFrame,
    by_method: pd.DataFrame,
    effects: pd.DataFrame,
    absolute_outcomes: tuple[str, ...],
    aliases: dict[str, str],
    method_order: tuple[str, ...],
    treatment: str,
    threshold: float,
    directory: Path,
) -> tuple[Path, ...]:
    """Write the three diagnostics figures and the optional absolute effect figure."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [write_balance_plot(balance, balance_summary, directory, method_order, threshold)]
    paths.append(_write_propensity(propensity, directory, treatment))
    paths.append(_write_method_overlap(by_method, directory, treatment, method_order))
    if absolute_outcomes:
        plotted = effects.loc[effects["outcome"].isin(absolute_outcomes)]
        if not plotted.empty:
            paths.append(
                _write_absolute(plotted, absolute_outcomes, aliases, directory, method_order)
            )
    return tuple(paths)


def write_balance_plot(
    balance: pd.DataFrame,
    summary: pd.DataFrame,
    directory: Path,
    method_order: tuple[str, ...],
    threshold: float,
) -> Path:
    """Write only the configured 2x2 covariate-balance figure."""
    required = {
        "method",
        "covariate",
        "diff_unadjusted",
        "diff_adjusted",
        "abs_diff_adjusted",
        "balanced",
    }
    missing = required - set(balance.columns)
    if missing:
        raise KeyError(f"Covariate balance input is missing required columns: {sorted(missing)}")
    frame = balance.loc[balance["method"].isin(method_order)].copy()
    if frame.empty:
        raise ValueError("Covariate balance input does not contain requested methods.")
    for column in ("diff_unadjusted", "diff_adjusted", "abs_diff_adjusted"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["diff_unadjusted", "diff_adjusted", "abs_diff_adjusted"]].isna().any().any():
        raise ValueError("Covariate balance input contains non-numeric balance values.")
    frame["abs_diff_unadjusted"] = frame["diff_unadjusted"].abs()
    observed = set(frame["covariate"].astype(str))
    ordered = [value for value in _COVARIATE_ORDER if value in observed]
    extra = (
        frame.loc[~frame["covariate"].isin(ordered)]
        .groupby("covariate", as_index=False)["abs_diff_unadjusted"]
        .max()
        .sort_values(["abs_diff_unadjusted", "covariate"], ascending=[False, True])
    )
    ordered.extend(str(value) for value in extra["covariate"])
    ymap = {value: index for index, value in enumerate(reversed(ordered))}
    frame["y"] = frame["covariate"].map(ymap)
    frame["label"] = frame["covariate"].map(lambda value: _COVARIATE_LABELS.get(value, value))
    methods = tuple(method for method in method_order if method in set(frame["method"].astype(str)))
    fig, grid = plt.subplots(2, 2, figsize=(10.2, 6.7), sharex=True, sharey=True)
    fig.patch.set_facecolor(_BACKGROUND)
    axes = tuple(grid.flat)
    for ax, method in zip(axes, methods, strict=False):
        rows = frame.loc[frame["method"].eq(method)]
        ax.set_facecolor(_BACKGROUND)
        colors = np.where(rows["balanced"].astype(bool), _BALANCED, _IMBALANCED)
        ax.scatter(
            rows["abs_diff_unadjusted"],
            rows["y"],
            s=48,
            facecolors="none",
            edgecolors=_UNADJUSTED,
            linewidth=1.2,
            zorder=4,
        )
        ax.scatter(
            rows["abs_diff_adjusted"],
            rows["y"],
            s=55,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.axvline(0, color=_ZERO, linewidth=1.2)
        ax.axvline(threshold, color=_THRESHOLD, linewidth=1.8, linestyle="--")
        max_smd = float(rows["abs_diff_adjusted"].max())
        ax.set_title(method, fontsize=13, fontweight="bold", pad=8)
        values = summary.loc[summary["method"].astype(str).eq(method)]
        sample_text = f"max |SMD| = {max_smd:.2f}"
        if not values.empty and {"n_non_inbred", "n_inbred", "ess_non_inbred"} <= set(
            values.columns
        ):
            row = values.iloc[0]
            n = round(float(row["n_non_inbred"])) + round(float(row["n_inbred"]))
            sample_text = (
                f"n = {n:,}\n"
                + (
                    ""
                    if method == "PSM"
                    else f"control ESS = {round(float(row['ess_non_inbred'])):,}\n"
                )
                + f"max |SMD| = {max_smd:.2f}"
            )
        ax.text(
            0.965,
            0.055,
            sample_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.5,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "white",
                "edgecolor": "#D8D2C8",
                "linewidth": 0.8,
                "alpha": 0.92,
            },
        )
        ax.set_xlim(0, 0.4)
        ax.xaxis.set_major_locator(FixedLocator(np.arange(0, 0.401, 0.1).tolist()))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=_GRID, linewidth=0.8)
        ax.grid(axis="x", which="minor", color=_GRID, linewidth=0.45, alpha=0.65)
        ax.grid(axis="y", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(_UNADJUSTED)
        ax.spines["bottom"].set_color(_UNADJUSTED)
        ax.tick_params(axis="x", labelsize=13)
    for ax in axes[len(methods) :]:
        ax.set_visible(False)
    labels = (
        frame[["covariate", "label", "y"]]
        .drop_duplicates("covariate")
        .sort_values("y", ascending=False)
    )
    for index, ax in enumerate(axes[: len(methods)]):
        ax.set_yticks(labels["y"].tolist())
        if index % 2 == 0:
            ax.set_yticklabels(labels["label"].tolist(), fontsize=12)
            ax.tick_params(axis="y", labelleft=True, length=0)
        else:
            ax.tick_params(axis="y", labelleft=False, length=0)
    fig.text(
        0.59,
        0.135,
        "Absolute standardized mean difference (SMD) [-]",
        ha="center",
        va="center",
        fontsize=13,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=_UNADJUSTED,
            markeredgewidth=1.2,
            markersize=7,
            label="Unadjusted",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_BALANCED,
            markeredgecolor="white",
            markersize=7,
            label="Adjusted",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_IMBALANCED,
            markeredgecolor="white",
            markersize=7,
            label="Adjusted, |SMD| ≥ 0.10",
        ),
        Line2D(
            [0],
            [0],
            color=_THRESHOLD,
            linewidth=1.8,
            linestyle="--",
            label="Balance threshold (0.10)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        fontsize=11,
    )
    fig.subplots_adjust(left=0.19, right=0.99, top=0.935, bottom=0.20, wspace=0.08, hspace=0.32)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "covariate_balance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_propensity(data: pd.DataFrame, directory: Path, treatment: str) -> Path:
    """Write pooled propensity-score overlap distributions."""
    frame = data[[treatment, "propensity_score"]].copy()
    frame[treatment] = frame[treatment].astype(bool)
    frame["propensity_score"] = pd.to_numeric(frame["propensity_score"], errors="coerce")
    frame = frame.dropna(subset=["propensity_score"])
    if frame.empty or not frame[treatment].any() or frame[treatment].all():
        raise ValueError(
            "Propensity score input must contain usable treated and control observations."
        )
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.patch.set_facecolor(_BACKGROUND)
    ax.set_facecolor(_BACKGROUND)
    bins = np.linspace(0, 1, 37).tolist()
    for treated, color, label in ((False, _CONTROL, "Non-inbred"), (True, _TREATED, "Inbred")):
        values = frame.loc[frame[treatment].eq(treated), "propensity_score"]
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.28,
            linewidth=1.4,
            color=color,
            label=label,
        )
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.8, color=color)
    _style_overlap(ax)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.93, bottom=0.16)
    path = directory / "propensity_score_overlap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_method_overlap(
    data: pd.DataFrame, directory: Path, treatment: str, methods_order: tuple[str, ...]
) -> Path:
    """Write propensity-score distributions weighted by each method."""
    required = {"method", treatment, "propensity_score", "weights"}
    missing = required - set(data)
    if missing:
        raise KeyError(f"By-method propensity score input is missing columns: {sorted(missing)}")
    frame = data.copy()
    frame["method"] = frame["method"].astype(str)
    frame = frame.loc[frame["method"].isin(methods_order)].copy()
    frame[treatment] = frame[treatment].astype(bool)
    for column in ("propensity_score", "weights"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["propensity_score", "weights"])
    methods = tuple(m for m in methods_order if m in set(frame["method"]))
    fig, grid = plt.subplots(2, 2, figsize=(10.2, 6.7), sharex=True, sharey=True)
    fig.patch.set_facecolor(_BACKGROUND)
    axes = tuple(grid.flat)
    bins = np.linspace(0, 1, 37)
    for ax, method in zip(axes, methods, strict=False):
        rows = frame.loc[frame["method"].eq(method)]
        ax.set_facecolor(_BACKGROUND)
        for treated, color in ((False, _CONTROL), (True, _TREATED)):
            group = rows.loc[rows[treatment].eq(treated)]
            for kind, alpha, width in (("stepfilled", 0.25, 1.3), ("step", 1.0, 1.8)):
                ax.hist(
                    group["propensity_score"],
                    bins=bins,
                    weights=group["weights"],
                    density=True,
                    histtype=kind,
                    alpha=alpha,
                    linewidth=width,
                    color=color,
                )
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(FixedLocator(np.arange(0, 1.001, 0.2).tolist()))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.set_title(method, fontsize=13, fontweight="bold", pad=8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.grid(axis="x", which="minor", color=_GRID, linewidth=0.45, alpha=0.65)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(_UNADJUSTED)
        ax.spines["bottom"].set_color(_UNADJUSTED)
        ax.tick_params(axis="both", labelsize=12)
    for ax in axes[len(methods) :]:
        ax.set_visible(False)
    for ax in axes[2:]:
        ax.set_xlabel("Estimated propensity score", fontsize=13)
    for ax in axes[::2]:
        ax.set_ylabel("Density", fontsize=13)
    fig.legend(
        handles=[
            Line2D([0], [0], color=_CONTROL, lw=2, label="Non-inbred"),
            Line2D([0], [0], color=_TREATED, lw=2, label="Inbred"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.94, bottom=0.16, wspace=0.08, hspace=0.30)
    path = directory / "propensity_score_overlap_by_method.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _style_overlap(ax: plt.Axes) -> None:
    """Apply shared styling to an overlap-distribution axis."""
    ax.set_xlim(0, 1)
    ax.set_xlabel("Estimated propensity score", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Propensity-score overlap", fontsize=14, fontweight="bold", pad=9)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(_UNADJUSTED)
    ax.spines["bottom"].set_color(_UNADJUSTED)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=False, fontsize=12, loc="upper left")


def _write_absolute(
    data: pd.DataFrame,
    outcomes: tuple[str, ...],
    aliases: dict[str, str],
    directory: Path,
    method_order: tuple[str, ...],
) -> Path:
    """Write estimated effects and confidence intervals for absolute outcomes."""
    frame = data.copy()
    for column in ("estimate", "ci_lower", "ci_upper"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["estimate", "ci_lower", "ci_upper"]].isna().any().any():
        raise ValueError("Absolute-outcome effects contain non-numeric estimates or intervals.")
    plotted = tuple(outcome for outcome in outcomes if outcome in set(frame["outcome"]))
    methods = tuple(m for m in method_order if frame["method"].astype(str).eq(m).any())
    cols, rows = 2, ceil(len(plotted) / 2)
    fig, grid = plt.subplots(rows, cols, figsize=(11.5, 3.4 * rows), squeeze=False)
    axes = tuple(grid.flat)
    fig.suptitle("Absolute-outcome matching effects", fontsize=15, fontweight="bold")
    for ax, outcome in zip(axes, plotted, strict=False):
        values = frame.loc[frame["outcome"].eq(outcome)]
        present = tuple(m for m in methods if values["method"].astype(str).eq(m).any())
        for position, method in enumerate(present):
            row = values.loc[values["method"].astype(str).eq(method)].iloc[0]
            estimate = float(row["estimate"])
            color = _METHOD_COLORS.get(method, "#2F5D7C")
            ax.errorbar(
                estimate,
                position,
                xerr=np.asarray(
                    [
                        [max(estimate - float(row["ci_lower"]), 0)],
                        [max(float(row["ci_upper"]) - estimate, 0)],
                    ]
                ),
                fmt="o",
                markersize=6,
                capsize=3,
                color=color,
                ecolor=color,
            )
        ax.axvline(0, color=_ZERO, linewidth=1, linestyle="--")
        ax.set_yticks(range(len(present)), labels=present)
        ax.invert_yaxis()
        ax.set_title(aliases.get(outcome, outcome).replace(r"\%", "%"), fontweight="bold")
        ax.set_xlabel("Estimated ATT (95% CI)")
        ax.grid(axis="x", color="#DDD8CF", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    for ax in axes[len(plotted) :]:
        ax.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = directory / "absolute_outcome_effects.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
