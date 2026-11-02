# Copyright © 2026 TU Wien

"""Publication-level mwPR distributions by professor title and inbreeding."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from acib.regression.plots.mwpr import (
    HAZEN_PERCENTILE_COLUMN,
    MWPR_COLUMN,
    MwprAggregationSpec,
    aggregate_mwpr,
)
from acib.regression.plots.style import (
    BACKGROUND_COLOR,
    INBREEDING_COLORS,
    LIGHT_GRID_COLOR,
)

DEFAULT_PROFESSOR_TITLE_ORDER = (
    "Full Prof.",
    "Ao. Prof.",
    "Ass. Prof.",
    "FH-Prof.",
    "FH-Ass. Prof.",
)
DEFAULT_OUTPUT_BASENAME = "professor_title_inbreeding_violin"
INBREEDING_LABELS = {False: "Non-inbred", True: "Inbred"}
_STATUS_LABEL_COLUMN = "_inbreeding_status_label"
_PLOT_VALUE_COLUMN = "_plot_value"
_STATUS_OFFSETS = {False: -0.18, True: 0.18}


@dataclass(frozen=True, slots=True)
class ProfessorViolinMetric:
    """Define a metric's label, scale, and display limits."""

    column: str
    label: str
    panel_label: str
    scale: float = 1.0
    y_limits: tuple[float, float] = (0.0, 101.5)
    reference_line: float = 50.0


DEFAULT_PROFESSOR_VIOLIN_METRICS = (
    ProfessorViolinMetric(HAZEN_PERCENTILE_COLUMN, "mwPR", "Publication-level mwPR"),
)


@dataclass(frozen=True, slots=True)
class ProfessorViolinPlotConfig:
    """Configure the professor title and inbreeding violin plot."""

    input_path: Path | None = None
    output_dir: Path | None = None
    output_basename: str = DEFAULT_OUTPUT_BASENAME
    metrics: tuple[ProfessorViolinMetric, ...] = field(
        default_factory=lambda: DEFAULT_PROFESSOR_VIOLIN_METRICS
    )
    title_column: str = "title"
    inbreeding_column: str = "is_inbred"
    publication_column: str = "eid"
    researcher_column: str = "sc_id"
    title_order: tuple[str, ...] = DEFAULT_PROFESSOR_TITLE_ORDER
    figure_size: tuple[float, float] = (12.5, 6.2)
    output_formats: tuple[str, ...] = ("png",)
    dpi: int = 300
    min_publications: float = 0.0
    title: str = "Publication-level mwPR distributions by professor title and inbreeding status"


@dataclass(frozen=True, slots=True)
class ProfessorViolinPlotData:
    """Hold filtered publication observations and mwPR marker summaries."""

    observations: pd.DataFrame
    mwpr_all: dict[str, pd.DataFrame]
    mwpr_by_inbreeding: dict[str, pd.DataFrame]
    title_order: tuple[str, ...]
    config: ProfessorViolinPlotConfig

    @property
    def marker_data(self) -> pd.DataFrame:
        return self.markers_for_metric(self.config.metrics[0])

    def markers_for_metric(self, metric: ProfessorViolinMetric) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        positions = {title: index for index, title in enumerate(self.title_order)}
        for row in self.mwpr_all[metric.column].to_dict(orient="records"):
            title = row[self.config.title_column]
            if title in positions:
                rows.append(
                    {
                        self.config.title_column: title,
                        "series": "All",
                        "x": positions[title],
                        MWPR_COLUMN: row[MWPR_COLUMN] * metric.scale,
                    }
                )
        for row in self.mwpr_by_inbreeding[metric.column].to_dict(orient="records"):
            title = row[self.config.title_column]
            status = bool(row[self.config.inbreeding_column])
            if title in positions:
                rows.append(
                    {
                        self.config.title_column: title,
                        self.config.inbreeding_column: status,
                        "series": INBREEDING_LABELS[status],
                        "x": positions[title] + _STATUS_OFFSETS[status],
                        MWPR_COLUMN: row[MWPR_COLUMN] * metric.scale,
                    }
                )
        return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class ProfessorImpactViolinPlot:
    """Render and write the professor mwPR violin plot."""

    data: ProfessorViolinPlotData
    figure: Figure
    axes: tuple[Axes, ...]

    @classmethod
    def from_config(cls, config: ProfessorViolinPlotConfig) -> ProfessorImpactViolinPlot:
        """Load explicitly configured observations and render the figure.

        :param config: Plot settings including the input path.
        :return: Constructed plot.
        :raises ValueError: If no input path is configured.
        """
        if config.input_path is None:
            raise ValueError("Professor violin plot requires an explicit input_path.")
        path = config.input_path
        data = (
            pd.read_parquet(path)
            if path.suffix.casefold() in {".parquet", ".pq"}
            else pd.read_csv(path)
        )
        return cls.from_dataframe(data=data, config=config)

    @classmethod
    def from_dataframe(
        cls, *, data: pd.DataFrame, config: ProfessorViolinPlotConfig | None = None
    ) -> ProfessorImpactViolinPlot:
        """Prepare observations and create the figure.

        :param data: Publication-level professor observations.
        :param config: Plot settings.
        :return: Constructed plot.
        """
        settings = config or ProfessorViolinPlotConfig()
        prepared = prepare_professor_violin_plot_data(data=data, config=settings)
        figure, axes = plot_professor_violin(prepared)
        return cls(prepared, figure, axes)

    def write(self, output_dir: Path | None = None) -> tuple[Path, ...]:
        """Write configured figure formats.

        :param output_dir: Optional output-directory override.
        :return: Paths written in format order.
        """
        directory = output_dir or self.data.config.output_dir
        if directory is None:
            raise ValueError("Professor violin plot requires an explicit output_dir.")
        directory.mkdir(parents=True, exist_ok=True)
        paths = tuple(
            directory / f"{self.data.config.output_basename}.{fmt}"
            for fmt in self.data.config.output_formats
        )
        for path in paths:
            self.figure.savefig(path, dpi=self.data.config.dpi, bbox_inches="tight")
        return paths


def prepare_professor_violin_plot_data(
    *, data: pd.DataFrame, config: ProfessorViolinPlotConfig
) -> ProfessorViolinPlotData:
    """Filter observations, order title groups, and calculate mwPR markers.

    :param data: Publication-level input.
    :param config: Plot settings.
    :return: Prepared observations and summaries.
    :raises ValueError: If no metrics are configured.
    :raises KeyError: If required input columns are absent.
    """
    if not config.metrics:
        raise ValueError("Professor violin plot requires at least one metric.")
    value_columns = tuple(metric.column for metric in config.metrics)
    required = {
        *value_columns,
        config.title_column,
        config.inbreeding_column,
        config.publication_column,
        config.researcher_column,
    }
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Professor violin input is missing required columns: {sorted(missing)}")
    columns = [
        *value_columns,
        config.title_column,
        config.inbreeding_column,
        config.publication_column,
        config.researcher_column,
    ]
    observations = data[columns].copy()
    for column in value_columns:
        observations[column] = pd.to_numeric(observations[column], errors="coerce")
    observations[config.inbreeding_column] = observations[config.inbreeding_column].map(
        normalize_bool
    )
    observations = observations.dropna(
        subset=[
            config.title_column,
            config.inbreeding_column,
            config.publication_column,
            config.researcher_column,
        ]
    )
    observations = observations[observations[list(value_columns)].notna().any(axis=1)].copy()
    observations[config.inbreeding_column] = observations[config.inbreeding_column].astype(bool)
    for metric in config.metrics:
        observations[metric.column] = observations[metric.column].clip(
            lower=metric.y_limits[0] / metric.scale, upper=100.0 / metric.scale
        )
    observed = {str(value) for value in observations[config.title_column].dropna().unique()}
    order = tuple(
        [title for title in config.title_order if title in observed]
        + sorted(observed - set(config.title_order))
    )
    observations[config.title_column] = pd.Categorical(
        observations[config.title_column], categories=order, ordered=True
    )
    observations = observations.sort_values([config.title_column, config.inbreeding_column])
    mwpr_all = {
        metric.column: aggregate_mwpr(
            observations,
            spec=MwprAggregationSpec(
                group_columns=(config.title_column,),
                value_column=metric.column,
                publication_column=config.publication_column,
                researcher_column=config.researcher_column,
                min_publications=config.min_publications,
            ),
        )
        for metric in config.metrics
    }
    mwpr_status = {
        metric.column: aggregate_mwpr(
            observations,
            spec=MwprAggregationSpec(
                group_columns=(config.title_column, config.inbreeding_column),
                value_column=metric.column,
                publication_column=config.publication_column,
                researcher_column=config.researcher_column,
                min_publications=config.min_publications,
            ),
        )
        for metric in config.metrics
    }
    return ProfessorViolinPlotData(
        observations.reset_index(drop=True), mwpr_all, mwpr_status, order, config
    )


def normalize_bool(value: Any) -> bool | None:
    """Parse common boolean spellings while preserving missing and unknown values."""
    text = f"{value}".strip().casefold()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return None


def plot_professor_violin(data: ProfessorViolinPlotData) -> tuple[Figure, tuple[Axes, ...]]:
    """Draw status-split distributions, all-observation outlines, and mwPR markers.

    :param data: Prepared plotting data.
    :return: Figure and axes tuple.
    """
    config = data.config
    n_metrics = len(config.metrics)
    if n_metrics == 1:
        figure, axis = plt.subplots(figsize=config.figure_size)
        axes = (axis,)
    else:
        ncols = 2
        nrows = int(np.ceil(n_metrics / ncols))
        figure, grid = plt.subplots(
            nrows, ncols, figsize=config.figure_size, sharex=True, squeeze=False
        )
        all_axes = tuple(grid.ravel())
        for unused in all_axes[n_metrics:]:
            unused.set_visible(False)
        axes = all_axes[:n_metrics]
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    for axis, metric in zip(axes, config.metrics, strict=True):
        axis.set_facecolor(BACKGROUND_COLOR)
        frame = data.observations.dropna(subset=[metric.column]).copy()
        frame[_PLOT_VALUE_COLUMN] = frame[metric.column] * metric.scale
        frame[_STATUS_LABEL_COLUMN] = frame[config.inbreeding_column].map(INBREEDING_LABELS)
        sns.violinplot(
            data=frame,
            x=config.title_column,
            y=_PLOT_VALUE_COLUMN,
            hue=_STATUS_LABEL_COLUMN,
            legend=False,
            hue_order=[INBREEDING_LABELS[False], INBREEDING_LABELS[True]],
            order=list(data.title_order),
            palette={
                INBREEDING_LABELS[False]: INBREEDING_COLORS[False],
                INBREEDING_LABELS[True]: INBREEDING_COLORS[True],
            },
            split=True,
            inner="quart",
            width=0.84,
            cut=0,
            density_norm="width",
            bw_adjust=0.8,
            gap=0.08,
            linewidth=0.85,
            saturation=1.0,
            ax=axis,
        )
        existing = len(axis.collections)
        sns.violinplot(
            data=frame,
            x=config.title_column,
            y=_PLOT_VALUE_COLUMN,
            order=list(data.title_order),
            inner=None,
            color="#4A4A4A",
            width=0.92,
            cut=0,
            density_norm="width",
            bw_adjust=0.8,
            linewidth=2.0,
            fill=False,
            ax=axis,
        )
        if axis.legend_ is not None:
            axis.legend_.remove()
        for collection in axis.collections[existing:]:
            collection.set_clip_on(False)
            collection.set_zorder(6)
        axis.scatter(
            data.markers_for_metric(metric)["x"],
            data.markers_for_metric(metric)[MWPR_COLUMN],
            marker="D",
            s=62,
            facecolors="white",
            edgecolors="#111111",
            linewidths=1.25,
            zorder=7,
        )
        counts = frame.groupby([config.title_column, config.inbreeding_column], observed=True)[
            config.publication_column
        ].nunique()
        for position, title in enumerate(data.title_order):
            for status, offset, alignment in ((False, -0.035, "right"), (True, 0.035, "left")):
                count = counts.get((title, status), np.nan)
                if not pd.isna(count):
                    axis.text(
                        position + offset,
                        7.0,
                        rf"$n$={int(count):,}",
                        ha=alignment,
                        va="bottom",
                        fontsize=12.5,
                        color="#4F4F4F",
                        bbox={
                            "facecolor": BACKGROUND_COLOR,
                            "edgecolor": "none",
                            "alpha": 0.82,
                            "pad": 0.4,
                        },
                    )
        axis.axhline(
            metric.reference_line, color="#767676", linewidth=1.15, linestyle="--", zorder=4
        )
        axis.set_ylim(*metric.y_limits)
        axis.set_xlim(-0.65, len(data.title_order) - 0.35)
        axis.set_xticks(np.arange(len(data.title_order)))
        axis.set_xticklabels(data.title_order, fontsize=20)
        axis.set_ylabel("wPR [%]", fontsize=20)
        axis.set_xlabel(cast(str, None))
        if len(config.metrics) > 1:
            axis.text(
                0.015,
                0.965,
                metric.panel_label,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=16,
                fontweight="bold",
            )
        axis.grid(axis="y", color=LIGHT_GRID_COLOR, linewidth=0.8)
        axis.grid(axis="x", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#B8B8B8")
        axis.spines["bottom"].set_color("#B8B8B8")
        axis.tick_params(axis="y", labelsize=20)
    handles = [
        Patch(facecolor="none", edgecolor="#4A4A4A", linewidth=1.8, label="All"),
        Patch(
            facecolor=INBREEDING_COLORS[False], edgecolor="white", alpha=0.78, label="Non-inbred"
        ),
        Patch(facecolor=INBREEDING_COLORS[True], edgecolor="white", alpha=0.78, label="Inbred"),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#111111",
            markersize=10,
            label="mwPR(F) [%] (Non-Inbred, All, Inbred)",
        ),
    ]
    figure.subplots_adjust(left=0.075, right=0.995, top=0.98, bottom=0.20)
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=True,
        framealpha=1.0,
        edgecolor="#DDDDDD",
        fontsize=15,
    )
    return figure, axes
