# Copyright © 2026 TU Wien

"""Adjusted predictions by inbreeding status for the Model 9 outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from acib.constants import HAZEN_PERCENTILE

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from patsy import build_design_matrices

from acib.regression.constants import PAPER_REGRESSION_MODEL_TERMS, TITLE
from acib.regression.plots.professor_violin import DEFAULT_PROFESSOR_TITLE_ORDER
from acib.regression.plots.style import (
    BACKGROUND_COLOR,
    CI_COLOR,
    GRID_COLOR,
    INBREEDING_COLORS,
)

DEFAULT_OUTPUT_BASENAME = "model9_inbreeding_predictions"
STATUS_ORDER: tuple[bool, ...] = (False, True)
STATUS_LABELS = {False: "Non-inbred", True: "Inbred"}
STATUS_Y_POSITIONS = {False: 0.54, True: 0.46}


@dataclass(frozen=True, slots=True)
class Model9InbreedingOutcome:
    """Define a Model 9 outcome and its displayed scale and axis."""

    column: str
    label: str
    scale: float = 1.0
    ticks: tuple[float, ...] = ()
    x_limits: tuple[float, float] | None = None


DEFAULT_MODEL9_INBREEDING_OUTCOMES = (
    Model9InbreedingOutcome(
        HAZEN_PERCENTILE, "mwPR", ticks=(54.0, 55.0, 56.0, 57.0), x_limits=(53.6, 56.9)
    ),
    Model9InbreedingOutcome(
        "pp_top_10", "PP(top 10%)", 100.0, (12.5, 13.0, 13.5, 14.0, 14.5), (12.3, 14.6)
    ),
    Model9InbreedingOutcome(
        "pp_top_5", "PP(top 5%)", 100.0, (6.25, 6.5, 6.75, 7.0, 7.25, 7.5), (6.2, 7.5)
    ),
    Model9InbreedingOutcome("pp_top_1", "PP(top 1%)", 100.0, (1.3, 1.4, 1.5, 1.6), (1.25, 1.6)),
)


@dataclass(frozen=True, slots=True)
class Model9InbreedingPredictionPlotConfig:
    """Configure regression inputs and output for the prediction figure."""

    input_path: Path | None = None
    output_dir: Path | None = None
    output_basename: str = DEFAULT_OUTPUT_BASENAME
    outcomes: tuple[Model9InbreedingOutcome, ...] = DEFAULT_MODEL9_INBREEDING_OUTCOMES
    statuses: tuple[bool, ...] = STATUS_ORDER
    model_id: str = "m9"
    title: str | None = None
    group_by_title: bool = False
    figure_size: tuple[float, float] = (15.0, 7.2)
    output_formats: tuple[str, ...] = ("png",)
    dpi: int = 300


@dataclass(frozen=True, slots=True)
class Model9InbreedingPredictionPlotData:
    """Hold estimated rows and their plotting configuration."""

    predictions: pd.DataFrame
    config: Model9InbreedingPredictionPlotConfig


@dataclass(frozen=True, slots=True)
class Model9InbreedingPredictionPlot:
    """Render and write Model 9 adjusted predictions."""

    data: Model9InbreedingPredictionPlotData
    figure: Figure
    axes: tuple[Axes, ...]

    @classmethod
    def from_config(
        cls, config: Model9InbreedingPredictionPlotConfig
    ) -> Model9InbreedingPredictionPlot:
        """Load explicitly configured regression input and render predictions.

        :param config: Plot settings including the input path.
        :return: Constructed plot.
        :raises ValueError: If no input path is configured.
        """
        if config.input_path is None:
            raise ValueError("Model 9 prediction plot requires an explicit input_path.")
        path = config.input_path
        data = (
            pd.read_parquet(path)
            if path.suffix.casefold() in {".parquet", ".pq"}
            else pd.read_csv(path)
        )
        return cls.from_dataframe(data=data, config=config)

    @classmethod
    def from_result(
        cls, *, result: Any, config: Model9InbreedingPredictionPlotConfig | None = None
    ) -> Model9InbreedingPredictionPlot:
        """Render predictions from already fitted suite models.

        :param result: Fitted regression suite result.
        :param config: Plot settings.
        :return: Constructed plot.
        :raises ValueError: If no predictions can be estimated.
        """
        settings = config or Model9InbreedingPredictionPlotConfig()
        predictions = prediction_rows(result=result, config=settings)
        if predictions.empty:
            raise ValueError("No Model 9 inbreeding-status predictions could be estimated.")
        plot_data = Model9InbreedingPredictionPlotData(predictions, settings)
        figure, axes = plot_model9_inbreeding_predictions(plot_data)
        return cls(plot_data, figure, axes)

    @classmethod
    def from_dataframe(
        cls, *, data: pd.DataFrame, config: Model9InbreedingPredictionPlotConfig | None = None
    ) -> Model9InbreedingPredictionPlot:
        """Fit the configured regression and render predictions from a frame.

        :param data: Regression input frame.
        :param config: Plot settings; defaults are used when omitted.
        :return: Constructed plot.
        :raises ImportError: When the regression suite implementation is unavailable.
        """
        try:
            from acib.regression.analysis_config import RegressionSuiteConfig
            from acib.regression.specs import RegressionModelSpec
            from acib.regression.suite import RegressionSuite
        except ImportError as exc:
            raise ImportError("Model 9 plotting requires the regression suite modules.") from exc
        settings = config or Model9InbreedingPredictionPlotConfig()
        suite = RegressionSuite(
            data=data,
            config=RegressionSuiteConfig(
                dependent_variables=tuple(item.column for item in settings.outcomes),
                model_specs=(
                    RegressionModelSpec(
                        model_id=settings.model_id,
                        terms=PAPER_REGRESSION_MODEL_TERMS[settings.model_id],
                    ),
                ),
                exclude_titles=(),
            ),
        )
        result = suite.fit()
        predictions = prediction_rows(result=result, config=settings)
        if predictions.empty:
            raise ValueError("No Model 9 inbreeding-status predictions could be estimated.")
        plot_data = Model9InbreedingPredictionPlotData(predictions, settings)
        figure, axes = plot_model9_inbreeding_predictions(plot_data)
        return cls(plot_data, figure, axes)

    def write(self, output_dir: Path | None = None) -> tuple[Path, ...]:
        """Write configured figure formats in their declared order.

        :param output_dir: Optional override for the configured output directory.
        :return: Paths written.
        """
        directory = output_dir or self.data.config.output_dir
        if directory is None:
            raise ValueError("Model 9 prediction plot requires an explicit output_dir.")
        directory.mkdir(parents=True, exist_ok=True)
        paths = tuple(
            directory / f"{self.data.config.output_basename}.{fmt}"
            for fmt in self.data.config.output_formats
        )
        for path in paths:
            self.figure.savefig(path, dpi=self.data.config.dpi, bbox_inches="tight")
        return paths


def prediction_rows(*, result: Any, config: Model9InbreedingPredictionPlotConfig) -> pd.DataFrame:
    """Calculate average adjusted predictions and confidence intervals.

    :param result: Fitted regression-suite result exposing fitted models.
    :param config: Plot settings.
    :return: Predictions ordered by outcome and status.
    """
    rows: list[dict[str, Any]] = []
    for outcome_index, outcome in enumerate(config.outcomes):
        model = result.models.get((outcome.column, config.model_id))
        if model is None:
            continue
        model_data = model.model_data
        if config.title is not None:
            if TITLE not in model_data.columns:
                continue
            model_data = model_data.loc[model_data[TITLE].eq(config.title)]
        if model_data.empty:
            continue
        titles: tuple[str | None, ...]
        if config.group_by_title:
            if TITLE not in model_data.columns:
                continue
            titles = tuple(
                title
                for title in DEFAULT_PROFESSOR_TITLE_ORDER
                if model_data[TITLE].eq(title).any()
            )
        else:
            titles = (config.title,)
        for title in titles:
            title_data = (
                model_data.loc[model_data[TITLE].eq(title)] if title is not None else model_data
            )
            available = {bool(value) for value in title_data["is_inbred"].dropna().unique()}
            for status_index, status in enumerate(config.statuses):
                if status not in available:
                    continue
                estimate = average_adjusted_prediction(
                    model=model, is_inbred=status, model_data=title_data
                )
                rows.append(
                    {
                        "dependent_variable": outcome.column,
                        "outcome_label": outcome.label,
                        "outcome_order": outcome_index,
                        "title": title,
                        "is_inbred": status,
                        "status_label": STATUS_LABELS[status],
                        "status_order": status_index,
                        **{key: value * outcome.scale for key, value in estimate.items()},
                        "n_obs": len(title_data),
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["outcome_order", "title", "status_order"], na_position="first")
        .reset_index(drop=True)
    )


def average_adjusted_prediction(
    *, model: Any, is_inbred: bool, model_data: pd.DataFrame | None = None
) -> dict[str, float]:
    """Estimate the average adjusted linear prediction and confidence interval.

    :param model: Fitted model wrapper.
    :param is_inbred: Status set for every row in the model's estimation data.
    :return: Estimate, standard error, and confidence limits.
    """
    new_data = (model.model_data if model_data is None else model_data).copy()
    new_data["is_inbred"] = is_inbred
    matrix = build_design_matrices(
        [model.fit.model.data.design_info], new_data, return_type="dataframe"
    )[0]
    contrast = matrix.loc[:, model.fit.params.index].mean(axis=0).to_numpy()
    test = model.fit.t_test(contrast)
    low, high = np.ravel(test.conf_int())
    return {
        "estimate": float(np.ravel(test.effect)[0]),
        "std_error": float(np.ravel(test.sd)[0]),
        "conf_low": float(low),
        "conf_high": float(high),
    }


def plot_model9_inbreeding_predictions(
    data: Model9InbreedingPredictionPlotData,
) -> tuple[Figure, tuple[Axes, ...]]:
    """Draw the four-outcome two-by-two prediction grid.

    :param data: Prepared prediction rows and plot settings.
    :return: Figure and flattened axes tuple.
    """
    config = data.config
    figure, grid = plt.subplots(2, 2, figsize=config.figure_size, squeeze=False)
    axes = tuple(grid.ravel())
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    if config.title is not None:
        figure.suptitle(config.title, fontsize=22, fontweight="bold", y=0.995)
    titles = tuple(
        title
        for title in DEFAULT_PROFESSOR_TITLE_ORDER
        if config.group_by_title and data.predictions["title"].eq(title).any()
    )
    title_positions = {
        title: len(titles) - title_index - 1 for title_index, title in enumerate(titles)
    }
    for index, (axis, outcome) in enumerate(zip(axes, config.outcomes, strict=False)):
        panel = data.predictions[data.predictions["dependent_variable"].eq(outcome.column)]
        axis.set_facecolor(BACKGROUND_COLOR)
        if panel.empty:
            axis.set_visible(False)
            continue
        for row in panel.to_dict(orient="records"):
            status = bool(row["is_inbred"])
            y_position = (
                title_positions[row["title"]] + (0.16 if not status else -0.16)
                if config.group_by_title
                else (0.54 if not status else 0.46)
            )
            axis.errorbar(
                row["estimate"],
                y_position,
                xerr=[[row["estimate"] - row["conf_low"]], [row["conf_high"] - row["estimate"]]],
                fmt="o",
                color=INBREEDING_COLORS[status],
                ecolor=CI_COLOR,
                elinewidth=2.6,
                capsize=5,
                capthick=2.2,
                markersize=13.0,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=3,
            )
            if not config.group_by_title:
                axis.text(
                    row["conf_high"]
                    + max((panel.conf_high.max() - panel.conf_low.min()) * 0.04, 0.05),
                    y_position,
                    f"{row['estimate']:.1f}",
                    ha="left",
                    va="center",
                    fontsize=18,
                    color="#2D2A26",
                )
        limits, ticks = prediction_axis_limits(panel=panel, outcome=outcome)
        axis.set_xlim(*limits)
        if ticks:
            axis.set_xticks(ticks)
        axis.set_title(outcome.label, loc="left", fontsize=22, fontweight="bold")
        if config.group_by_title:
            axis.set_ylim(-0.5, len(titles) - 0.5)
            axis.set_yticks([title_positions[title] for title in titles])
            axis.set_yticklabels(titles, fontsize=22)
        else:
            axis.set_ylim(0.39, 0.61)
            axis.set_yticks([0.54, 0.46])
            axis.set_yticklabels(["Non-inbred", "Inbred"] if index % 2 == 0 else [], fontsize=22)
        axis.set_ylabel(None)
        axis.set_xlabel(None)
        axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        axis.grid(axis="y", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#B8B3AA")
        axis.spines["bottom"].set_color("#B8B3AA")
        axis.tick_params(axis="x", labelsize=22)
        axis.tick_params(axis="y", length=0)
    for axis in axes[len(config.outcomes) :]:
        axis.set_visible(False)
    if config.group_by_title:
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D(
                [],
                [],
                color=INBREEDING_COLORS[status],
                marker="o",
                markersize=9,
                linewidth=2,
                label=STATUS_LABELS[status],
            )
            for status in config.statuses
        ]
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=len(legend_handles),
            frameon=False,
            fontsize=20,
            bbox_to_anchor=(0.5, 1.0),
        )
    figure.supxlabel("Average adjusted prediction [%]", fontsize=22, y=0.052)
    figure.subplots_adjust(
        left=0.16 if config.group_by_title else 0.14,
        right=0.99,
        top=0.89 if config.group_by_title else (0.91 if config.title is not None else 0.965),
        bottom=0.16,
        wspace=0.25 if config.group_by_title else 0.12,
        hspace=0.38 if config.group_by_title else 0.32,
    )
    return figure, axes


def prediction_axis_limits(
    *, panel: pd.DataFrame, outcome: Model9InbreedingOutcome
) -> tuple[tuple[float, float], tuple[float, ...]]:
    """Choose an x-axis that contains confidence intervals and value labels.

    Use the publication-scale limits and ticks when the estimates fit. Otherwise,
    derive limits from this panel and let Matplotlib choose readable tick marks.

    :param panel: Prediction rows for one outcome.
    :param outcome: Display scale and preferred publication-axis settings.
    :return: Axis limits and optional explicit ticks.
    """
    lower = float(panel["conf_low"].min())
    upper = float(panel["conf_high"].max())
    interval_span = upper - lower
    scale = max(interval_span, abs(lower) * 0.02, abs(upper) * 0.02, 0.1)
    label_offset = max(interval_span * 0.04, 0.05)
    label_end = upper + label_offset

    if outcome.x_limits is not None:
        preferred_lower, preferred_upper = outcome.x_limits
        if lower >= preferred_lower and label_end <= preferred_upper:
            return outcome.x_limits, outcome.ticks

    limits = (lower - scale * 0.08, label_end + scale * 0.2)
    return limits, ()
