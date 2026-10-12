# Copyright © 2026 TU Wien

"""Fit regression model specifications and expose suite diagnostics/reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import structlog

from acib.regression.analysis_config import (
    OUTCOME_DISPLAY_LABELS,
    RegressionSuiteConfig,
)
from acib.regression.config import RegressionOutputs
from acib.regression.constants import (
    ABSOLUTE_PERFORMANCE_OUTCOMES,
    ABSOLUTE_TOP_PERCENTILE_OUTCOMES,
    RELATIVE_PERFORMANCE_OUTCOMES,
)
from acib.regression.data import (
    dependent_variable_descriptives,
    model_frame,
    prepare_regression_data,
    required_columns_from_formula,
)
from acib.regression.descriptives import descriptive_statistics_latex_table
from acib.regression.model import FittedRegressionModel
from acib.regression.models import fit_regression_model, validate_mle_fit
from acib.regression.output import write_regression_suite_outputs
from acib.regression.plots.model9_inbreeding_predictions import (
    Model9InbreedingPredictionPlot,
    Model9InbreedingPredictionPlotConfig,
)
from acib.regression.plots.professor_violin import (
    ProfessorImpactViolinPlot,
    ProfessorViolinPlotConfig,
)
from acib.regression.specs import RegressionModelSpec
from acib.regression.tables import (
    aic_bic_table,
    best_models_by_metric,
    inbreeding_discipline_spec_comparison,
    ranked_aic_bic_table,
)

_logger = structlog.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegressionSuiteResult:
    """Hold fitted models and expose tables, formulas, and reports."""

    models: dict[tuple[str, str], FittedRegressionModel]
    dependent_variables: tuple[str, ...]
    relative_performance_outcomes: tuple[str, ...] = RELATIVE_PERFORMANCE_OUTCOMES
    absolute_performance_outcomes: tuple[str, ...] = ABSOLUTE_PERFORMANCE_OUTCOMES
    outcome_aliases: Mapping[str, str] = field(default_factory=lambda: dict(OUTCOME_DISPLAY_LABELS))

    @property
    def coefficients(self) -> pd.DataFrame:
        """Return model coefficient records in suite order."""
        frames = [model.coefficients for model in self.models.values()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @property
    def fit_statistics(self) -> pd.DataFrame:
        """Return one fit-statistics record per fitted model."""
        return pd.DataFrame([model.fit_statistics for model in self.models.values()])

    @property
    def formulas(self) -> dict[str, str]:
        """Return formulas and fitted MLE centering metadata."""
        result = {
            f"{outcome}_{model_id}": model.formula
            for (outcome, model_id), model in self.models.items()
        }
        result.update(
            {
                f"{outcome}_{model_id}_inflation": model.inflation_formula
                for (outcome, model_id), model in self.models.items()
                if model.inflation_formula is not None
            }
        )
        result.update(
            {
                f"{outcome}_{model_id}_centering": (
                    f"first_year = first_year - "
                    f"{model.mle_diagnostics['mle_centered_first_year']}"
                )
                for (outcome, model_id), model in self.models.items()
                if model.mle_diagnostics
                and pd.notna(model.mle_diagnostics.get("mle_centered_first_year"))
            }
        )
        return result

    @property
    def summaries(self) -> dict[str, str]:
        """Return textual model summaries by outcome and model identifier."""
        return {
            f"{outcome}_{model_id}": model.summary_text
            for (outcome, model_id), model in self.models.items()
        }

    @property
    def aic_bic(self) -> pd.DataFrame:
        """Return the AIC/BIC comparison table."""
        return aic_bic_table(self.fit_statistics)

    @property
    def aic_bic_ranked(self) -> pd.DataFrame:
        """Return AIC/BIC comparisons ranked within outcome."""
        return ranked_aic_bic_table(self.fit_statistics)

    @property
    def best_by_aic(self) -> pd.DataFrame:
        """Return each outcome's minimum-AIC model."""
        return best_models_by_metric(self.fit_statistics, metric="AIC")

    @property
    def best_by_bic(self) -> pd.DataFrame:
        """Return each outcome's minimum-BIC model."""
        return best_models_by_metric(self.fit_statistics, metric="BIC")

    @property
    def inbreeding_spec_comparison(self) -> pd.DataFrame:
        """Return the inbreeding coefficient across discipline specifications."""
        return inbreeding_discipline_spec_comparison(
            coefficients=self.coefficients, fit_statistics=self.fit_statistics
        )

    def latex_tables_for_data(self, *, data: pd.DataFrame | None) -> dict[str, str]:
        """Render available LaTeX reports, keyed by their original filenames."""
        from acib.regression.latex import (
            absolute_performance_latex_table,
            absolute_top_percentile_latex_table,
            can_render_absolute_top_percentile_latex_table,
            can_render_model10_without_international_mobility_table,
            model10_without_international_mobility_latex_table,
            model_comparison_latex_table,
            renderable_absolute_performance_model_ids,
            renderable_model_comparison_outcomes,
            renderable_research_performance_model_ids,
            research_performance_latex_table,
        )

        tables = {
            f"research_performance_{mid}.tex": research_performance_latex_table(
                self.models,
                model_id=mid,
                outcomes=self.relative_performance_outcomes,
                outcome_aliases=self.outcome_aliases,
            )
            for mid in renderable_research_performance_model_ids(
                self.models, outcomes=self.relative_performance_outcomes
            )
        }
        tables.update(
            {
                f"absolute_performance_{mid}.tex": absolute_performance_latex_table(
                    self.models,
                    model_id=mid,
                    outcomes=self.absolute_performance_outcomes,
                    outcome_aliases=self.outcome_aliases,
                )
                for mid in renderable_absolute_performance_model_ids(
                    self.models, outcomes=self.absolute_performance_outcomes
                )
            }
        )
        tables.update(
            {
                f"model_comparison_{outcome}.tex": model_comparison_latex_table(
                    self.models, dependent_variable=outcome, outcome_aliases=self.outcome_aliases
                )
                for outcome in renderable_model_comparison_outcomes(
                    self.models,
                    outcomes=(
                        *self.relative_performance_outcomes,
                        *self.absolute_performance_outcomes,
                    ),
                )
            }
        )
        if set(self.absolute_performance_outcomes).issuperset(
            ABSOLUTE_TOP_PERCENTILE_OUTCOMES
        ) and can_render_absolute_top_percentile_latex_table(self.models, model_id="m9"):
            tables["model9_absolute_top_percentile_robustness.tex"] = (
                absolute_top_percentile_latex_table(
                    self.models, model_id="m9", outcome_aliases=self.outcome_aliases
                )
            )
        if can_render_model10_without_international_mobility_table(
            self.models,
            relative_outcomes=self.relative_performance_outcomes,
            absolute_outcomes=self.absolute_performance_outcomes,
        ):
            tables["model10_without_international_mobility.tex"] = (
                model10_without_international_mobility_latex_table(
                    self.models,
                    relative_outcomes=self.relative_performance_outcomes,
                    absolute_outcomes=self.absolute_performance_outcomes,
                    outcome_aliases=self.outcome_aliases,
                )
            )
        if data is not None:
            tables["descriptive_statistics.tex"] = descriptive_statistics_latex_table(
                data,
                models=self.models,
                model_id="m9",
                absolute_performance_outcomes=self.absolute_performance_outcomes,
            )
        return tables

    @property
    def latex_tables(self) -> dict[str, str]:
        """Render model-only LaTeX reports."""
        return self.latex_tables_for_data(data=None)

    @property
    def model_9_latex_table(self) -> str:
        """Render the relative-performance table for Model 9.

        :return: Model 9 LaTeX table.
        """
        from acib.regression.latex import research_performance_latex_table

        return research_performance_latex_table(
            self.models,
            model_id="m9",
            outcomes=self.relative_performance_outcomes,
            outcome_aliases=self.outcome_aliases,
        )

    def latex_table_for_model(self, model_id: str) -> str:
        """Render the relative-performance table for one model.

        :param model_id: Model specification identifier.
        :return: Rendered LaTeX table.
        """
        from acib.regression.latex import research_performance_latex_table

        return research_performance_latex_table(
            self.models,
            model_id=model_id,
            outcomes=self.relative_performance_outcomes,
            outcome_aliases=self.outcome_aliases,
        )

    def print_model_summaries(self) -> None:
        """Print fitted summaries by dependent variable and ordered model ID."""
        for outcome in self.dependent_variables:
            print(f"\n{'=' * 100}\nOutcome: {outcome}\n{'=' * 100}")
            models = sorted(
                ((mid, model) for (name, mid), model in self.models.items() if name == outcome),
                key=lambda item: (
                    (0, int(item[0][1:]))
                    if item[0].startswith("m") and item[0][1:].isdigit()
                    else (1, item[0])
                ),
            )
            for model_id, model in models:
                print(f"\n{'-' * 100}\n{model_id.upper()}\n{'-' * 100}\n{model.summary_text}")

    def write(
        self,
        outputs: RegressionOutputs,
        *,
        data: pd.DataFrame | None = None,
        professor_articles: pd.DataFrame | None = None,
    ) -> None:
        """Write configured artifacts and available plots.

        :param outputs: Explicit stage-configured output paths.
        :param data: Optional data for outcome descriptives and LaTeX report.
        :param professor_articles: Optional publication-level professor records.
        """
        descriptives = (
            dependent_variable_descriptives(data, self.dependent_variables)
            if data is not None
            else pd.DataFrame()
        )
        outputs.plots_dir.mkdir(parents=True, exist_ok=True)
        write_regression_suite_outputs(
            coefficients=self.coefficients,
            fit_statistics=self.fit_statistics,
            formulas=self.formulas,
            summaries=self.summaries,
            descriptives=descriptives,
            aic_bic=self.aic_bic,
            aic_bic_ranked=self.aic_bic_ranked,
            best_by_aic=self.best_by_aic,
            best_by_bic=self.best_by_bic,
            inbreeding_spec_comparison=self.inbreeding_spec_comparison,
            latex_tables=self.latex_tables_for_data(data=data),
            author_regression_input=None,
            professor_article_input=None,
            outputs=outputs,
        )
        self.write_model9_inbreeding_prediction_plot(outputs=outputs)
        self.write_professor_violin_plot(
            plots_dir=outputs.plots_dir,
            professor_articles=professor_articles,
        )

    def write_model9_inbreeding_prediction_plot(
        self, *, outputs: RegressionOutputs
    ) -> tuple[Path, ...]:
        """Write Model 9 inbreeding predictions when fitted OLS models permit it."""
        plot_config = Model9InbreedingPredictionPlotConfig(output_dir=outputs.plots_dir)
        plot_outcomes = {outcome.column for outcome in plot_config.outcomes}
        model9_plot_models = {
            outcome: model
            for (outcome, model_id), model in self.models.items()
            if model_id == plot_config.model_id and outcome in plot_outcomes
        }
        if not model9_plot_models:
            return ()
        if any(model.model_type != "ols" for model in model9_plot_models.values()):
            _logger.info("Skipping model 9 prediction plot for non-OLS model results")
            return ()
        from matplotlib import pyplot as plt

        plot = Model9InbreedingPredictionPlot.from_result(
            result=self,
            config=plot_config,
        )
        paths = plot.write()
        plt.close(plot.figure)
        return paths

    def write_professor_violin_plot(
        self, *, plots_dir: Path, professor_articles: pd.DataFrame | None
    ) -> tuple[Path, ...]:
        """Write the professor violin plot when its input has required fields."""
        if professor_articles is None or professor_articles.empty:
            return ()
        from matplotlib import pyplot as plt

        config = ProfessorViolinPlotConfig(output_dir=plots_dir)
        missing = {
            config.title_column,
            config.inbreeding_column,
            config.publication_column,
            config.researcher_column,
            *(metric.column for metric in config.metrics),
        } - set(professor_articles)
        if missing:
            _logger.info(
                "Skipping professor violin plot because required columns are missing",
                missing_columns=sorted(missing),
            )
            return ()
        plot = ProfessorImpactViolinPlot.from_dataframe(data=professor_articles, config=config)
        paths = plot.write()
        plt.close(plot.figure)
        return paths


class RegressionSuite:
    """Fit and expose a complete configured regression suite."""

    def __init__(self, *, data: pd.DataFrame, config: RegressionSuiteConfig) -> None:
        """Prepare suite data while preserving input row order.

        :param data: Regression analysis input.
        :param config: Analysis-domain settings.
        """
        self.config = config
        self.data = prepare_regression_data(data, exclude_titles=config.exclude_titles)

    def fit(self) -> RegressionSuiteResult:
        """Fit each outcome/specification pair in configured order."""
        models: dict[tuple[str, str], FittedRegressionModel] = {}
        for outcome in self.config.dependent_variables:
            family = (self.config.model_types or {}).get(outcome, "ols")
            for spec in self.config.model_specs:
                model = self._fit_one(
                    dependent_variable=outcome,
                    spec=spec,
                    model_type=family,
                    inflation_formula=(self.config.zinb_inflation_formulas or {}).get(outcome, "1"),
                )
                if model is not None:
                    models[(outcome, spec.model_id)] = model
        return RegressionSuiteResult(
            models,
            self.config.dependent_variables,
            self.config.relative_performance_outcomes,
            self.config.absolute_performance_outcomes,
            self.config.outcome_aliases,
        )

    def _fit_one(
        self,
        *,
        dependent_variable: str,
        spec: RegressionModelSpec,
        model_type: str = "ols",
        inflation_formula: str = "1",
    ) -> FittedRegressionModel | None:
        """Fit one model, retaining the source missingness and diagnostic rules."""
        formula = spec.formula(dependent_variable, data=self.data)
        frame = model_frame(self.data, formula=formula, drop_missing=self.config.drop_missing)
        if model_type == "zinb":
            columns = required_columns_from_formula(inflation_formula)
            missing = sorted(column for column in columns if column not in self.data)
            if missing:
                raise KeyError(
                    "Missing regression input column(s) for ZINB inflation formula "
                    f"{inflation_formula!r}: {missing}"
                )
            for column in columns:
                if column not in frame:
                    frame[column] = self.data.loc[frame.index, column]
            if self.config.drop_missing:
                frame = frame.dropna()
        if frame.empty:
            _logger.warning("Skipping regression model with no complete rows", formula=formula)
            return None
        fit = fit_regression_model(
            model_type=model_type,
            formula=formula,
            data=frame,
            inflation_formula=inflation_formula,
            covariance_type=self.config.covariance_type,
            zinb_regularization=self.config.zinb_regularization,
            zinb_fit_method=self.config.zinb_fit_method,
            zinb_max_iterations=self.config.zinb_max_iterations,
            zinb_dispersion_power=self.config.zinb_dispersion_power,
            mle_gradient_tolerance=self.config.mle_gradient_tolerance,
        )
        diagnostics = (
            validate_mle_fit(
                fit,
                model_id=spec.model_id,
                dependent_variable=dependent_variable,
                model_type=model_type,
                policy=self.config.mle_failure_policy,
                gradient_tolerance=self.config.mle_gradient_tolerance,
            )
            if model_type in {"nb", "zinb"}
            else {}
        )
        return FittedRegressionModel(
            dependent_variable=dependent_variable,
            spec=spec,
            formula=formula,
            fit=fit,
            model_data=frame,
            mle_diagnostics=diagnostics,
            model_type=model_type,
            inflation_formula=inflation_formula if model_type == "zinb" else None,
            field_share_reference=spec.field_share_reference(data=self.data),
            share_control_type=spec.share_control_type(),
        )
