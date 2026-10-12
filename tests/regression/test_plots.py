# Copyright © 2026 TU Wien

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from acib.regression.plots.model9_inbreeding_predictions import (
    Model9InbreedingPredictionPlot,
    Model9InbreedingPredictionPlotConfig,
)
from acib.regression.plots.professor_violin import (
    ProfessorImpactViolinPlot,
    ProfessorViolinPlotConfig,
    prepare_professor_violin_plot_data,
)


def professor_plot_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "eid": "shared",
                "sc_id": 1,
                "title": "Full Prof.",
                "is_inbred": False,
                "hazen_perc_med": 80.0,
            },
            {
                "eid": "shared",
                "sc_id": 2,
                "title": "Ass. Prof.",
                "is_inbred": True,
                "hazen_perc_med": 80.0,
            },
            {
                "eid": "solo-1",
                "sc_id": 1,
                "title": "Full Prof.",
                "is_inbred": False,
                "hazen_perc_med": 40.0,
            },
            {
                "eid": "solo-2",
                "sc_id": 3,
                "title": "Ass. Prof.",
                "is_inbred": False,
                "hazen_perc_med": 60.0,
            },
        ]
    )


def regression_frame() -> pd.DataFrame:
    titles = ["Full Prof.", "Ao. Prof.", "Ass. Prof.", "FH-Prof.", "FH-Ass. Prof."]
    rows = []
    for index in range(80):
        inbred = index % 2 == 1
        age = 4 + index % 17
        year = 1995 + index % 23
        value = 54.0 - (2.0 if inbred else 0.0) + 0.15 * age + 0.04 * (year - 1995)
        rows.append(
            {
                "hazen_perc_med": value,
                "pp_top_10": value / 100,
                "pp_top_5": value / 100,
                "pp_top_1": value / 100,
                "is_inbred": inbred,
                "academic_age": age,
                "title": titles[index % len(titles)],
                "coaffiliation_publication_share": index % 5 / 10,
                "mean_coauthors_per_publication": 2 + index % 7 / 3,
                "med_share": (index % 4) / 4,
                "eng_share": 1 - (index % 4) / 4,
                "first_year": year,
                "n_away_years_domestic": index % 4,
                "n_away_years_abroad": index % 6,
            }
        )
    return pd.DataFrame(rows)


def test_professor_violin_data_computes_publication_weighted_markers() -> None:
    data = prepare_professor_violin_plot_data(
        data=professor_plot_frame(), config=ProfessorViolinPlotConfig()
    )
    title_rows = data.mwpr_all["hazen_perc_med"].set_index("title")
    assert data.title_order == ("Full Prof.", "Ass. Prof.")
    assert title_rows.loc["Full Prof.", "mwpr"] == pytest.approx((80 * 0.5 + 40) / 1.5)
    assert set(data.marker_data["series"]) == {"All", "Non-inbred", "Inbred"}


def test_professor_violin_writes_configured_png(tmp_path: Path) -> None:
    plot = ProfessorImpactViolinPlot.from_dataframe(
        data=professor_plot_frame(),
        config=ProfessorViolinPlotConfig(
            output_dir=tmp_path, output_basename="test_professor_violin", output_formats=("png",)
        ),
    )
    paths = plot.write()
    assert paths == (tmp_path / "test_professor_violin.png",)
    assert paths[0].is_file()
    assert len(plot.axes) == 1


def test_model9_dataframe_plot_returns_eight_status_predictions(tmp_path: Path) -> None:
    plot = Model9InbreedingPredictionPlot.from_dataframe(
        data=regression_frame(),
        config=Model9InbreedingPredictionPlotConfig(output_dir=tmp_path),
    )
    predictions = plot.data.predictions
    assert len(predictions) == 8
    assert set(predictions["status_label"]) == {"Non-inbred", "Inbred"}
    assert set(predictions["outcome_label"]) == {"mwPR", "PP(top 10%)", "PP(top 5%)", "PP(top 1%)"}
    assert predictions["estimate"].notna().all()
    assert (predictions["conf_low"] <= predictions["estimate"]).all()
    assert (predictions["estimate"] <= predictions["conf_high"]).all()
    assert plot.write() == (tmp_path / "model9_inbreeding_predictions.png",)


def test_model9_plot_can_load_configured_csv_and_format(tmp_path: Path) -> None:
    source = regression_frame()
    input_path = tmp_path / "regression.csv"
    source.to_csv(input_path, index=False)
    plot = Model9InbreedingPredictionPlot.from_config(
        Model9InbreedingPredictionPlotConfig(
            input_path=input_path,
            output_dir=tmp_path,
            output_basename="script_model9_inbreeding_predictions",
            output_formats=("png",),
        )
    )
    paths = plot.write()
    assert paths == (tmp_path / "script_model9_inbreeding_predictions.png",)
    assert paths[0].is_file()


def test_model9_predictions_from_fitted_suite_result() -> None:
    from acib.regression.analysis_config import RegressionSuiteConfig
    from acib.regression.constants import PAPER_REGRESSION_MODEL_TERMS
    from acib.regression.specs import RegressionModelSpec
    from acib.regression.suite import RegressionSuite

    outcomes = ("hazen_perc_med", "pp_top_10", "pp_top_5", "pp_top_1")
    fitted = RegressionSuite(
        data=regression_frame(),
        config=RegressionSuiteConfig(
            dependent_variables=outcomes,
            model_specs=(RegressionModelSpec("m9", PAPER_REGRESSION_MODEL_TERMS["m9"]),),
            exclude_titles=(),
        ),
    ).fit()
    plot = Model9InbreedingPredictionPlot.from_result(result=fitted)
    assert len(plot.data.predictions) == 8
    assert len(plot.axes) == 4


def test_model9_prediction_plot_requires_output_directory() -> None:
    plot = Model9InbreedingPredictionPlot.from_dataframe(data=regression_frame())
    with pytest.raises(ValueError, match="output_dir"):
        plot.write()


def test_professor_plot_requires_input_path_for_config_loader(tmp_path: Path) -> None:
    from acib.regression.plots.professor_violin import ProfessorViolinPlotConfig

    with pytest.raises(ValueError, match="input_path"):
        ProfessorImpactViolinPlot.from_config(ProfessorViolinPlotConfig())


def test_professor_plot_loads_parquet_config(tmp_path: Path) -> None:
    input_path = tmp_path / "professors.parquet"
    professor_plot_frame().to_parquet(input_path, index=False)
    plot = ProfessorImpactViolinPlot.from_config(
        ProfessorViolinPlotConfig(input_path=input_path, output_dir=tmp_path)
    )
    assert len(plot.data.observations) == 4
    assert plot.write() == (tmp_path / "professor_title_inbreeding_violin.png",)


def test_plot_configs_use_the_configured_regression_artifact_paths(tmp_path: Path) -> None:
    from acib.regression.config import RegressionOutputs

    outputs = RegressionOutputs(directory=tmp_path / "reports")
    model9 = Model9InbreedingPredictionPlotConfig(
        input_path=outputs.regression_input, output_dir=outputs.plots_dir
    )
    violin = ProfessorViolinPlotConfig(
        input_path=outputs.professor_article_input, output_dir=outputs.plots_dir
    )
    assert model9.input_path == tmp_path / "reports" / "author_regression_input.csv"
    assert model9.output_dir == tmp_path / "reports" / "plots"
    assert violin.input_path == tmp_path / "reports" / "professor_article_input.parquet"
    assert violin.output_dir == tmp_path / "reports" / "plots"
