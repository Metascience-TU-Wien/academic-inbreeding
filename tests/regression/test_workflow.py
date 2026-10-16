# Copyright © 2026 TU Wien

from __future__ import annotations

from pathlib import Path

import pandas as pd

from acib.regression.config import (
    RegressionConfig,
    RegressionModelConfig,
    RegressionOutputs,
)
from acib.regression.workflow import run_regression_suite


def test_workflow_reuses_valid_regression_input_and_writes_latex(tmp_path: Path) -> None:
    """Reuse a canonical researcher artifact without requiring external rebuild inputs.

    :param tmp_path: Temporary output location.
    """
    outputs = RegressionOutputs(directory=tmp_path / "regression")
    outputs.directory.mkdir()
    pd.DataFrame(
        {
            "sc_id": range(8),
            "pp_top_10": [0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.5, 0.6],
            "academic_age": range(10, 18),
            "is_inbred": [True, False] * 4,
            "title": ["Full Prof."] * 8,
        }
    ).to_csv(outputs.regression_input, index=False)
    config = RegressionConfig(
        article_author_file_path=tmp_path / "not-required-articles.parquet",
        mobility_author_proxy=tmp_path / "not-required-mobility.csv",
        outputs=outputs,
        dependent_variables={"pp_top_10": "ols"},
        model_specs=(RegressionModelConfig("custom", ("academic_age",)),),
        exclude_titles=(),
        rebuild_input=False,
    )

    result = run_regression_suite(config)

    assert result.fit_statistics.loc[0, "n_obs"] == 8
    assert outputs.regression_input.is_file()
    assert (outputs.latex_dir / "descriptive_statistics.tex").is_file()
