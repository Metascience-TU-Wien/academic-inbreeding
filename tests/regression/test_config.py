# Copyright © 2026 TU Wien

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaError

from acib.pipeline.config import PROJECT_ROOT, load_regression_config
from acib.regression.schemas import RegressionInputOutputSchema


def test_config_resolves_canonical_inputs_and_outputs_without_downstream_stages(
    tmp_path: Path,
) -> None:
    article_author_path = tmp_path / "external" / "articles.parquet"
    impact_path = tmp_path / "external" / "impact.csv"
    config_file = tmp_path / "pipeline.toml"
    config_file.write_text(
        f"""[data]
article_author_file_path = "{article_author_path}"
article_impact_file_path = "{impact_path}"

[mobility.outputs]
directory = "data/test-mobility"

[regression]
rebuild_input = false

[regression.dependent_variables]
pp_top_10 = "ols"

[regression.outcome_metadata.pp_top_10]
outcome_type = "relative"
alias = "PP top ten percent"

[regression.outputs]
directory = "data/test-regression"
""",
        encoding="utf-8",
    )

    config = load_regression_config(config_file)

    assert config.article_author_file_path == article_author_path
    assert config.article_impact_file_path == impact_path
    assert config.mobility_author_proxy == (
        PROJECT_ROOT / "data/test-mobility/author_inbreeding_proxy.csv"
    )
    assert config.outputs.regression_input == (
        PROJECT_ROOT / "data/test-regression/author_regression_input.csv"
    )
    assert config.outputs.plots_dir == PROJECT_ROOT / "data/test-regression/plots"
    assert config.dependent_variables == {"pp_top_10": "ols"}
    assert "statistical_matching" not in config_file.read_text(encoding="utf-8")
    assert "mobility_impact" not in config_file.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"sc_id": [1, 1], "is_inbred": [False, True], "title": ["Full Prof."] * 2}),
        pd.DataFrame({"sc_id": [1], "is_inbred": ["True"], "title": ["Full Prof."]}),
    ],
    ids=("duplicate-researcher-id", "non-boolean-inbreeding-status"),
)
def test_regression_input_contract_rejects_invalid_researcher_rows(frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError):
        RegressionInputOutputSchema.validate(frame)
