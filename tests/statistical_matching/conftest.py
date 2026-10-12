# Copyright © 2026 TU Wien

"""Deterministic data and configuration helpers for matching tests."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from acib.statistical_matching.config import (
    StatisticalMatchingConfig,
    StatisticalMatchingOutputs,
)


@pytest.fixture
def synthetic_matching_data() -> pd.DataFrame:
    """Return the source project's deterministic statistical-matching fixture."""
    rows = []
    titles = ["Full Prof.", "Ao. Prof.", "Ass. Prof.", "FH-Prof."]
    fields = ["Engineering", "Medicine"]
    for title_position, title in enumerate(titles):
        for field_position, field in enumerate(fields):
            for age in [12, 18, 28]:
                for first_year in [1995, 2012]:
                    for is_inbred in [False, True]:
                        for replicate in range(2):
                            treatment_effect = -3.0 if is_inbred else 0.0
                            pp_effect = -0.02 if is_inbred else 0.0
                            rows.append(
                                {
                                    "is_inbred": is_inbred,
                                    "academic_age": age + replicate,
                                    "title": title,
                                    "primary_field": field,
                                    "first_year": first_year,
                                    "n_public": 40 + age * 2 + treatment_effect,
                                    "hazen_perc_med": 50.0
                                    + title_position
                                    + field_position
                                    + age * 0.1
                                    + treatment_effect,
                                    "pp_top_10": 0.12 + age * 0.001 + pp_effect,
                                    "pp_top_5": 0.06 + age * 0.0005 + pp_effect / 2,
                                    "pp_top_1": 0.012 + age * 0.0001 + pp_effect / 10,
                                    "P_top10": 2.0 + age * 0.05 + treatment_effect,
                                    "P_top5": 1.0 + age * 0.025 + treatment_effect / 2,
                                    "P_top1": 0.2 + age * 0.01 + treatment_effect / 10,
                                }
                            )
    rows.append(
        {
            "is_inbred": True,
            "academic_age": 18,
            "title": "FH-Ass. Prof.",
            "primary_field": "Medicine",
            "first_year": 2012,
            "n_public": 80.0,
            "hazen_perc_med": 99.0,
            "pp_top_10": 0.99,
            "pp_top_5": 0.99,
            "pp_top_1": 0.99,
            "P_top10": 99.0,
            "P_top5": 99.0,
            "P_top1": 99.0,
        }
    )
    return pd.DataFrame(rows)


def matching_config(tmp_path: Path, **kwargs: Any) -> StatisticalMatchingConfig:
    """Build config with distinct temporary artifact paths."""
    outputs = StatisticalMatchingOutputs(directory=tmp_path / "outputs")
    return StatisticalMatchingConfig(input_path=tmp_path / "input.csv", outputs=outputs, **kwargs)
