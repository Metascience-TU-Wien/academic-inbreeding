# Copyright © 2026 TU Wien

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def researcher_names() -> pd.DataFrame:
    """Provide deterministic researcher names for matching strategy tests."""
    return pd.DataFrame(
        {
            "sc_id": [1, 2, 3, 4],
            "researcher_name": [
                "smith, anna",
                "doe, john michael",
                "smith, anna maria claudia elise",
                "windisch gratz, michala",
            ],
        }
    )


@pytest.fixture
def author_names() -> pd.DataFrame:
    """Provide known author names exercising the four matching priorities."""
    return pd.DataFrame(
        {
            "author_id": [10, 20, 30, 40],
            "eid": ["P1", "P2", "P3", "P4"],
            "author_name": [
                "smith, anna",
                "doe, john",
                "smith, anna maria claudia elisa",
                "windischgratz, michala",
            ],
            "affiliation_id": [60018163, 60018163, 60018163, 60018163],
        }
    )
