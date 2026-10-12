# Copyright © 2026 TU Wien

from __future__ import annotations

import pandas as pd

from acib.regression.build import prepare_table_for_output


def test_prepare_table_for_output_serializes_nested_values() -> None:
    data = pd.DataFrame(
        {"eid": ["p1"], "article_general_asjc_subject_areas": [{2700: "General Medicine"}]}
    )
    result = prepare_table_for_output(data)
    assert result.loc[0, "article_general_asjc_subject_areas"] == '{"2700": "General Medicine"}'
