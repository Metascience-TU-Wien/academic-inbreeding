# Copyright © 2026 TU Wien

import warnings
from pathlib import Path

import pandas as pd

from acib.mobility.workflow import _read_dataframe


def test_csv_reader_uses_stable_inference_and_preserves_identifier_strings(
    tmp_path: Path,
) -> None:
    path = tmp_path / "researcher_metadata.csv"
    pd.DataFrame(
        {
            "sc_id": [1, 2],
            "person_functions": ["2", '["Professor", "Dean"]'],
            "institute_link": ["12", "https://example.test/institute"],
            "person_location_link": ["", "https://example.test/person"],
            "orcid": ["0000-0001", "0000-0002"],
        }
    ).to_csv(path, index=False)

    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", pd.errors.DtypeWarning)
        result = _read_dataframe(
            path,
            dtype={
                "person_functions": "string",
                "institute_link": "string",
                "person_location_link": "string",
                "orcid": "string",
            },
        )

    assert not [
        warning for warning in observed if issubclass(warning.category, pd.errors.DtypeWarning)
    ]
    assert result["orcid"].tolist() == ["0000-0001", "0000-0002"]
    assert result["person_functions"].tolist() == ["2", '["Professor", "Dean"]']
    assert result["sc_id"].dtype == "int64"
