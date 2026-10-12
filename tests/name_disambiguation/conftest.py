# Copyright © 2026 TU Wien

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def author_frame() -> pd.DataFrame:
    """Return three small candidate records with a clear identity pair."""
    return pd.DataFrame(
        [
            {
                "sc_id": "1",
                "_id": "row-1",
                "author_id": "10",
                "eid": "P1",
                "name": "Müller, Thomas",
                "surname": "Müller",
                "given_name": "Thomas",
                "indexed_name": "Müller T",
                "initials": "T",
                "publication_range": "{2020,2024}",
                "name_variants": "[]",
                "subject_areas": '[{"area": "Software", "code": 1712}]',
            },
            {
                "sc_id": "1",
                "_id": "row-2",
                "author_id": "20",
                "eid": "P2",
                "name": "Müller, Thomas",
                "surname": "Müller",
                "given_name": "Thomas",
                "indexed_name": "Müller T",
                "initials": "T",
                "publication_range": "{2021,2025}",
                "name_variants": "[]",
                "subject_areas": '[{"area": "Artificial Intelligence", "code": 1702}]',
            },
            {
                "sc_id": "2",
                "_id": "row-3",
                "author_id": "30",
                "eid": "P3",
                "name": "Schmidt, Eva",
                "surname": "Schmidt",
                "given_name": "Eva",
                "indexed_name": "Schmidt E",
                "initials": "E",
                "publication_range": "{2019,2022}",
                "name_variants": "[]",
                "subject_areas": '[{"area": "General Medicine", "code": 2700}]',
            },
        ]
    )
