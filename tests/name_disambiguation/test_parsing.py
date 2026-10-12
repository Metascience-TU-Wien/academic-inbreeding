# Copyright © 2026 TU Wien

import pandas as pd

from acib.name_disambiguation.data import prepare_disambiguation_author_frame
from acib.name_disambiguation.parsing import (
    normalize_initials,
    parse_embedded_json,
    parse_publication_range,
    parse_subject_areas,
    prepare_author_frame,
)
from acib.name_disambiguation.scoring import score_all_pairs


def test_parse_embedded_json_handles_double_encoded_payload() -> None:
    payload = '"[{\\"area\\": \\"Software\\", \\"code\\": 1712}]"'
    assert parse_embedded_json(payload) == [{"area": "Software", "code": 1712}]


def test_parse_subject_areas_normalizes_integral_codes_and_missing_zero() -> None:
    areas = parse_subject_areas(
        '[{"area":"Pharmacology","code":3004.0},'
        ' {"area":"Medicine","code":"2700.0"},'
        ' {"area":"Unknown","code":0}]'
    )
    assert [area.code for area in areas] == [3004, 2700, None]


def test_parse_subject_areas_accepts_python_literals_and_structured_values() -> None:
    areas = parse_subject_areas([{"area": "Safety, Risk, Reliability and Quality", "code": 2213.0}])
    assert [(area.area, area.code) for area in areas] == [
        ("Safety, Risk, Reliability and Quality", 2213)
    ]
    assert parse_subject_areas("[{'area': 'Safety', 'code': 2213}]")[0].code == 2213


def test_publication_range_and_initial_placeholder_fallback() -> None:
    assert parse_publication_range("{2003,2026}").start == 2003
    assert parse_publication_range("{2003,2026}").end == 2026
    assert parse_publication_range("invalid").start is None
    assert normalize_initials("FALSE") == ""
    prepared = prepare_author_frame(
        pd.DataFrame(
            [
                {
                    "author_id": "1",
                    "name": "Müller, Josef S.",
                    "surname": "Müller",
                    "given_name": "Josef S.",
                    "initials": "FALSE",
                    "indexed_name": "Müller J.S.",
                    "publication_range": "",
                    "name_variants": "[]",
                    "subject_areas": "[]",
                }
            ]
        )
    )
    assert prepared.loc[0, "canonical_initials"] == "JS"


def test_prepare_disambiguation_frame_accepts_nullable_integer_and_string_columns() -> None:
    source = pd.DataFrame(
        {
            "author_id": pd.Series([1, pd.NA], dtype="Int64"),
            "name": pd.Series(["Müller, Thomas", pd.NA], dtype="string"),
            "surname": ["Müller", pd.NA],
            "given_name": ["Thomas", pd.NA],
            "indexed_name": ["Müller T", pd.NA],
            "initials": ["T", pd.NA],
            "publication_range": ["{2020,2024}", pd.NA],
            "name_variants": ["[]", pd.NA],
            "subject_areas": ["[]", pd.NA],
        }
    )
    prepared = prepare_disambiguation_author_frame(source)
    assert prepared["author_id"].tolist() == ["1", ""]
    assert prepared.loc[1, "canonical_name"] == ""
    assert prepared.loc[1, "subject_areas_parsed"] == []


def test_preparation_keeps_missing_orcid_empty_and_never_scores_a_false_match() -> None:
    authors = pd.DataFrame(
        [
            {
                "author_id": "1",
                "name": "Leitner, Karl Heinz",
                "surname": "Leitner",
                "given_name": "Karl Heinz",
                "indexed_name": "Leitner K.H.",
                "initials": "KH",
                "publication_range": "",
                "name_variants": "[]",
                "subject_areas": "[]",
                "orcid": pd.NA,
            },
            {
                "author_id": "2",
                "name": "Leitner, H.",
                "surname": "Leitner",
                "given_name": "H.",
                "indexed_name": "Leitner H.",
                "initials": "H",
                "publication_range": "",
                "name_variants": "[]",
                "subject_areas": "[]",
                "orcid": pd.NA,
            },
        ]
    )
    prepared = prepare_disambiguation_author_frame(authors)
    decisions = score_all_pairs(prepared)
    labels = {
        component["label"] for components in decisions["components"] for component in components
    }
    assert prepared["orcid"].tolist() == ["", ""]
    assert "orcid_match" not in labels
