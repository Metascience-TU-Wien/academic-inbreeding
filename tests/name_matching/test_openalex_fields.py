# Copyright © 2026 TU Wien

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from acib.name_matching.openalex_fields import (
    OPENALEX_FIELD_IDS_COLUMN,
    OPENALEX_FIELD_LABELS_COLUMN,
    add_inferred_openalex_field_columns,
)


def test_openalex_fields_use_organization_not_relevant_names() -> None:
    researchers = pd.DataFrame(
        [
            {
                "organization_names_relevant": ["Veterinärmedizin"],
                "organization": "Fakultät Wirtschaft",
            }
        ]
    )
    result = add_inferred_openalex_field_columns(researchers)

    assert json.loads(result.loc[0, OPENALEX_FIELD_IDS_COLUMN]) == [
        "https://openalex.org/fields/14"
    ]
    assert json.loads(result.loc[0, OPENALEX_FIELD_LABELS_COLUMN]) == [
        "Business, Management and Accounting"
    ]


def test_openalex_fields_map_buehnengestaltung_to_arts_and_humanities() -> None:
    researchers = pd.DataFrame(
        [
            {
                "organization": ('[{"type": "unordered", "value": ["Bühnengestaltung"]}]'),
                "organization_names_relevant": "['Bühnengestaltung']",
            }
        ]
    )

    result = add_inferred_openalex_field_columns(researchers)

    assert json.loads(result.loc[0, OPENALEX_FIELD_IDS_COLUMN]) == [
        "https://openalex.org/fields/12"
    ]
    assert json.loads(result.loc[0, OPENALEX_FIELD_LABELS_COLUMN]) == ["Arts and Humanities"]


def test_openalex_fields_fall_back_to_organization_and_deduplicate() -> None:
    researchers = pd.DataFrame(
        [
            {
                "organization_names_relevant": "[]",
                "organization": (
                    '[{"type": "unordered", "value": '
                    '["Fakultät Wirtschaft", "Fakultät Wirtschaft"]}]'
                ),
            }
        ]
    )
    result = add_inferred_openalex_field_columns(researchers)

    assert json.loads(result.loc[0, OPENALEX_FIELD_IDS_COLUMN]) == [
        "https://openalex.org/fields/14"
    ]
    assert json.loads(result.loc[0, OPENALEX_FIELD_LABELS_COLUMN]) == [
        "Business, Management and Accounting"
    ]


def test_openalex_fields_map_multiple_parent_categories_to_distinct_ids() -> None:
    researchers = pd.DataFrame(
        [
            {
                "organization": (
                    '[{"type": "unordered", "value": ["Fakultät Wirtschaft", "Informatik"]}]'
                )
            }
        ]
    )

    result = add_inferred_openalex_field_columns(researchers)

    assert json.loads(result.loc[0, OPENALEX_FIELD_IDS_COLUMN]) == [
        "https://openalex.org/fields/14",
        "https://openalex.org/fields/17",
    ]
    assert json.loads(result.loc[0, OPENALEX_FIELD_LABELS_COLUMN]) == [
        "Business, Management and Accounting",
        "Computer Science",
    ]


def test_openalex_fields_ignore_existing_asjc_codes() -> None:
    researchers = pd.DataFrame(
        [
            {
                "inferred_asjc_parent_codes": "[3400, 3600]",
                "organization": ('[{"type": "unordered", "value": ["Fakultät Wirtschaft"]}]'),
            }
        ]
    )

    result = add_inferred_openalex_field_columns(researchers)

    assert json.loads(result.loc[0, OPENALEX_FIELD_IDS_COLUMN]) == [
        "https://openalex.org/fields/14",
    ]
    assert json.loads(result.loc[0, OPENALEX_FIELD_LABELS_COLUMN]) == [
        "Business, Management and Accounting",
    ]


def test_nanoglykobiologie_maps_to_all_approved_fields() -> None:
    researchers = pd.DataFrame(
        [{"organization": ('[{"type": "unordered", "value": ["NanoGlykobiologie"]}]')}]
    )

    result = add_inferred_openalex_field_columns(researchers)

    assert json.loads(result.loc[0, OPENALEX_FIELD_IDS_COLUMN]) == [
        "https://openalex.org/fields/13",
        "https://openalex.org/fields/15",
        "https://openalex.org/fields/16",
        "https://openalex.org/fields/24",
    ]
    assert json.loads(result.loc[0, OPENALEX_FIELD_LABELS_COLUMN]) == [
        "Biochemistry, Genetics and Molecular Biology",
        "Chemical Engineering",
        "Chemistry",
        "Immunology and Microbiology",
    ]


@pytest.mark.parametrize(
    ("organization", "expected_ids"),
    [
        ("Glykobiologie", [13]),
        ("Mikrobielle Glykobiologie", [13, 24]),
        ("Molekulare Pflanzenbiologie", [11, 13]),
        ("Fachbereich Bio Data Science", [13, 17]),
        ("Institut für Wärmetechnik", [21, 22]),
        ("Lehrstuhl für Umformtechnik", [22, 25]),
        ("Arbeitsgruppe Leichtbau- und Umformtechnologien", [22, 25]),
        ("Department Automatisierungstechnik", [17, 22]),
        ("Institut für Wissenschafts- und Technikforschung", [33]),
        ("Research Institute for International Taxation", [14, 20]),
        ("Max Perutz Labs", [13]),
        ("Max Perutz Labs 2", [13]),
        ("Max Perutz Labs - 750", [13]),
        ("Schauspiel", [12]),
        ("Saiteninstrumente", [12]),
        ("TransArts", [12]),
        ("Zeichnung und Druckgrafik", [12]),
        ("More-than-Human Design and Regenerative AI", [12, 17]),
        ("Smart Building Technologies", [22]),
        ("Institut für Tragwerksentwurf", [22]),
        ("Universitätslehrgang Intensivpflege", [29]),
        ("ULG Interpersonelle Kommunikation", [33]),
        ("Fakultät für Biologie", [11]),
        ("Applied Life Sciences", [11]),
        ("Fakultät Technik", [22]),
        ("Sprachen, Schlüsselkompetenzen und Interne Weiterbildung", [12, 33]),
        ("Universitätssport", [36]),
        ("Auswahlkommission für das Masterstudium Advanced Materials Science", [25]),
        ("Auswahlkommission für das Masterstudium Data Science", [17]),
    ],
)
def test_specific_organization_keywords_map_to_expected_fields(
    organization: str, expected_ids: list[int]
) -> None:
    researchers = pd.DataFrame(
        [
            {
                "organization": json.dumps(
                    [{"type": "ordered", "value": [organization]}],
                    ensure_ascii=False,
                )
            }
        ]
    )

    result = add_inferred_openalex_field_columns(researchers)

    assert json.loads(result.loc[0, OPENALEX_FIELD_IDS_COLUMN]) == [
        f"https://openalex.org/fields/{field_id}" for field_id in expected_ids
    ]


def test_mode_uses_university_context_without_general_mode_match() -> None:
    organizations = [
        {
            "organization": json.dumps(
                [{"type": "unordered", "value": ["Mode"]}],
                ensure_ascii=False,
            ),
            "university_name": "Uni AK Wien",
        },
        {
            "organization": json.dumps(
                [{"type": "unordered", "value": ["Mode"]}],
                ensure_ascii=False,
            ),
            "university_name": "Uni Wien",
        },
    ]

    result = add_inferred_openalex_field_columns(pd.DataFrame(organizations))

    assert json.loads(result.loc[0, OPENALEX_FIELD_IDS_COLUMN]) == [
        "https://openalex.org/fields/12"
    ]
    assert result.loc[1, OPENALEX_FIELD_IDS_COLUMN] == ""


@pytest.mark.parametrize(
    "child_units",
    [
        ["AIST"],
        ["Assistive Technology Lab"],
        ["PEEC"],
        ["AIST", "Assistive Technology Lab", "PEEC"],
    ],
)
def test_hagenberg_research_groups_use_fh_upper_austria_context(
    child_units: list[str],
) -> None:
    organization_units = ["Research Center Hagenberg", *child_units]
    organization = json.dumps(
        [{"type": "ordered", "value": organization_units}],
        ensure_ascii=False,
    )
    result = add_inferred_openalex_field_columns(
        pd.DataFrame([{"organization": organization, "university_name": "FH Oberösterreich"}])
    )

    assert json.loads(result.loc[0, OPENALEX_FIELD_IDS_COLUMN]) == [
        "https://openalex.org/fields/17"
    ]


def test_heal_uses_its_hagenberg_research_context() -> None:
    organization = json.dumps(
        [{"type": "ordered", "value": ["Research Center Hagenberg", "HEAL"]}],
        ensure_ascii=False,
    )
    result = add_inferred_openalex_field_columns(
        pd.DataFrame([{"organization": organization, "university_name": "FH Oberösterreich"}])
    )

    assert json.loads(result.loc[0, OPENALEX_FIELD_IDS_COLUMN]) == [
        "https://openalex.org/fields/17",
        "https://openalex.org/fields/22",
    ]


@pytest.mark.parametrize(
    ("organization_units", "university_name"),
    [
        (["Research Center Hagenberg", "HEAL"], "FH Joanneum"),
        (["Research Center Hagenberg", "AIST"], "FH Joanneum"),
        (["AIST"], "FH Oberösterreich"),
    ],
)
def test_hagenberg_group_rule_requires_supported_unit_and_context(
    organization_units: list[str], university_name: str
) -> None:
    organization = json.dumps(
        [{"type": "ordered", "value": organization_units}],
        ensure_ascii=False,
    )
    result = add_inferred_openalex_field_columns(
        pd.DataFrame([{"organization": organization, "university_name": university_name}])
    )

    assert result.loc[0, OPENALEX_FIELD_IDS_COLUMN] == ""


@pytest.mark.parametrize(
    "organization",
    [
        "Büro DekanIn Biologie",
        "Gebäude und Technik",
        "Haustechnik",
        "Bühnentechnik",
        "AV-Technik",
        "Senat",
        "Betriebsrat",
        "Österreichische Hochschülerschaft",
        "Projektservices / Tax",
        "Field of Expertise : Advanced Materials Science",
        "Forschungsassistenz",
        "Fachbibliothek Biologie und Botanik",
        "Personalentwicklung",
        "DLE IT Services",
        "AG Studienkommission für die Studienrichtung Biologie",
        "Auswahlkommission für das Masterstudium Informatik",
        "Referat Genderforschung",
        "Service Unit of Research Coordination",
        "Büro des Zentrums für Lehrer*innenbildung",
        "Forschung",
        "Research Institutes",
        "Department of Science and Technology",
        "Kommunikation und Öffentlichkeitsarbeit",
        "Max Perutz Labs Support GmbH",
    ],
)
def test_administrative_and_expertise_names_do_not_infer_fields(
    organization: str,
) -> None:
    researchers = pd.DataFrame(
        [
            {
                "organization": json.dumps(
                    [{"type": "unordered", "value": [organization]}],
                    ensure_ascii=False,
                )
            }
        ]
    )

    result = add_inferred_openalex_field_columns(researchers)

    assert result.loc[0, OPENALEX_FIELD_IDS_COLUMN] == ""
    assert result.loc[0, OPENALEX_FIELD_LABELS_COLUMN] == ""


def test_generic_research_assistance_and_expertise_do_not_infer_subject_fields() -> None:
    organization = json.dumps(
        [
            {
                "type": "unordered",
                "value": [
                    "Forschungsassistenz",
                    "Field of Expertise : Advanced Materials Science",
                ],
            }
        ],
        ensure_ascii=False,
    )
    result = add_inferred_openalex_field_columns(pd.DataFrame([{"organization": organization}]))

    assert result.loc[0, OPENALEX_FIELD_IDS_COLUMN] == ""
    assert result.loc[0, OPENALEX_FIELD_LABELS_COLUMN] == ""


def test_ordered_and_unordered_units_are_inferred_independently() -> None:
    researchers = pd.DataFrame(
        [
            {
                "organization": json.dumps(
                    [
                        {
                            "type": "ordered",
                            "value": [
                                "Institut für Wärmetechnik",
                                "Field of Expertise : Mobility & Production",
                            ],
                        },
                        {
                            "type": "unordered",
                            "value": [
                                "Büro DekanIn Biologie",
                                "Institut für RNA Biologie und Genomik",
                            ],
                        },
                    ],
                    ensure_ascii=False,
                )
            }
        ]
    )

    result = add_inferred_openalex_field_columns(researchers)

    assert json.loads(result.loc[0, OPENALEX_FIELD_IDS_COLUMN]) == [
        "https://openalex.org/fields/13",
        "https://openalex.org/fields/21",
        "https://openalex.org/fields/22",
    ]


def test_complete_unmatched_organization_dataset_uses_valid_field_output() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "openalex_unmatched_organization_values.csv"
    organizations = pd.read_csv(fixture_path, dtype=str, keep_default_na=False)
    result = add_inferred_openalex_field_columns(organizations)

    assert len(result) == 2483
    assert organizations["organization"].str.contains('"type": "ordered"').any()
    assert organizations["organization"].str.contains('"type": "unordered"').any()
    assert len(result) == len(organizations)

    for row in result.itertuples(index=False):
        ids = json.loads(getattr(row, OPENALEX_FIELD_IDS_COLUMN) or "[]")
        labels = json.loads(getattr(row, OPENALEX_FIELD_LABELS_COLUMN) or "[]")
        assert len(ids) == len(labels)
        assert len(ids) == len(set(ids))
        assert all(
            field_id.startswith("https://openalex.org/fields/")
            and 11 <= int(field_id.rsplit("/", maxsplit=1)[1]) <= 36
            for field_id in ids
        )


def test_openalex_fields_preserve_supplied_evidence_and_leave_unmapped_blank() -> None:
    researchers = pd.DataFrame(
        [
            {
                "organization": "Fakultät Wirtschaft",
                OPENALEX_FIELD_IDS_COLUMN: '["https://openalex.org/fields/99"]',
                OPENALEX_FIELD_LABELS_COLUMN: '["User supplied"]',
            },
            {
                "organization_names_relevant": None,
                "organization": "Universitätsverwaltung",
            },
        ]
    )
    original = researchers.copy(deep=True)

    result = add_inferred_openalex_field_columns(researchers)

    assert result.loc[0, OPENALEX_FIELD_IDS_COLUMN] == '["https://openalex.org/fields/99"]'
    assert result.loc[0, OPENALEX_FIELD_LABELS_COLUMN] == '["User supplied"]'
    assert pd.isna(result.loc[1, OPENALEX_FIELD_IDS_COLUMN])
    assert pd.isna(result.loc[1, OPENALEX_FIELD_LABELS_COLUMN])
    pd.testing.assert_frame_equal(researchers, original)
