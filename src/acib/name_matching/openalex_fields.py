# Copyright © 2026 TU Wien

"""Adapt researcher organization metadata to OpenAlex Field evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache

import pandas as pd

from acib.common.subjects import (
    keyword_matches,
    normalize_organization_name,
    organization_names,
)

OPENALEX_FIELD_IDS_COLUMN = "inferred_openalex_field_ids"
OPENALEX_FIELD_LABELS_COLUMN = "inferred_openalex_field_labels"
_RELEVANT_ORGANIZATION_COLUMN = "organization_names_relevant"
_ORGANIZATION_COLUMN = "organization"


@dataclass(frozen=True, slots=True)
class _OrganizationFieldRule:
    """Match organization-name keywords directly to one OpenAlex Field."""

    field: tuple[str, str]
    keywords: tuple[str, ...]


_FIELD_RULES: tuple[_OrganizationFieldRule, ...] = (
    _OrganizationFieldRule(
        ("https://openalex.org/fields/11", "Agricultural and Biological Sciences"),
        (
            "agriculture",
            "agrar",
            "landwirtschaft",
            "agronom",
            "forst",
            "wald",
            "pflanzenbau",
            "tierzucht",
            "nutztier",
            "zoolog",
            "food science",
            "lebensmittelwissenschaft",
            "horticulture",
        ),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/12", "Arts and Humanities"),
        (
            "kunst",
            "kulturwissenschaft",
            "theologie",
            "religion",
            "philosophie",
            "geschichte",
            "germanistik",
            "anglistik",
            "musik",
            "theater",
            "buhnengestaltung",
            "buehnengestaltung",
            "stage design",
            "literatur",
            "archiv",
            "konservierung",
            "restaurierung",
        ),
    ),
    _OrganizationFieldRule(
        (
            "https://openalex.org/fields/13",
            "Biochemistry, Genetics and Molecular Biology",
        ),
        (
            "biochem",
            "biotechnolog",
            "genetik",
            "genetics",
            "molecular biology",
            "molekularbiolog",
            "zellbiolog",
            "cell biology",
        ),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/14", "Business, Management and Accounting"),
        (
            "wirtschaft",
            "business",
            "betriebswirtschaft",
            "management",
            "leadership",
            "human resource",
            "personal development",
            "marketing",
            "entrepreneurship",
            "accounting",
            "rechnungswesen",
            "controlling",
            "innovation",
        ),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/15", "Chemical Engineering"),
        ("chemical engineering", "verfahrenstechnik", "bioverfahrenstechnik"),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/16", "Chemistry"),
        ("chemie", "chemistry", "chemical science"),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/17", "Computer Science"),
        (
            "informatik",
            "computer science",
            "software",
            "artificial intelligence",
            "information technology",
            "computer vision",
        ),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/18", "Decision Sciences"),
        (
            "decision science",
            "operations research",
            "operation research",
            "entscheidungswissenschaft",
            "analytics",
        ),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/19", "Earth and Planetary Sciences"),
        (
            "erdwissenschaft",
            "geologie",
            "geology",
            "geographie",
            "geography",
            "meteorologie",
            "klima",
            "planetary science",
            "geophysik",
            "geochemie",
        ),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/20", "Economics, Econometrics and Finance"),
        ("volkswirtschaft", "economics", "econometrics", "finance", "finanz", "bank"),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/21", "Energy"),
        ("energy", "energie", "renewable", "erneuerbar", "kreislaufwirtschaft"),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/22", "Engineering"),
        (
            "engineering",
            "ingenieur",
            "architektur",
            "maschinenbau",
            "mechatronik",
            "elektrotechnik",
            "werkstoff",
            "construction",
            "hochbau",
            "geotechnik",
            "raumplanung",
            "landtechnik",
        ),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/23", "Environmental Science"),
        (
            "umwelt",
            "environment",
            "nachhaltigkeit",
            "sustainability",
            "naturschutz",
            "ökologie",
            "oekologie",
            "ecology",
            "abfall",
            "wasser",
            "water science",
        ),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/24", "Immunology and Microbiology"),
        ("immunolog", "mikrobiolog", "microbiolog", "virolog", "bakteriolog"),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/25", "Materials Science"),
        ("material science", "materialwissenschaft", "werkstoff", "polymer"),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/26", "Mathematics"),
        ("mathematik", "mathematics", "statistik", "statistics"),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/27", "Medicine"),
        (
            "medizin",
            "medicine",
            "klinik",
            "clinical",
            "chirurgie",
            "radiologie",
            "pathologie",
            "public health",
            "gesundheit",
        ),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/28", "Neuroscience"),
        ("neuro", "neurowissenschaft"),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/29", "Nursing"),
        ("pflegewissenschaft", "nursing", "krankenpflege"),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/30", "Pharmacology, Toxicology and Pharmaceutics"),
        ("pharmakolog", "pharmacolog", "pharmazie", "pharmacy", "toxikolog", "toxicolog"),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/31", "Physics and Astronomy"),
        ("physik", "physics", "astronomie", "astronomy", "astrophysik"),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/32", "Psychology"),
        ("psycholog", "psychology"),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/33", "Social Sciences"),
        (
            "soziologie",
            "sociology",
            "politikwissenschaft",
            "political science",
            "kommunikationswissenschaft",
            "communication",
            "pädagog",
            "padagog",
            "education",
            "recht",
            "law",
            "transport",
            "gender studies",
            "sozialwissenschaft",
        ),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/34", "Veterinary"),
        ("veterinary", "veterinaer", "veterinarmedizin", "vet med", "vetmed", "tiermedizin"),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/35", "Dentistry"),
        ("dentistry", "dental", "zahnmedizin", "zahnheilkunde"),
    ),
    _OrganizationFieldRule(
        ("https://openalex.org/fields/36", "Health Professions"),
        (
            "health profession",
            "health sciences",
            "gesundheitsberuf",
            "physiotherapie",
            "ergotherapie",
            "medizinische technik",
        ),
    ),
)
_VETERINARY_FIELD = ("https://openalex.org/fields/34", "Veterinary")
_VETERINARY_TERMS = (
    "veterinary",
    "veterinaer",
    "veterinarmedizin",
    "vet med",
    "vetmed",
    "tiermedizin",
)


def add_inferred_openalex_field_columns(researchers: pd.DataFrame) -> pd.DataFrame:
    """Add OpenAlex Field IDs and labels inferred from researcher organizations.

    Relevant organization names are preferred and the original organization
    value is the fallback. Matching rules map organization text directly to
    OpenAlex Field IDs; no ASJC-code columns or intermediate taxonomy are used.
    Existing nonblank OpenAlex Field values are preserved.

    :param researchers: Researcher metadata rows.
    :return: A copy with inferred OpenAlex Field ID and label columns.
    """
    result = researchers.copy()
    if OPENALEX_FIELD_IDS_COLUMN not in result:
        result[OPENALEX_FIELD_IDS_COLUMN] = ""
    if OPENALEX_FIELD_LABELS_COLUMN not in result:
        result[OPENALEX_FIELD_LABELS_COLUMN] = ""

    for index, row in result.iterrows():
        if not _is_blank(row[OPENALEX_FIELD_IDS_COLUMN]):
            continue
        names = _organization_names(row.get(_RELEVANT_ORGANIZATION_COLUMN))
        if not names:
            names = _organization_names(row.get(_ORGANIZATION_COLUMN))
        if not names:
            continue
        fields = _infer_fields(tuple(names))
        if not fields:
            continue

        result.at[index, OPENALEX_FIELD_IDS_COLUMN] = json.dumps(
            [field[0] for field in fields], ensure_ascii=False
        )
        if _is_blank(row[OPENALEX_FIELD_LABELS_COLUMN]):
            result.at[index, OPENALEX_FIELD_LABELS_COLUMN] = json.dumps(
                [field[1] for field in fields], ensure_ascii=False
            )
    return result


def _organization_names(value: object) -> list[str]:
    """Extract usable organization names from one source cell."""
    if _is_blank(value):
        return []
    if isinstance(value, str):
        return list(_parse_organization_names(value))
    return _filter_organization_names(organization_names(value))


@cache
def _parse_organization_names(value: str) -> tuple[str, ...]:
    """Parse and cache repeated serialized organization cells."""
    return tuple(_filter_organization_names(organization_names(value)))


def _filter_organization_names(names: list[str]) -> list[str]:
    """Remove placeholder values from parsed organization names."""
    return [
        name
        for name in names
        if normalize_organization_name(name) not in {"", "[]", "none", "null"}
    ]


@cache
def _infer_fields(names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Infer and stably order Fields for one distinct set of organization names."""
    normalized_names = [normalize_organization_name(name) for name in names]
    combined_names = " | ".join(normalized_names)
    fields = {
        rule.field
        for rule in _FIELD_RULES
        if any(keyword_matches(keyword, combined_names) for keyword in rule.keywords)
    }
    is_veterinary = any(
        term in normalized_name
        for term in _VETERINARY_TERMS
        for normalized_name in normalized_names
    )
    if is_veterinary:
        fields.discard(("https://openalex.org/fields/27", "Medicine"))
        fields.add(_VETERINARY_FIELD)
    return tuple(sorted(fields, key=lambda field: field[0]))


def _is_blank(value: object) -> bool:
    """Return whether a metadata cell contains no meaningful value."""
    if value is None or value is pd.NA:
        return True
    if not isinstance(value, (list, dict, tuple, set)) and pd.isna(value):
        return True
    return str(value).strip().casefold() in {"", "nan", "none", "null", "[]", "<na>"}
