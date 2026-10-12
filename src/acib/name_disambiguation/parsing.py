# Copyright © 2026 TU Wien

"""Parsing and canonicalization of Scopus author records."""

from __future__ import annotations

import json
import math
import re
from ast import literal_eval
from collections.abc import Mapping
from typing import Any

import pandas as pd

from acib.common.names import split_name
from acib.constants import (
    AUTHOR_ID,
    GIVEN_NAME,
    HISTORICAL_IDENTIFIER,
    SUBJECT_AREAS,
    SURNAME,
)
from acib.name_disambiguation.constants import (
    CANONICAL_GIVEN_NAME,
    CANONICAL_INDEXED_NAME,
    CANONICAL_INITIALS,
    CANONICAL_NAME,
    CANONICAL_SURNAME,
    DOCUMENT_COUNT,
    NAME_VARIANTS_PARSED,
    ORCID,
    PUBLICATION_END,
    PUBLICATION_START,
    SUBJECT_AREA_LABELS,
    SUBJECT_AREAS_PARSED,
)
from acib.name_disambiguation.models import (
    AuthorRecord,
    NameVariant,
    PublicationRange,
    SubjectArea,
)
from acib.name_disambiguation.normalization import (
    normalize_disambiguation_name as normalize_person_name,
)
from acib.name_matching.constants import (
    AUTHOR_INTERNAL_ID,
    INDEXED_NAME,
    INITIALS,
    NAME,
    NAME_VARIANTS,
    PUBLICATION_RANGE,
)

_INITIALS_PLACEHOLDERS = {"", "false", "true", "nan", "none", "null", "undefined", "n/a", "na"}
_RAW_RECORD_COLUMN = "raw_record"
_PUBLICATION_RANGE_PARSED_COLUMN = "publication_range_parsed"
_CITATION_COUNT_COLUMN = "citation_count"
_CITED_BY_COUNT_COLUMN = "cited_by_count"
_H_INDEX_COLUMN = "h_index"
_COAUTHOR_COUNT_COLUMN = "coauthor_count"
_PARENT_ID_COLUMN = "parent_id"


def prepare_author_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add parsed and canonical columns to an author frame.

    :param frame: Source or matched author-profile rows.
    :return: Copy containing canonical names and parsed evidence fields.
    """
    prepared = frame.copy()
    prepared[_RAW_RECORD_COLUMN] = prepared.apply(normalize_row, axis=1)
    parts = prepared.apply(
        lambda row: canonical_name_parts(
            row.get(NAME, ""), row.get(SURNAME, ""), row.get(GIVEN_NAME, "")
        ),
        axis=1,
        result_type="expand",
    )
    prepared[[CANONICAL_SURNAME, CANONICAL_GIVEN_NAME]] = parts
    prepared[CANONICAL_NAME] = prepared.apply(
        lambda row: compose_canonical_full_name(row[CANONICAL_SURNAME], row[CANONICAL_GIVEN_NAME]),
        axis=1,
    )
    prepared[CANONICAL_INDEXED_NAME] = prepared[INDEXED_NAME].map(normalize_person_name)
    prepared[CANONICAL_INITIALS] = prepared.apply(
        lambda row: normalize_initials(row.get(INITIALS, ""))
        or initials_from_given_name(row[CANONICAL_GIVEN_NAME]),
        axis=1,
    )
    prepared[_PUBLICATION_RANGE_PARSED_COLUMN] = prepared[PUBLICATION_RANGE].map(
        parse_publication_range
    )
    prepared[PUBLICATION_START] = prepared[_PUBLICATION_RANGE_PARSED_COLUMN].map(
        lambda item: item.start
    )
    prepared[PUBLICATION_END] = prepared[_PUBLICATION_RANGE_PARSED_COLUMN].map(
        lambda item: item.end
    )
    prepared[NAME_VARIANTS_PARSED] = prepared[NAME_VARIANTS].map(parse_name_variants)
    prepared[SUBJECT_AREAS_PARSED] = prepared[SUBJECT_AREAS].map(parse_subject_areas)
    prepared[SUBJECT_AREA_LABELS] = prepared[SUBJECT_AREAS_PARSED].map(
        lambda areas: sorted({area.abbreviation or area.area for area in areas})
    )
    return prepared


def author_records_from_frame(frame: pd.DataFrame) -> list[AuthorRecord]:
    """Convert prepared rows into typed author records.

    :param frame: Prepared author dataframe.
    :return: Parsed records in dataframe row order.
    """
    return frame.apply(author_from_row, axis=1).tolist()


def normalize_row(row: Mapping[str, Any]) -> dict[str, str]:
    """Convert row values to strings, representing missing values as empty.

    :param row: Source record mapping.
    :return: String-valued record with missing fields represented as empty text.
    """
    normalized: dict[str, str] = {}
    for key, value in row.items():
        try:
            missing = bool(pd.isna(value))
        except Exception:  # noqa: BLE001 - object-valued cells may make pd.isna ambiguous
            missing = False
        normalized[str(key)] = "" if missing else str(value)
    return normalized


def author_from_row(row: Mapping[str, Any]) -> AuthorRecord:
    """Build an author record from one source row.

    :param row: Source or prepared author row.
    :return: Parsed author record with normalized names and evidence.
    """
    values = normalize_row(row)
    name, surname, given = values.get(NAME, ""), values.get(SURNAME, ""), values.get(GIVEN_NAME, "")
    indexed, initials = values.get(INDEXED_NAME, ""), values.get(INITIALS, "")
    canonical_surname, canonical_given = canonical_name_parts(name, surname, given)
    return AuthorRecord(
        row_id=values.get(AUTHOR_INTERNAL_ID, ""),
        author_id=values.get(AUTHOR_ID, ""),
        name=name,
        surname=surname,
        given_name=given,
        indexed_name=indexed,
        initials=initials,
        canonical_name=compose_canonical_full_name(canonical_surname, canonical_given),
        canonical_surname=canonical_surname,
        canonical_given_name=canonical_given,
        canonical_indexed_name=normalize_person_name(indexed),
        canonical_initials=normalize_initials(initials)
        or initials_from_given_name(canonical_given),
        orcid=values.get(ORCID, ""),
        document_count=parse_int(values.get(DOCUMENT_COUNT)),
        citation_count=parse_int(values.get(_CITATION_COUNT_COLUMN)),
        cited_by_count=parse_int(values.get(_CITED_BY_COUNT_COLUMN)),
        h_index=parse_int(values.get(_H_INDEX_COLUMN)),
        coauthor_count=parse_int(values.get(_COAUTHOR_COUNT_COLUMN)),
        publication_range=parse_publication_range(values.get(PUBLICATION_RANGE, "")),
        name_variants=parse_name_variants(values.get(NAME_VARIANTS, "")),
        subject_areas=parse_subject_areas(values.get(SUBJECT_AREAS, "")),
        parent_id=values.get(_PARENT_ID_COLUMN, ""),
        historical_identifier=values.get(HISTORICAL_IDENTIFIER, ""),
        raw=values,
    )


def parse_int(value: str | None) -> int | None:
    """Parse an integer or return ``None`` for empty or malformed input.

    :param value: Text to parse.
    :return: Parsed integer, or ``None`` when parsing fails.
    """
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_name_variants(value: str) -> list[NameVariant]:
    """Parse serialized name variants and canonicalize each entry.

    :param value: JSON or Python-literal representation of name variants.
    :return: Parsed variants in source order.
    """
    payload = parse_embedded_json(value)
    if not isinstance(payload, list):
        return []
    variants = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        indexed, initials = str(entry.get("indexed_name", "")), str(entry.get("initials", ""))
        surname, given = str(entry.get("surname", "")), str(entry.get("given_name", ""))
        cs, cg = canonical_name_parts(indexed, surname, given)
        variants.append(
            NameVariant(
                indexed,
                initials,
                surname,
                given,
                cs,
                cg,
                compose_canonical_full_name(cs, cg),
                normalize_person_name(indexed),
                normalize_initials(initials) or initials_from_given_name(cg),
                parse_int(str(entry.get("doc_count", ""))),
            )
        )
    return variants


def parse_subject_areas(value: Any) -> list[SubjectArea]:
    """Parse serialized Scopus subject areas.

    :param value: JSON-like subject-area data.
    :return: Parsed subject-area records.
    """
    payload = parse_embedded_json(value)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, (list, tuple)):
        return []
    return [
        SubjectArea(
            string_value(item.get("area")),
            string_value(item.get("abbreviation")),
            parse_subject_area_code(item.get("code")),
        )
        for item in payload
        if isinstance(item, dict)
    ]


def parse_subject_area_code(value: Any) -> int | None:
    """Parse a finite integral subject-area code, excluding zero.

    :param value: Candidate numeric code.
    :return: Nonzero integer code, or ``None`` for invalid input.
    """
    if value is None:
        return None
    code: int | None
    if isinstance(value, int):
        code = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        code = int(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        code = int(number) if math.isfinite(number) and number.is_integer() else None
    else:
        return None
    return code if code else None


def string_value(value: Any) -> str:
    """Convert a nonmissing value to text.

    :param value: Source value.
    :return: String form or empty text for missing values.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def parse_publication_range(value: str) -> PublicationRange:
    """Extract the first one or two four-digit years from a range string.

    :param value: Source publication-range text.
    :return: Inclusive parsed year bounds, or an empty range.
    """
    matches = re.findall(r"\d{4}", value or "")
    if not matches:
        return PublicationRange()
    years = [int(item) for item in matches]
    return PublicationRange(years[0], years[1] if len(years) > 1 else years[0])


def parse_embedded_json(value: Any) -> Any:
    """Parse JSON, JSON-encoded strings, or Python literal representations.

    :param value: Serialized or already structured value.
    :return: Parsed object, or ``None`` for missing/invalid input.
    """
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        return value
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    candidate = str(value).strip()
    for _ in range(3):
        if not candidate:
            return None
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            break
        if isinstance(parsed, str):
            candidate = parsed.strip()
            continue
        return parsed
    try:
        return literal_eval(candidate)
    except (SyntaxError, ValueError):
        return None


def canonical_name_parts(name: str, surname: str, given_name: str) -> tuple[str, str]:
    """Prefer explicit surname/given fields, falling back to comma-form name.

    :param name: Combined source name.
    :param surname: Explicit surname field.
    :param given_name: Explicit given-name field.
    :return: Canonical surname and given-name pair.
    """
    canonical_surname, canonical_given = normalize_person_name(surname), normalize_person_name(
        given_name
    )
    parsed_surname, parsed_given = parse_name(name)
    return canonical_surname or parsed_surname, canonical_given or parsed_given


def parse_name(name: str) -> tuple[str, str]:
    """Parse and normalize a surname-first comma-separated name.

    :param name: Combined name text.
    :return: Canonical surname and given-name pair.
    """
    surname, given = split_name(name)
    return normalize_person_name(surname or ""), normalize_person_name(given or "")


def compose_canonical_full_name(surname: str, given_name: str) -> str:
    """Join canonical name parts using the legacy surname-first format.

    :param surname: Canonical surname.
    :param given_name: Canonical given name.
    :return: Combined name or the available part.
    """
    return f"{surname}, {given_name}" if surname and given_name else surname or given_name


def normalize_initials(value: object) -> str:
    """Normalize valid initials and reject placeholders or full names.

    :param value: Source initials.
    :return: Uppercase alphabetic initials, or empty text when invalid.
    """
    if value is None or isinstance(value, bool) or pd.isna(value):
        return ""
    raw = str(value).strip()
    normalized = normalize_person_name(raw).strip()
    if normalized.lower() in _INITIALS_PLACEHOLDERS:
        return ""
    tokens = [token for token in re.split(r"[.\s]+", raw) if token]
    if not tokens or (len(tokens) > 1 and any(len(token) != 1 for token in tokens)):
        return ""
    if len(tokens) == 1 and len(tokens[0]) > 1 and not tokens[0].isupper():
        return ""
    return "".join(
        char for char in normalize_person_name("".join(tokens)).upper() if char.isalpha()
    )


def initials_from_given_name(given_name: str) -> str:
    """Return uppercase initials from canonical given-name tokens.

    :param given_name: Canonical given name.
    :return: Concatenated uppercase initials.
    """
    return "".join(token[0].upper() for token in given_name.split() if token)
