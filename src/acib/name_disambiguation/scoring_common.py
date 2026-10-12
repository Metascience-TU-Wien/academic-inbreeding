# Copyright © 2026 TU Wien

"""Shared normalization and low-level helper functions for scoring."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

from acib.common.names import given_name_compatible as _shared_given_name_compatible
from acib.common.names import (
    given_tokens,
)
from acib.common.names import (
    incompatible_full_given_names as _shared_incompatible_full_given_names,
)
from acib.common.names import surname_compatible as _shared_surname_compatible
from acib.common.names import (
    surname_tokens,
)

from .normalization import normalize_disambiguation_name
from .parsing import parse_embedded_json

_SURNAME_PARTICLES = {
    "da",
    "de",
    "del",
    "della",
    "den",
    "der",
    "di",
    "dos",
    "du",
    "la",
    "le",
    "van",
    "von",
}

_PLACEHOLDER_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "undefined",
    "n/a",
    "na",
}

SEVERE_CONFLICT_LABELS = {
    "orcid_conflict",
    "given_conflict",
    "given_gender_conflict",
}


def normalize_person_name(value: str) -> str:
    """Normalize a person name with the legacy transliteration order.

    :param value: Name text to normalize.
    :return: Lowercase text after legacy transliteration and punctuation removal.
    """
    return normalize_disambiguation_name(value)


def given_name_compatible(left: str, right: str) -> bool:
    """Compare names with the legacy compatibility function's normalization.

    :param left: First given-name value.
    :param right: Second given-name value.
    :return: Whether the values are compatible.
    """
    return _shared_given_name_compatible(left.strip(), right.strip())


def surname_compatible(left: str, right: str) -> bool:
    """Compare surnames with the legacy compatibility function's normalization.

    :param left: First surname value.
    :param right: Second surname value.
    :return: Whether the values are compatible.
    """
    return _shared_surname_compatible(left.strip(), right.strip())


def incompatible_full_given_names(left: str, right: str) -> bool:
    """Check full-name conflict after legacy edge-whitespace handling.

    :param left: First given-name value.
    :param right: Second given-name value.
    :return: Whether both full names are incompatible.
    """
    return _shared_incompatible_full_given_names(left.strip(), right.strip())


def compose_full_name(surname: str, given_name: str) -> str:
    """Compose a surname-first full name when both parts are present.

    :param surname: Family name.
    :param given_name: Given name.
    :return: ``surname, given_name`` or an empty string.
    """
    if not surname or not given_name:
        return ""
    return f"{surname}, {given_name}"


def core_surname_tokens_from_values(values: Iterable[str]) -> set[str]:
    """Collect normalized surname tokens excluding common particles.

    :param values: Surname variants.
    :return: Unique nonparticle surname tokens.
    """
    tokens: set[str] = set()

    for value in values:
        tokens.update(
            token for token in surname_tokens(value) if token and token not in _SURNAME_PARTICLES
        )

    return tokens


def full_given_tokens(values: Iterable[str]) -> set[str]:
    """Collect tokens representing fully spelled given names.

    :param values: Given-name variants.
    :return: Unique tokens longer than an initial.
    """
    tokens: set[str] = set()

    for value in values:
        tokens.update(token for token in given_tokens(value) if len(token) > 1)

    return tokens


def full_given_names_conflict(
    left_values: Iterable[str],
    right_values: Iterable[str],
) -> bool:
    """Check whether every full-name combination is incompatible.

    :param left_values: Given-name variants for the first profile.
    :param right_values: Given-name variants for the second profile.
    :return: Whether all full-name pairs conflict.
    """
    left_full = [value for value in left_values if full_given_tokens([value])]

    right_full = [value for value in right_values if full_given_tokens([value])]

    if not left_full or not right_full:
        return False

    return all(
        incompatible_full_given_names(left_value, right_value)
        for left_value in left_full
        for right_value in right_full
    )


def best_initials_pair(
    left_values: Iterable[str],
    right_values: Iterable[str],
) -> tuple[str, str] | None:
    """Select the first pair with the greatest shared-initial count.

    :param left_values: Initials variants for the first profile.
    :param right_values: Initials variants for the second profile.
    :return: Best pair, or ``None`` when either side is empty.
    """
    best_pair: tuple[str, str] | None = None
    best_score = -1

    for left_value in left_values:
        for right_value in right_values:
            score = count_shared_initials(
                left_value,
                right_value,
            )

            if score > best_score:
                best_pair = (
                    left_value,
                    right_value,
                )
                best_score = score

    return best_pair


def count_shared_initials(
    left: str,
    right: str,
) -> int:
    """Count distinct initials present on both sides, independent of order.

    :param left: First initials string.
    :param right: Second initials string.
    :return: Number of shared distinct characters.
    """
    return len(set(left) & set(right))


def initials_conflict(
    left_values: Iterable[str],
    right_values: Iterable[str],
) -> bool:
    """Check whether all nonempty initials combinations conflict.

    :param left_values: Initials variants for the first profile.
    :param right_values: Initials variants for the second profile.
    :return: Whether no pair shares an initial.
    """
    left_list = [value for value in left_values if value]

    right_list = [value for value in right_values if value]

    if not left_list or not right_list:
        return False

    for left_value in left_list:
        for right_value in right_list:
            if (
                count_shared_initials(
                    left_value,
                    right_value,
                )
                > 0
            ):
                return False

    return True


def normalize_text(value: Any) -> str:
    """Normalize free text and drop known placeholder values.

    :param value: Source text-like value.
    :return: Normalized text or empty string for placeholders.
    """
    text = normalize_person_name(str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()

    if text.lower() in _PLACEHOLDER_VALUES:
        return ""

    return text


def normalize_country(value: Any) -> str:
    """Normalize country codes or names for evidence comparisons.

    :param value: Country-like value.
    :return: Uppercase short code or normalized country name.
    """
    candidate = str(value or "").strip()

    if not candidate or candidate.lower() in _PLACEHOLDER_VALUES:
        return ""

    if len(candidate) <= 3 and candidate.isalpha():
        return candidate.upper()

    return normalize_text(candidate)


def normalize_identifier(value: Any) -> str:
    """Normalize identifier-like values while preserving numeric precision.

    :param value: Scalar identifier value.
    :return: Normalized identifier or empty string for missing values.
    """
    if value is None:
        return ""

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if pd.isna(value):
            return ""

        if value.is_integer():
            return str(int(value))

        candidate = format(value, ".15g")

    else:
        candidate = str(value).strip()

    if not candidate or candidate.lower() in _PLACEHOLDER_VALUES:
        return ""

    candidate = candidate.strip().strip("[]{}()\"'")

    candidate = re.sub(
        r"\s+",
        "",
        candidate,
    )

    if candidate.lower() in _PLACEHOLDER_VALUES:
        return ""

    return candidate


def parse_collection_field(value: Any) -> list[str]:
    """
    Parse a scalar, list, or JSON-ish field into string values.

    Recursion is restricted to types that this function explicitly
    knows how to terminate. This prevents infinite recursion for
    unsupported Python-literal types while preserving the existing
    behavior for strings, numbers, collections, and mappings.

    :param value: Scalar or nested collection representation.
    :return: Flattened string values.
    """
    if value is None:
        return []

    if isinstance(value, (int, bool)):
        return [str(value)]

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, float):
        return [str(value)]

    if isinstance(value, (list, tuple, set)):
        items: list[str] = []

        for entry in value:
            items.extend(parse_collection_field(entry))

        return items

    if isinstance(value, dict):
        dict_items: list[str] = []

        for entry in value.values():
            dict_items.extend(parse_collection_field(entry))

        return dict_items

    candidate = str(value).strip()

    if not candidate or candidate.lower() in _PLACEHOLDER_VALUES:
        return []

    parsed = parse_embedded_json(candidate)

    if (
        parsed is not None
        and parsed != candidate
        and isinstance(
            parsed,
            (
                str,
                int,
                float,
                bool,
                list,
                tuple,
                set,
                dict,
            ),
        )
    ):
        return parse_collection_field(parsed)

    if re.search(r"[;|\n]", candidate):
        candidate_items: list[str] = []

        for part in re.split(
            r"[;|\n]",
            candidate,
        ):
            candidate_items.extend(parse_collection_field(part))

        return candidate_items

    return [candidate.strip().strip("[]{}()\"'")]


def parse_identifier_collection(
    value: Any,
) -> set[str]:
    """
    Parse one or many identifier values into a normalized set.

    Recursion is restricted to types this function explicitly
    supports. This avoids cycles caused by unsupported literal
    values while preserving the existing normalization behavior.

    :param value: Scalar or collection of identifier values.
    :return: Unique normalized identifier strings.
    """
    if value is None:
        return set()

    if isinstance(value, (int, bool)):
        normalized = normalize_identifier(value)

        return {normalized} if normalized else set()

    if isinstance(value, float) and pd.isna(value):
        return set()

    if isinstance(value, float):
        normalized = normalize_identifier(value)

        return {normalized} if normalized else set()

    if isinstance(value, (list, tuple, set)):
        values: set[str] = set()

        for entry in value:
            values.update(parse_identifier_collection(entry))

        return values

    candidate = str(value).strip()

    if not candidate or candidate.lower() in _PLACEHOLDER_VALUES:
        return set()

    parsed = parse_embedded_json(candidate)

    if (
        parsed is not None
        and parsed != candidate
        and isinstance(
            parsed,
            (
                str,
                int,
                float,
                bool,
                list,
                tuple,
                set,
            ),
        )
    ):
        return parse_identifier_collection(parsed)

    if re.search(r"[;,|]", candidate):
        candidate_values: set[str] = set()

        for part in re.split(
            r"[;,|]",
            candidate,
        ):
            normalized = normalize_identifier(part)

            if normalized:
                candidate_values.add(normalized)

        return candidate_values

    normalized = normalize_identifier(candidate)

    return {normalized} if normalized else set()


def canonical_first_name(value: str) -> str:
    """Return the first fully spelled token in a normalized name.

    :param value: Given-name text.
    :return: First noninitial token, or empty string.
    """
    tokens = [token for token in normalize_text(value).split() if len(token) > 1]

    return tokens[0] if tokens else ""


def extract_publication_field_values(
    row_dict: dict[str, Any],
    *,
    exact_keys: tuple[str, ...] = (),
    contains: tuple[str, ...] = (),
    normalizer: Any,
) -> set[str]:
    """Extract normalized values from matching keys in a publication row.

    :param row_dict: Source publication row as a mapping.
    :param exact_keys: Exact source-key names to include.
    :param contains: Key fragments that select additional source fields.
    :param normalizer: Function applied to each flattened source value.
    :return: Nonempty normalized values.
    """
    values: set[str] = set()

    for key, value in row_dict.items():
        lowered_key = key.lower()

        if lowered_key in exact_keys or any(token in lowered_key for token in contains):
            for item in parse_collection_field(value):
                normalized = normalizer(item)

                if normalized:
                    values.add(normalized)

    return values


def unique_values(
    values: Iterable[Any],
) -> tuple[Any, ...]:
    """Return truthy values once each, preserving their first-seen order.

    :param values: Values to deduplicate.
    :return: Unique ordered tuple.
    """
    ordered: list[Any] = []
    seen: set[Any] = set()

    for value in values:
        if not value or value in seen:
            continue

        seen.add(value)
        ordered.append(value)

    return tuple(ordered)


def preview_values(
    values: Iterable[str],
    limit: int = 3,
) -> str:
    """Render a bounded comma-separated preview of values.

    :param values: Values to display.
    :param limit: Maximum number of values before ellipsis.
    :return: Human-readable preview.
    """
    items = list(values)

    if len(items) <= limit:
        return ", ".join(items)

    return ", ".join(items[:limit]) + ", ..."


def first_match(
    left_values: Iterable[str],
    right_values: Iterable[str],
    *,
    predicate: Any,
) -> tuple[str, str] | None:
    """Return the first cross-product pair accepted by a predicate.

    :param left_values: Values considered on the left side.
    :param right_values: Values considered on the right side.
    :param predicate: Pair compatibility function.
    :return: First compatible pair, or ``None``.
    """
    for left_value in left_values:
        for right_value in right_values:
            if predicate(
                left_value,
                right_value,
            ):
                return (
                    left_value,
                    right_value,
                )

    return None


def score_from_bins(
    count: int,
    mapping: dict[int, float],
    default_overflow: float,
) -> float:
    """Map a positive evidence count to a configured score bucket.

    :param count: Number of observed evidence items.
    :param mapping: Exact count-to-score bins.
    :param default_overflow: Score for counts absent from ``mapping``.
    :return: Score for the observed count.
    """
    if count <= 0:
        return 0.0

    if count in mapping:
        return mapping[count]

    return default_overflow


def sigmoid(value: float) -> float:
    """Calculate the logistic transform used for pair confidence.

    :param value: Raw score input.
    :return: Logistic probability.
    """
    return 1.0 / (1.0 + math.exp(-value))
