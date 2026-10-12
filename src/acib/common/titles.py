# Copyright © 2026 TU Wien

"""Normalize and classify academic titles for shared pipeline use.

The classification order and institution-specific adjustments preserve the
legacy title utility's semantics for name disambiguation and future stages.
"""

from __future__ import annotations

import ast
import re

import pandas as pd

from acib.constants import PERSON_TITLE, TITLE, TITLE_CLASS

TITLE_COLUMN = PERSON_TITLE
TITLE_CLASS_COLUMN = TITLE_CLASS
TITLE_OUTPUT_COLUMN = TITLE
PROFESSOR_TITLE_HIERARCHY = (
    "Full Prof.",
    "Ao. Prof.",
    "Ass. Prof.",
    "FH-Prof.",
    "FH-Ass. Prof.",
)
PROFESSOR_TITLE_CLASSES = frozenset(PROFESSOR_TITLE_HIERARCHY)
MCI_UNIVERSITY_NAMES = frozenset({"mci hochschule"})


def add_title_class(
    data: pd.DataFrame,
    *,
    title_column: str = TITLE_COLUMN,
    title_class_column: str = TITLE_CLASS_COLUMN,
    output_title_column: str | None = None,
) -> pd.DataFrame:
    """Return a copy with a class column derived from free-text titles.

    :param data: Input records.
    :param title_column: Column containing raw titles.
    :param title_class_column: Column to populate with classified titles.
    :param output_title_column: Optional second output column.
    :return: A copy of the input with requested title columns.
    :raises KeyError: If the required title column is missing.
    """
    if title_column not in data.columns:
        raise KeyError(f"Input data is missing required title column: {title_column}")
    result = data.copy()
    result[title_class_column] = result[title_column].map(classify_title)
    if output_title_column is not None:
        result[output_title_column] = result[title_class_column]
    return result


def highest_professor_title_class(values: object) -> str:
    """Return the highest-ranked professor class found in values.

    :param values: A scalar or collection of classified title values.
    :return: The highest class in the configured hierarchy, or ``No Title``.
    """
    if values is None:
        return "No Title"
    iterable = values if isinstance(values, (list, tuple, set, pd.Series)) else [values]
    present = {str(value) for value in iterable if str(value) in PROFESSOR_TITLE_CLASSES}
    for title_class in PROFESSOR_TITLE_HIERARCHY:
        if title_class in present:
            return title_class
    return "No Title"


def professor_title_mask(
    data: pd.DataFrame,
    *,
    title_column: str = TITLE_COLUMN,
    title_class_column: str = TITLE_CLASS_COLUMN,
    professor_title_pattern: str = "prof",
) -> pd.Series:
    """Return rows marked as professor by raw or classified title.

    :param data: Input records.
    :param title_column: Column containing raw titles.
    :param title_class_column: Optional column containing title classes.
    :param professor_title_pattern: Case-insensitive raw-title pattern; an
        empty value disables raw-title matching.
    :return: A Boolean mask indexed like ``data``.
    :raises KeyError: If the raw title column is missing.
    """
    if title_column not in data.columns:
        raise KeyError(f"Input data is missing required title column: {title_column}")
    classified_mask = (
        data[title_class_column].isin(PROFESSOR_TITLE_CLASSES)
        if title_class_column in data.columns
        else pd.Series(False, index=data.index)
    )
    if not professor_title_pattern:
        return classified_mask
    raw_mask = (
        data[title_column]
        .astype("string")
        .str.contains(professor_title_pattern, case=False, na=False)
    )
    return raw_mask | classified_mask


def normalize_title_text(value: object) -> str:
    """Normalize title casing, whitespace, missing values, and dash variants.

    :param value: A title-like value.
    :return: Normalized lowercase text, or an empty string for missing values.
    """
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_title_text(value: object) -> str:
    """Return the normalized title with non-alphanumeric characters removed.

    :param value: A title-like value.
    :return: A compact string for pattern matching.
    """
    return re.sub(r"[^a-z0-9]+", "", normalize_title_text(value))


def classify_title(value: object) -> str:
    """Classify a free-text title, respecting legacy rule precedence.

    :param value: A title-like value.
    :return: The detected title class.
    """
    normalized = normalize_title_text(value)
    compact = compact_title_text(value)
    if not normalized:
        return "No Title"
    if is_fh_assistant_professor(normalized=normalized, compact=compact):
        return "FH-Ass. Prof."
    if is_associate_professor(normalized=normalized, compact=compact):
        return "Ao. Prof."
    if is_assistant_professor(normalized=normalized, compact=compact):
        return "Ass. Prof."
    if is_explicit_full_professor(normalized=normalized, compact=compact):
        return "Full Prof."
    if is_fh_professor(normalized=normalized, compact=compact):
        return "FH-Prof."
    if is_full_professor(normalized=normalized, compact=compact):
        return "Full Prof."
    if is_private_lecturer(normalized=normalized, compact=compact):
        return "Priv.-Doz."
    if is_doctor_title(normalized=normalized, compact=compact):
        return "Dr."
    if "phd" in compact:
        return "PhD"
    return classify_lower_title(compact=compact, normalized=normalized)


def classify_title_for_institution(value: object, universities: object = None) -> str:
    """Classify a title and apply institution-specific title semantics.

    :param value: A title-like value.
    :param universities: Institution value or sequence of institution values.
    :return: Institution-adjusted title class.
    """
    title_class = classify_title(value)
    title_class = adjust_title_for_institution(title_class, universities=universities)
    if title_class == "Ass. Prof." and is_uas_only_assignment(universities):
        return "FH-Ass. Prof."
    if (
        title_class == "Full Prof."
        and is_uas_only_assignment(universities)
        and not is_explicit_full_professor(
            normalized=normalize_title_text(value), compact=compact_title_text(value)
        )
    ):
        return "FH-Prof."
    return title_class


def adjust_title_for_institution(title_class: str, universities: object = None) -> str:
    """Downgrade MCI-only full professor labels to FH professor labels.

    :param title_class: Existing title classification.
    :param universities: Institution value or sequence of institution values.
    :return: Adjusted or unchanged title class.
    """
    if title_class == "Full Prof." and is_mci_only_university_assignment(universities):
        return "FH-Prof."
    return title_class


def is_mci_only_university_assignment(universities: object) -> bool:
    """Return whether all normalized institution values identify MCI.

    :param universities: Institution value or sequence of values.
    :return: Whether the nonempty normalized set is a subset of MCI.
    """
    values = normalized_university_values(universities)
    return bool(values) and values <= MCI_UNIVERSITY_NAMES


def is_uas_only_assignment(universities: object) -> bool:
    """Return whether all normalized institutions are universities of applied sciences.

    :param universities: Institution value or sequence of values.
    :return: Whether the nonempty normalized set consists only of UAS names.
    """
    values = normalized_university_values(universities)
    return bool(values) and all(is_uas_name(value) for value in values)


def is_uas_name(value: str) -> bool:
    """Return whether an institution name matches a recognized UAS marker.

    :param value: Normalized institution name.
    :return: Whether the name contains a recognized marker.
    """
    name = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    return (
        "fachhochschule" in name
        or "university of applied sciences" in name
        or "mci hochschule" in name
        or "hochschule burgenland" in name
        or "campus02" in name
        or "lauder business" in name
        or "fernfh" in name
        or bool(re.search(r"\bfh\b|\buas\b", name))
    )


def normalized_university_values(value: object) -> set[str]:
    """Normalize institution input into a set of case-folded names.

    :param value: Scalar, collection, or string representation of a sequence.
    :return: Set of normalized non-missing institution names.
    """
    if value is None:
        return set()
    if isinstance(value, float) and pd.isna(value):
        return set()
    if isinstance(value, (list, tuple, set, pd.Series)):
        raw_values = list(value)
    else:
        text = str(value).strip()
        if not text or text.casefold() in {"nan", "none", "null", "[]"}:
            return set()
        raw_values = parse_university_sequence(text)
    result: set[str] = set()
    for item in raw_values:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text.casefold() in {"nan", "none", "null"}:
            continue
        result.add(re.sub(r"\s+", " ", text).casefold())
    return result


def parse_university_sequence(text: str) -> list[object]:
    """Parse a literal sequence or retain the input as a single value.

    :param text: Text potentially containing a Python sequence literal.
    :return: Parsed items, or a one-item list containing ``text``.
    """
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)
    return [text]


def is_fh_professor(*, normalized: str, compact: str) -> bool:
    """Return whether a title contains an FH professor marker.

    :param normalized: Normalized title text.
    :param compact: Compacted title text.
    :return: Whether the FH-professor rule matches.
    """
    if is_fh_assistant_professor(normalized=normalized, compact=compact):
        return False
    return (
        bool(re.search(r"\bfh\s*-?\s*prof\.?", normalized))
        or bool(re.search(r"\bhon\.?\s*-?\s*prof\.?\s*\(?\s*fh\s*\)?", normalized))
        or bool(re.search(r"\bprof(?:\.in|in|essorin|essor)?\.?\s*\(?\s*fh\s*\)?", normalized))
        or "fhhonprof" in compact
        or "fhprof" in compact
    )


def is_fh_assistant_professor(*, normalized: str, compact: str) -> bool:
    """Return whether a title contains an FH assistant professor marker.

    :param normalized: Normalized title text.
    :param compact: Compacted title text.
    :return: Whether the FH-assistant-professor rule matches.
    """
    if re.search(
        r"\b(?:assoz\.?|assoc\.?|associate|assistant|ass\.?)\s*-?\s*"
        r"prof(?:essor)?\.?\s*\(\s*fh\s*\)",
        normalized,
    ):
        return True
    return bool(re.search(r"\bfh\s*-?\s*assistenzprof\.?", normalized)) or any(
        marker in compact
        for marker in ("fhassprof", "fhassistenzprof", "fhassistantprof", "fhassistprof")
    )


def is_explicit_full_professor(*, normalized: str, compact: str) -> bool:
    """Return whether a title contains an explicit full professor marker.

    :param normalized: Normalized title text.
    :param compact: Compacted title text.
    :return: Whether the explicit full-professor rule matches.
    """
    return bool(
        re.search(
            r"\b(univ\.?\s*-?\s*prof\.?|universitätsprofessor|universitaetsprofessor)\b",
            normalized,
        )
    ) or any(marker in compact for marker in ("univprof", "universityprof"))


def is_full_professor(*, normalized: str, compact: str) -> bool:
    """Return whether a title contains a full professor marker.

    :param normalized: Normalized title text.
    :param compact: Compacted title text.
    :return: Whether the full-professor rule matches.
    """
    if is_explicit_full_professor(normalized=normalized, compact=compact):
        return True
    if "prof" not in compact:
        return False
    return not is_associate_professor(
        normalized=normalized, compact=compact
    ) and not is_assistant_professor(normalized=normalized, compact=compact)


def is_associate_professor(*, normalized: str, compact: str) -> bool:
    """Return whether a title contains an associate/Ao professor marker.

    :param normalized: Normalized title text.
    :param compact: Compacted title text.
    :return: Whether the associate-professor rule matches.
    """
    if re.search(
        r"\b(ao\.?\s*univ\.?\s*-?\s*prof\.?|ao\.?\s*prof\.?|"
        r"a\.?\s*o\.?\s*prof\.?|außerord\.?\s*prof\.?|"
        r"ausserord\.?\s*prof\.?)",
        normalized,
    ):
        return True
    return any(
        marker in compact
        for marker in (
            "aounivprof",
            "aoprof",
            "assocprof",
            "associateprof",
            "assozprof",
            "assozunivprof",
            "associateunivprof",
        )
    )


def is_assistant_professor(*, normalized: str, compact: str) -> bool:
    """Return whether a title contains an assistant professor marker.

    :param normalized: Normalized title text.
    :param compact: Compacted title text.
    :return: Whether the assistant-professor rule matches.
    """
    if re.search(
        r"\b(ass\.?\s*-?\s*prof\.?|assistenzprofessor|assistant\s+professor|assistant\s+prof\.?)",
        normalized,
    ):
        return True
    return any(
        marker in compact
        for marker in (
            "assprof",
            "fhassistenzprof",
            "assistenzprof",
            "assistenzprofessor",
            "assistantprof",
            "assistprof",
        )
    )


def is_private_lecturer(*, normalized: str, compact: str) -> bool:
    """Return whether a title contains a private lecturer marker.

    :param normalized: Normalized title text.
    :param compact: Compacted title text.
    :return: Whether the private-lecturer rule matches.
    """
    if re.search(r"\b(priv\.?\s*-?\s*doz\.?|privatdozent|privatdozentin|pd\.?)\b", normalized):
        return True
    return any(
        marker in compact
        for marker in ("privdoz", "privatdoz", "privatdozent", "privatdozentin", "univdoz")
    )


def is_doctor_title(*, normalized: str, compact: str) -> bool:
    """Return whether a title contains a doctor-title marker.

    :param normalized: Normalized title text.
    :param compact: Compacted title text.
    :return: Whether the doctor-title rule matches.
    """
    return (
        bool(re.search(r"\bdr\b|dr\.", normalized))
        or compact.startswith("dr")
        or any(
            marker in compact
            for marker in (
                "drmed",
                "drrer",
                "drtechn",
                "drphil",
                "drmont",
                "druniv",
                "dring",
                "ddr",
            )
        )
    )


def classify_lower_title(*, compact: str, normalized: str) -> str:
    """Classify common non-professor titles in legacy precedence order.

    :param compact: Compacted title text.
    :param normalized: Normalized title text.
    :return: Detected lower-degree or function title class.
    """
    function_markers = [
        ("projektass", "Projektass."),
        ("univass", "Univ.Ass."),
        ("univlektor", "Univ.Lektor"),
    ]
    for marker, label in function_markers:
        if marker in compact or marker in normalized:
            return label
    if has_any_marker(compact, normalized, ("msc", "m.sc", "masterofscience")):
        return "MSc"
    if has_any_marker(
        compact,
        normalized,
        (
            "ma",
            "m.a",
            "masterofarts",
            "mba",
            "mph",
            "mres",
            "mfa",
            "mmed",
            "mmus",
            "mtech",
            "meng",
            "mpharm",
            "med",
            "m.ed",
            "mgr",
            "mestre",
            "mestr",
        ),
    ):
        return "MA."
    if has_any_marker(compact, normalized, ("mag", "magister")):
        return "Mag."
    if has_any_marker(compact, normalized, ("dipl-ing", "dipling", "di")):
        return "DI"
    if has_any_marker(compact, normalized, ("bsc", "b.sc", "bachelorofscience")):
        return "BSc"
    if has_any_marker(
        compact, normalized, ("ba", "b.a", "bachelorofarts", "bed", "b.ed", "bakk", "bakkalaureat")
    ):
        return "BA."
    if has_any_marker(compact, normalized, ("ing",)):
        return "Ing."
    return "Other"


def has_any_marker(compact: str, normalized: str, markers: tuple[str, ...]) -> bool:
    """Return whether compact or tokenized title text contains a marker.

    :param compact: Compacted title text.
    :param normalized: Normalized title text.
    :param markers: Marker strings to check.
    :return: Whether any marker is present.
    """
    tokens = set(re.findall(r"[a-z]+", normalized.lower()))
    for marker in markers:
        compact_marker = marker.replace(".", "").replace("-", "")
        if len(compact_marker) <= 3:
            if ("." in marker and marker in normalized) or compact_marker in tokens:
                return True
            continue
        if marker in normalized or compact_marker in compact:
            return True
    return False
