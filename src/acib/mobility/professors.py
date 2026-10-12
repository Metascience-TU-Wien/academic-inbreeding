# Copyright © 2026 TU Wien

"""Prepare professor-specific author profiles for mobility analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd

from acib.common.titles import (
    PROFESSOR_TITLE_CLASSES as PROFESSOR_TITLE_VALUES,
    PROFESSOR_TITLE_HIERARCHY,
    TITLE_CLASS_COLUMN,
    TITLE_OUTPUT_COLUMN,
    classify_title_for_institution,
    highest_professor_title_class,
    professor_title_mask,
)
from acib.constants import (
    AUTHOR_ID,
    PERSON_TITLE,
    RESEARCHER_ID,
    TITLE_CLASS,
    UNIVERSITY_NAME,
)
from acib.mobility.constants import (
    COAFFILIATED_PROFESSOR_UNIVERSITIES,
    IS_COAFFILIATED_PROFESSOR,
    IS_MERGED_PROFESSOR_PROFILE,
    MERGED_PROFESSOR_ROWS,
    N_COAFFILIATED_PROFESSOR_UNIVERSITIES,
    PROFESSOR_NAMES,
    PROFESSOR_SC_IDS,
    PROFESSOR_TITLE_CLASSES as PROFESSOR_TITLE_CLASSES_COLUMN,
    PROFESSOR_TITLES,
)
from acib.mobility.input import parse_mapping, parse_sequence
from acib.name_disambiguation.constants import (
    CURRENT_AFFILIATION_IDS,
    SELECTED_AUTHOR_IDS,
    SELECTED_DISAMBIGUATED_CLUSTER_ID,
    SELECTED_HISTORY,
)
from acib.name_matching.constants import RESEARCHER_NAME

_PROFESSOR_PROFILE_MERGE_KEY = "_professor_profile_merge_key"


def prepare_professor_author_profiles(
    authors: pd.DataFrame,
    *,
    professor_title_pattern: str = "prof",
) -> pd.DataFrame:
    """Classify, filter, and merge same-person professor rows.

    :param authors: Selected researcher-author records.
    :param professor_title_pattern: Case-insensitive raw-title filter pattern.
    :return: Professor profile records with merged same-person metadata.
    """
    titled_authors = add_or_update_title_columns(authors)
    professor_authors = titled_authors.loc[
        professor_title_mask(titled_authors, professor_title_pattern=professor_title_pattern)
    ].copy()
    return merge_same_person_professor_profiles(professor_authors)


def add_or_update_title_columns(authors: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with institution-adjusted title classes and output titles.

    :param authors: Researcher-author records.
    :return: Copy with recomputed title class and title columns.
    """
    result = authors.copy()
    if PERSON_TITLE not in result.columns:
        result[PERSON_TITLE] = ""
    result[TITLE_CLASS] = result.apply(
        lambda row: classify_title_for_institution(row.get(PERSON_TITLE), row.get(UNIVERSITY_NAME)),
        axis=1,
    )
    result[TITLE_OUTPUT_COLUMN] = result[TITLE_CLASS]
    return result


def merge_same_person_professor_profiles(authors: pd.DataFrame) -> pd.DataFrame:
    """Collapse professor rows sharing the selected author profile identity.

    :param authors: Professor-filtered author rows.
    :return: Merged profile rows in source group order.
    """
    if authors.empty:
        return authors.copy()
    merge_keys = professor_profile_merge_keys(authors)
    if merge_keys is None:
        return authors.copy()
    result = authors.copy()
    result[_PROFESSOR_PROFILE_MERGE_KEY] = merge_keys
    result = result[result[_PROFESSOR_PROFILE_MERGE_KEY].notna()].copy()
    if result.empty:
        return result.drop(columns=[_PROFESSOR_PROFILE_MERGE_KEY])
    rows = [
        merge_professor_profile_group(group.drop(columns=[_PROFESSOR_PROFILE_MERGE_KEY]))
        for _, group in result.groupby(_PROFESSOR_PROFILE_MERGE_KEY, sort=False)
    ]
    return pd.DataFrame(rows).reset_index(drop=True)


def professor_profile_merge_keys(authors: pd.DataFrame) -> pd.Series | None:
    """Build merge keys from representative ID and full selected-author ID sets.

    :param authors: Candidate professor rows.
    :return: Nullable merge-key series, or ``None`` without an identity field.
    """
    if SELECTED_DISAMBIGUATED_CLUSTER_ID in authors.columns:
        cluster_ids = authors[SELECTED_DISAMBIGUATED_CLUSTER_ID].map(normalize_merge_key)
        if SELECTED_AUTHOR_IDS in authors.columns:
            author_id_sets = authors[SELECTED_AUTHOR_IDS].map(normalized_author_id_set_key)
            return cluster_ids.combine(
                author_id_sets,
                lambda cluster_id, author_ids: (
                    f"{cluster_id}|{author_ids}" if cluster_id is not None else None
                ),
            )
        return cluster_ids
    merge_column = professor_profile_merge_column(authors)
    if merge_column is None:
        return None
    return authors[merge_column].map(normalize_merge_key)


def professor_profile_merge_column(authors: pd.DataFrame) -> str | None:
    """Return the available raw author identifier used as a fallback key.

    :param authors: Candidate professor rows.
    :return: Author-ID field, or ``None`` when absent.
    """
    return AUTHOR_ID if AUTHOR_ID in authors.columns else None


def merge_professor_profile_group(group: pd.DataFrame) -> dict[str, Any]:
    """Merge one identity group and aggregate professor-specific metadata.

    :param group: Rows belonging to one professor profile.
    :return: A merged record preserving first-row base metadata.
    """
    base = group.iloc[0].to_dict()
    universities = unique_text_values(group.get(UNIVERSITY_NAME))
    title_classes = ordered_title_classes(group[TITLE_CLASS_COLUMN])
    highest_title = highest_professor_title_class(title_classes)
    base.update(
        {
            PROFESSOR_SC_IDS: unique_values(group.get(RESEARCHER_ID)),
            PROFESSOR_NAMES: unique_text_values(group.get(RESEARCHER_NAME)),
            PROFESSOR_TITLES: unique_text_values(group.get(PERSON_TITLE)),
            PROFESSOR_TITLE_CLASSES_COLUMN: title_classes,
            TITLE_CLASS: highest_title,
            TITLE_OUTPUT_COLUMN: highest_title,
            PERSON_TITLE: raw_title_for_class(group=group, title_class=highest_title),
            COAFFILIATED_PROFESSOR_UNIVERSITIES: universities,
            IS_COAFFILIATED_PROFESSOR: len(universities) > 1,
            N_COAFFILIATED_PROFESSOR_UNIVERSITIES: len(universities),
            IS_MERGED_PROFESSOR_PROFILE: len(group) > 1,
            MERGED_PROFESSOR_ROWS: len(group),
        }
    )
    if SELECTED_AUTHOR_IDS in group.columns:
        base[SELECTED_AUTHOR_IDS] = merged_sequence(group[SELECTED_AUTHOR_IDS])
    if SELECTED_HISTORY in group.columns:
        base[SELECTED_HISTORY] = merged_history(group[SELECTED_HISTORY])
    if CURRENT_AFFILIATION_IDS in group.columns:
        base[CURRENT_AFFILIATION_IDS] = merged_sequence(group[CURRENT_AFFILIATION_IDS])
    return base


def normalize_merge_key(value: Any) -> str | None:
    """Normalize a possibly missing profile identifier to a stable string.

    :param value: Identifier value.
    :return: Normalized identifier or ``None`` when missing.
    """
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def normalized_author_id_set_key(value: Any) -> str:
    """Serialize selected author IDs in sorted order for profile matching.

    :param value: Serialized or parsed selected author identifiers.
    :return: Comma-separated sorted identifiers.
    """
    author_ids = sorted(parse_sequence(value))
    return ",".join(str(author_id) for author_id in author_ids)


def unique_values(values: pd.Series | None) -> list[Any]:
    """Return nonmissing values deduplicated by stripped string in source order.

    :param values: Values to deduplicate, if present.
    :return: First occurrence of each distinct textual value.
    """
    if values is None:
        return []
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if value is None or pd.isna(value):
            continue
        key = str(value).strip()
        if not key or key.casefold() in {"nan", "none", "null"} or key in seen:
            continue
        result.append(value)
        seen.add(key)
    return result


def unique_text_values(values: pd.Series | None) -> list[str]:
    """Return deduplicated nonmissing values as stripped strings.

    :param values: Values to process, if present.
    :return: Text values in first-seen order.
    """
    return [str(value).strip() for value in unique_values(values)]


def ordered_title_classes(values: pd.Series) -> list[str]:
    """Return present title classes in hierarchy order followed by sorted others.

    :param values: Classified titles for a profile.
    :return: Unique ordered title classes.
    """
    present = {str(value) for value in values.dropna() if str(value) in PROFESSOR_TITLE_VALUES}
    ordered = [title for title in PROFESSOR_TITLE_HIERARCHY if title in present]
    other = sorted({str(value) for value in values.dropna()} - set(ordered))
    return ordered + other


def raw_title_for_class(*, group: pd.DataFrame, title_class: str) -> str:
    """Select the first raw title matching the winning class, or first available title.

    :param group: Rows for one merged professor profile.
    :param title_class: Winning title class.
    :return: Selected raw title, or an empty string if unavailable.
    """
    if title_class != "No Title":
        matches = group.loc[group[TITLE_CLASS_COLUMN].eq(title_class), PERSON_TITLE]
        titles = unique_text_values(matches)
        if titles:
            return titles[0]
    titles = unique_text_values(group.get(PERSON_TITLE))
    return titles[0] if titles else ""


def merged_sequence(values: pd.Series) -> list[int]:
    """Merge parsed integer sequences with first-seen ordering and deduplication.

    :param values: Serialized or parsed sequences.
    :return: Unique identifiers in row and sequence order.
    """
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        for item in parse_sequence(value):
            if item not in seen:
                result.append(item)
                seen.add(item)
    return result


def merged_history(values: pd.Series) -> dict[str, list[int]]:
    """Merge history mappings, preserving first-seen key and identifier order.

    :param values: Serialized or parsed history mappings.
    :return: Combined mapping of current IDs to unique historical IDs.
    """
    result: dict[str, list[int]] = {}
    seen: dict[str, set[int]] = {}
    for value in values:
        for author_id, historical_ids in parse_mapping(value).items():
            result.setdefault(author_id, [])
            seen.setdefault(author_id, set())
            for historical_id in historical_ids:
                if historical_id not in seen[author_id]:
                    result[author_id].append(historical_id)
                    seen[author_id].add(historical_id)
    return result
