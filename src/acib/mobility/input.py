# Copyright © 2026 TU Wien

"""Prepare article-author and affiliation inputs for mobility analysis."""

from __future__ import annotations

import ast
import hashlib
import math
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from acib.constants import (
    AFFILIATION_ID,
    AFFILIATION_ID_PARENT,
    AFFILIATION_ID_PARENT_CHATGPT,
    AFFILIATION_ID_PARENT_SPLINK_CHATGPT,
    AFFILIATION_PARENT_SOURCE,
    AFFILIATION_UMBRELLA_ID,
    AUTHOR_ID,
    EID,
    HISTORICAL_IDENTIFIER,
    PERSON_TITLE,
    RESEARCHER_ID,
    SUBJECT_AREAS,
    UNIVERSITY_NAME,
)
from acib.mobility.config import MobilityConfig
from acib.mobility.constants import (
    AFFILIATION_PREFERRED_NAME,
    ARTICLE_AFFILIATION_ID,
    ARTICLE_AFFILIATION_INDEX,
    AUTHOR_PROFILE_AMBIGUITY_REASON,
    AUTHOR_PROFILE_AMBIGUOUS,
    AUTHOR_PROFILE_ID_SET_KEY,
    MOBILITY_AUTHOR_PROFILE_ID,
    PUBLICATION_AUTHOR_COUNT,
    PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS,
    UNMAPPED_AFFILIATION_ID,
)
from acib.mobility.schemas import MobilityAffiliationSchema, MobilityArticleAuthorSchema
from acib.name_disambiguation.asjc import asjc_entry, level_two_asjc_code
from acib.name_disambiguation.constants import (
    AUTHOR_IDS as ARTICLE_AUTHOR_ID,
    SELECTED_AUTHOR_IDS,
    SELECTED_DISAMBIGUATED_CLUSTER_ID,
    SELECTED_HISTORY,
)
from acib.name_disambiguation.parsing import parse_subject_areas
from acib.name_matching.constants import NAME
from acib.schemas.matching import MatchingOutputSchema
from acib.schemas.researchers import ResearcherDisambiguationSelectionSchema

_RESEARCHER_METADATA_STRING_DTYPES = {
    "person_functions": "string",
    "institute_link": "string",
    "person_location_link": "string",
    "orcid": "string",
}
_PARENT_AFFILIATION_PRECEDENCE = (
    AFFILIATION_ID_PARENT_CHATGPT,
    AFFILIATION_ID_PARENT,
    AFFILIATION_PARENT_SOURCE,
    AFFILIATION_ID_PARENT_SPLINK_CHATGPT,
    AFFILIATION_UMBRELLA_ID,
)


@dataclass(frozen=True, slots=True)
class MobilityInputs:
    """Validated in-memory inputs for mobility analysis.

    :param authors: Final researcher-author selections.
    :param researcher_metadata: Matching output with researcher metadata.
    :param articles: Article-author evidence rows.
    :param affiliations: Geospatial affiliation crosswalk.
    """

    authors: pd.DataFrame
    researcher_metadata: pd.DataFrame
    articles: pd.DataFrame
    affiliations: pd.DataFrame


def load_mobility_inputs(config: MobilityConfig) -> MobilityInputs:
    """Load configured mobility inputs and validate their stage contracts.

    :param config: Resolved canonical mobility paths.
    :return: Validated in-memory mobility inputs.
    :raises FileNotFoundError: If any configured input is absent.
    :raises ImportError: If GeoPandas is unavailable for the GeoPackage input.
    """
    config.check_input_paths()
    authors = ResearcherDisambiguationSelectionSchema.validate(
        _read_mobility_dataframe(config.authors), lazy=True
    )
    researcher_metadata = MatchingOutputSchema.validate(
        _read_mobility_dataframe(
            config.researcher_metadata,
            dtype=_RESEARCHER_METADATA_STRING_DTYPES,
        ),
        lazy=True,
    )
    articles = MobilityArticleAuthorSchema.validate(
        _read_mobility_dataframe(config.article_author), lazy=True
    )
    affiliations = _read_mobility_geodataframe(config.affiliations)
    affiliation_attributes = pd.DataFrame(
        affiliations.drop(columns=[affiliations.geometry.name], errors="ignore")
    )
    MobilityAffiliationSchema.validate(affiliation_attributes, lazy=True)
    return MobilityInputs(
        authors=authors,
        researcher_metadata=researcher_metadata,
        articles=articles,
        affiliations=affiliations,
    )


def _read_mobility_dataframe(
    path: Path,
    *,
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read a configured mobility CSV or Parquet table.

    :param path: Input file path.
    :param dtype: Optional CSV column dtypes.
    :return: Loaded table.
    """
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False, dtype=dtype)


def _read_mobility_geodataframe(path: Path) -> pd.DataFrame:
    """Read the configured affiliation GeoPackage.

    :param path: GeoPackage path.
    :return: Geospatial affiliation records.
    :raises ImportError: If GeoPandas is unavailable.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("geopandas is required to read mobility affiliation files") from exc
    return gpd.read_file(path)


_HISTORICAL_AUTHOR_ID = "historical_author_id"
_CURRENT_AUTHOR_ID = "current_author_id"
_SELECTED_AUTHOR_ID = "selected_author_id"
_DISAMBIGUATED_AUTHOR_ID = "disambiguated_author_id"
_HISTORY = "history"
_PARALLEL_INDEX = "_parallel_gdf_index"
_ARTICLE_ROW_ID = "article_id"
_AUTHOR_SEPARATOR = r"[;,]"
_MULTIPLE_AFFILIATION_SEPARATOR = "-"
_RESEARCHER_JOIN_KEY = "_mobility_researcher_join_key"


def process_input_data(
    *,
    author_affiliations: pd.DataFrame,
    articles: pd.DataFrame,
    resolved_authors: pd.DataFrame,
    explode_by_affiliations: bool = True,
    max_workers: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare publication-author rows and cleaned affiliations.

    Publication counts and ASJC summaries are computed before filtering to local
    resolved authors, preserving full-publication evidence.

    :param author_affiliations: Affiliation metadata table.
    :param articles: Article-author evidence rows.
    :param resolved_authors: Final disambiguation selections.
    :param explode_by_affiliations: Whether to return one row per affiliation.
    :param max_workers: Maximum worker count for large affiliation batches.
    :return: Prepared article-author rows and normalized affiliations.
    """
    resolved_authors = add_mobility_author_profile_ids(resolved_authors)
    affiliations = clean_affiliations(author_affiliations)
    article_authors = articles.loc[articles[ARTICLE_AFFILIATION_ID].notna()].copy()
    article_authors = explode_articles_to_author_affiliations(
        article_authors, affiliations, max_workers=max_workers
    )
    article_authors = apply_historical_author_ids(article_authors, resolved_authors)
    article_authors = apply_disambiguated_author_ids(article_authors, resolved_authors)
    publication_author_counts = count_publication_authors(article_authors)
    publication_subject_areas = extract_publication_general_asjc_subject_areas(article_authors)
    article_authors = keep_resolved_authors(article_authors, resolved_authors)
    article_authors = attach_publication_author_counts(article_authors, publication_author_counts)
    article_authors = attach_publication_general_asjc_subject_areas(
        article_authors, publication_subject_areas
    )
    if explode_by_affiliations:
        article_authors = explode_affiliations(article_authors)
    return article_authors, affiliations


def enrich_authors_with_researcher_metadata(
    *,
    authors: pd.DataFrame,
    researcher_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich selected profiles with title and university fields by researcher ID.

    :param authors: Mobility's selected author-profile rows.
    :param researcher_metadata: Matching output with researcher metadata.
    :return: Author rows enriched without changing disambiguation identifiers.
    """
    if RESEARCHER_ID not in authors.columns or RESEARCHER_ID not in researcher_metadata.columns:
        return authors
    metadata_columns = [
        column
        for column in (PERSON_TITLE, UNIVERSITY_NAME)
        if column in researcher_metadata.columns
    ]
    if not metadata_columns:
        return authors

    metadata = researcher_metadata[[RESEARCHER_ID, *metadata_columns]].copy()
    metadata[_RESEARCHER_JOIN_KEY] = metadata[RESEARCHER_ID].map(normalize_identifier)
    metadata = metadata.drop_duplicates(subset=[_RESEARCHER_JOIN_KEY])
    author_rows = authors.copy()
    author_rows[_RESEARCHER_JOIN_KEY] = author_rows[RESEARCHER_ID].map(normalize_identifier)
    result = author_rows.merge(
        metadata,
        on=_RESEARCHER_JOIN_KEY,
        how="left",
        suffixes=("", "_metadata"),
    ).drop(columns=[_RESEARCHER_JOIN_KEY])
    for column in metadata_columns:
        metadata_column = f"{column}_metadata"
        if metadata_column not in result.columns:
            continue
        if column in authors.columns:
            result[column] = result[column].combine_first(result[metadata_column])
            result = result.drop(columns=[metadata_column])
        else:
            result = result.rename(columns={metadata_column: column})
    return result


def add_mobility_author_profile_ids(resolved_authors: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic mobility IDs and flag overlapping nonidentical profiles.

    :param resolved_authors: Disambiguation selections.
    :return: Copy enriched with profile ID and ambiguity fields.
    """
    result = resolved_authors.copy()
    profile_sets = result.apply(selected_author_profile_tuple, axis=1)
    result[AUTHOR_PROFILE_ID_SET_KEY] = profile_sets.map(author_profile_set_key)
    result[MOBILITY_AUTHOR_PROFILE_ID] = profile_sets.map(mobility_profile_id)
    profiles_by_id: dict[int, set[tuple[int, ...]]] = {}
    for profile in profile_sets:
        for identifier in profile:
            profiles_by_id.setdefault(identifier, set()).add(profile)
    ambiguous = {
        profile for profiles in profiles_by_id.values() if len(profiles) > 1 for profile in profiles
    }
    result[AUTHOR_PROFILE_AMBIGUOUS] = profile_sets.map(lambda profile: profile in ambiguous)
    result[AUTHOR_PROFILE_AMBIGUITY_REASON] = result[AUTHOR_PROFILE_AMBIGUOUS].map(
        lambda value: "overlapping_nonidentical_selected_author_ids" if value else ""
    )
    return result


def selected_author_profile_tuple(row: pd.Series) -> tuple[int, ...]:
    """Return selected profile IDs, using legacy scalar fields when needed.

    :param row: One resolved-author record.
    :return: Sorted selected IDs, or an empty tuple when none are available.
    """
    if SELECTED_AUTHOR_IDS in row.index:
        identifiers = parse_sequence(row.get(SELECTED_AUTHOR_IDS))
        if identifiers:
            return tuple(sorted(identifiers))
    for column in (SELECTED_DISAMBIGUATED_CLUSTER_ID, AUTHOR_ID):
        if column in row.index:
            identifier = parse_single_identifier(row.get(column))
            if identifier is not None:
                return (identifier,)
    return ()


def author_profile_set_key(profile_set: tuple[int, ...]) -> str:
    """Serialize profile IDs in stable comma-separated form.

    :param profile_set: Selected IDs for one profile.
    :return: Comma-separated identifier string.
    """
    return ",".join(str(identifier) for identifier in profile_set)


def mobility_profile_id(profile_set: tuple[int, ...]) -> int | None:
    """Return a stable identifier for a selected profile.

    :param profile_set: Selected IDs for one profile.
    :return: The singleton ID, a deterministic negative ID, or ``None``.
    """
    if not profile_set:
        return None
    if len(profile_set) == 1:
        return profile_set[0]
    digest = hashlib.sha1(author_profile_set_key(profile_set).encode("ascii")).hexdigest()
    return -(int(digest[:15], 16) % 9_000_000_000_000_000 + 1)


def parse_single_identifier(value: Any) -> int | None:
    """Parse the first valid integral identifier from a value.

    :param value: Scalar or serialized identifier value.
    :return: Parsed identifier, or ``None`` when no value is usable.
    """
    identifiers = parse_sequence(value)
    return identifiers[0] if identifiers else None


def count_publication_authors(article_authors: pd.DataFrame) -> pd.DataFrame:
    """Count unique author IDs per publication before local-author filtering."""
    if EID not in article_authors or ARTICLE_AUTHOR_ID not in article_authors:
        return pd.DataFrame(columns=[EID, PUBLICATION_AUTHOR_COUNT])
    authors = article_authors[[EID, ARTICLE_AUTHOR_ID]].copy()
    authors[ARTICLE_AUTHOR_ID] = authors[ARTICLE_AUTHOR_ID].map(normalize_identifier)
    authors = authors.dropna(subset=[EID, ARTICLE_AUTHOR_ID]).drop_duplicates()
    if authors.empty:
        return pd.DataFrame(columns=[EID, PUBLICATION_AUTHOR_COUNT])
    return (
        authors.groupby(EID, sort=False)[ARTICLE_AUTHOR_ID]
        .nunique()
        .rename(PUBLICATION_AUTHOR_COUNT)
        .reset_index()
    )


def attach_publication_author_counts(
    article_authors: pd.DataFrame, publication_author_counts: pd.DataFrame
) -> pd.DataFrame:
    """Attach counts when a usable count table and publication key exist."""
    if publication_author_counts.empty or EID not in article_authors:
        return article_authors
    return article_authors.merge(publication_author_counts, on=EID, how="left")


def extract_publication_general_asjc_subject_areas(article_authors: pd.DataFrame) -> pd.DataFrame:
    """Derive sorted ASJC level-two labels once per publication."""
    if EID not in article_authors or SUBJECT_AREAS not in article_authors:
        return pd.DataFrame(columns=[EID, PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS])
    publications = article_authors[[EID, SUBJECT_AREAS]].dropna(subset=[EID])
    publications = publications.drop_duplicates(subset=[EID])
    if publications.empty:
        return pd.DataFrame(columns=[EID, PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS])
    result = publications.copy()
    result[PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS] = result[SUBJECT_AREAS].map(
        general_asjc_subject_area_dict
    )
    return result[[EID, PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS]]


def attach_publication_general_asjc_subject_areas(
    article_authors: pd.DataFrame, publication_subject_areas: pd.DataFrame
) -> pd.DataFrame:
    """Attach publication-level ASJC summaries if available."""
    if publication_subject_areas.empty or EID not in article_authors:
        return article_authors
    return article_authors.merge(publication_subject_areas, on=EID, how="left")


def general_asjc_subject_area_dict(value: Any) -> dict[int, str]:
    """Map recognized ASJC codes to their sorted level-two descriptions."""
    result: dict[int, str] = {}
    for code in subject_area_codes(value):
        parent = level_two_asjc_code(code)
        entry = asjc_entry(parent) if parent is not None else None
        if entry is not None:
            result[parent] = entry.description  # type: ignore[index]
    return dict(sorted(result.items()))


def subject_area_codes(value: object) -> set[int]:
    """Extract integral subject codes from parsed or serialized subject data."""
    if value is None:
        return set()
    if isinstance(value, dict):
        return subject_area_codes_from_entry(value)
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        items = value.tolist() if isinstance(value, (np.ndarray, pd.Series)) else value
        return {code for item in items for code in subject_area_codes(item)}
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return set()
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return set()
    parsed = parse_subject_areas(text)
    if parsed:
        return {int(area.code) for area in parsed if area.code is not None}
    return parse_subject_area_code_tokens(text)


def subject_area_codes_from_entry(entry: Any) -> set[int]:
    """Extract a code from a subject entry, falling back to token parsing."""
    if not isinstance(entry, dict):
        return parse_subject_area_code_tokens(str(entry))
    code = parse_subject_area_code(entry.get("code"))
    return {code} if code is not None else set()


def parse_subject_area_code_tokens(value: str) -> set[int]:
    """Parse comma- or semicolon-delimited subject codes."""
    return {
        code
        for token in value.replace(";", ",").split(",")
        if (code := parse_subject_area_code(token)) is not None
    }


def parse_subject_area_code(value: Any) -> int | None:
    """Parse a finite, nonzero integral subject-area code."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        numeric = float(str(value).strip())
    except ValueError:
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    code = int(numeric)
    return code if code != 0 else None


def normalize_identifier(value: Any) -> str | None:
    """Normalize null and integral-float spellings of identifiers."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return None
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def clean_affiliations(affiliations: pd.DataFrame) -> pd.DataFrame:
    """Normalize parent precedence and preferred affiliation names."""
    result = affiliations.copy()
    for column in (
        AFFILIATION_ID_PARENT,
        AFFILIATION_PARENT_SOURCE,
        AFFILIATION_ID_PARENT_CHATGPT,
        AFFILIATION_ID_PARENT_SPLINK_CHATGPT,
        AFFILIATION_UMBRELLA_ID,
    ):
        ensure_column(result, column, np.nan)
    for column in _PARENT_AFFILIATION_PRECEDENCE:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    nonempty_parent_columns = [
        result[column] for column in _PARENT_AFFILIATION_PRECEDENCE if result[column].notna().any()
    ]
    if nonempty_parent_columns:
        parent_ids = nonempty_parent_columns[0]
        for candidate_ids in nonempty_parent_columns[1:]:
            parent_ids = parent_ids.combine_first(candidate_ids)
    else:
        parent_ids = result[_PARENT_AFFILIATION_PRECEDENCE[0]]
    result[AFFILIATION_ID_PARENT] = parent_ids
    if AFFILIATION_PREFERRED_NAME not in result:
        result[AFFILIATION_PREFERRED_NAME] = result.get(NAME)
    else:
        fallback = result.get(NAME, pd.Series(pd.NA, index=result.index))
        result[AFFILIATION_PREFERRED_NAME] = result[AFFILIATION_PREFERRED_NAME].combine_first(
            fallback
        )
    return result


def explode_articles_to_author_affiliations(
    articles: pd.DataFrame, affiliations: pd.DataFrame, *, max_workers: int = 3
) -> pd.DataFrame:
    """Explode publication authors and resolve affiliation parents and positions."""
    affiliation_map = get_affiliation_id_map(affiliations)
    article_authors = get_articles_per_author(articles)
    return parallelize_dataframe(
        apply_parent_affiliation_id_and_idx,
        article_authors,
        affiliation_map=affiliation_map,
        chunk_size=5000,
        max_workers=max_workers,
    )


def get_articles_per_author(article_df: pd.DataFrame) -> pd.DataFrame:
    """Explode parallel author and affiliation lists using source delimiters."""
    frame = article_df.copy()
    if AFFILIATION_ID in frame:
        frame = frame.rename(columns={AFFILIATION_ID: ARTICLE_AFFILIATION_ID})
    frame[_ARTICLE_ROW_ID] = np.arange(len(frame))
    columns = [ARTICLE_AFFILIATION_ID]
    if ARTICLE_AUTHOR_ID in frame:
        columns.append(ARTICLE_AUTHOR_ID)
    for column in columns:
        frame.loc[:, column] = frame[column].str.split(_AUTHOR_SEPARATOR)
    frame = frame.explode(column=columns, ignore_index=True)
    frame = frame[frame[ARTICLE_AFFILIATION_ID] != ""]
    return frame[frame[ARTICLE_AFFILIATION_ID].notnull()].reset_index(drop=True)


def get_affiliation_id_map(affiliations: pd.DataFrame) -> dict[Any, Any]:
    """Build a child-to-parent mapping for nonmissing parent relationships."""
    if AFFILIATION_ID_PARENT not in affiliations:
        return {}
    parents = affiliations.loc[
        affiliations[AFFILIATION_ID_PARENT].notna(), [AFFILIATION_ID, AFFILIATION_ID_PARENT]
    ]
    if parents.empty:
        return {}
    parents = parents.copy()
    parents[AFFILIATION_ID_PARENT] = parents[AFFILIATION_ID_PARENT].astype("int64")
    return parents.set_index(AFFILIATION_ID)[AFFILIATION_ID_PARENT].to_dict()


def apply_parent_affiliation_id_and_idx(
    article_author_df: pd.DataFrame, affiliation_map: dict[Any, Any]
) -> pd.DataFrame:
    """Resolve affiliation IDs to parent IDs while preserving original IDs."""
    frame = article_author_df.copy()
    frame[UNMAPPED_AFFILIATION_ID] = frame[ARTICLE_AFFILIATION_ID]
    frame[ARTICLE_AFFILIATION_INDEX] = frame[ARTICLE_AFFILIATION_ID].apply(
        lambda value: _get_parent_affiliation_idx(value, affiliation_map)
    )
    frame[ARTICLE_AFFILIATION_ID] = frame[ARTICLE_AFFILIATION_ID].apply(
        lambda value: _get_parent_affiliation_id(value, affiliation_map)
    )
    return frame


def _parent_ids(value: str, affiliation_map: dict[Any, Any]) -> np.ndarray:
    """Resolve delimited affiliation IDs to their mapped parent IDs.

    :param value: Delimited affiliation identifiers.
    :param affiliation_map: Child-to-parent affiliation mapping.
    :return: Array of unique resolved identifiers.
    """
    ids = np.asarray(value.split(_MULTIPLE_AFFILIATION_SEPARATOR)).astype(np.int64)
    return np.asarray(list({affiliation_map.get(identifier, identifier) for identifier in ids}))


def _get_parent_affiliation_idx(value: str, affiliation_map: dict[Any, Any]) -> dict[Any, Any]:
    """Map sorted parent IDs to their positions in source order.

    :param value: Delimited affiliation identifiers.
    :param affiliation_map: Child-to-parent affiliation mapping.
    :return: Parent identifier-to-position mapping.
    """
    ids = _parent_ids(value, affiliation_map)
    sorted_ids = sorted(ids)
    return dict(zip(sorted_ids, list(np.argsort(ids))))


def _get_parent_affiliation_id(value: str, affiliation_map: dict[Any, Any]) -> list[np.int64]:
    """Return sorted resolved parent affiliation IDs.

    :param value: Delimited affiliation identifiers.
    :param affiliation_map: Child-to-parent affiliation mapping.
    :return: Sorted parent identifiers.
    """
    return list(np.array(sorted(_parent_ids(value, affiliation_map))).astype(np.int64))


def apply_historical_author_ids(
    article_authors: pd.DataFrame, resolved_authors: pd.DataFrame
) -> pd.DataFrame:
    """Remap historical article IDs to their selected current IDs."""
    history = selected_history_long(resolved_authors)
    if history.empty:
        return coerce_int_column(article_authors, ARTICLE_AUTHOR_ID)
    frame = coerce_int_column(article_authors, ARTICLE_AUTHOR_ID).merge(
        history, left_on=ARTICLE_AUTHOR_ID, right_on=_HISTORICAL_AUTHOR_ID, how="left"
    )
    frame[ARTICLE_AUTHOR_ID] = np.where(
        frame[_CURRENT_AUTHOR_ID].notna(), frame[_CURRENT_AUTHOR_ID], frame[ARTICLE_AUTHOR_ID]
    )
    return frame.drop(columns=[_CURRENT_AUTHOR_ID, _HISTORICAL_AUTHOR_ID])


def apply_disambiguated_author_ids(
    article_authors: pd.DataFrame, resolved_authors: pd.DataFrame
) -> pd.DataFrame:
    """Map each selected profile ID to its mobility profile ID."""
    clusters = selected_author_cluster_long(resolved_authors)
    if clusters.empty:
        return article_authors
    frame = coerce_int_column(article_authors, ARTICLE_AUTHOR_ID).merge(
        clusters, left_on=ARTICLE_AUTHOR_ID, right_on=_SELECTED_AUTHOR_ID, how="left"
    )
    frame[ARTICLE_AUTHOR_ID] = np.where(
        frame[_DISAMBIGUATED_AUTHOR_ID].notna(),
        frame[_DISAMBIGUATED_AUTHOR_ID],
        frame[ARTICLE_AUTHOR_ID],
    )
    return frame.drop(columns=[_SELECTED_AUTHOR_ID, _DISAMBIGUATED_AUTHOR_ID])


def keep_resolved_authors(
    article_authors: pd.DataFrame, resolved_authors: pd.DataFrame
) -> pd.DataFrame:
    """Keep rows whose remapped author ID is present among resolved selections."""
    identifiers = resolved_author_ids(resolved_authors)
    if identifiers.empty:
        return article_authors
    return article_authors[article_authors[ARTICLE_AUTHOR_ID].isin(identifiers)].copy()


def explode_affiliations(article_authors: pd.DataFrame) -> pd.DataFrame:
    """Return one row per affiliation and coerce affiliation IDs to int64."""
    frame = article_authors.explode(ARTICLE_AFFILIATION_ID).reset_index(drop=True)
    frame = coerce_int_column(frame, ARTICLE_AFFILIATION_ID)
    frame = frame.dropna(subset=[ARTICLE_AFFILIATION_ID]).copy()
    frame[ARTICLE_AFFILIATION_ID] = frame[ARTICLE_AFFILIATION_ID].astype("int64")
    return frame


def selected_history_long(resolved_authors: pd.DataFrame) -> pd.DataFrame:
    """Combine disambiguation history maps and historical identifier lists."""
    frames = []
    if SELECTED_HISTORY in resolved_authors:
        frames.append(selected_history_mapping_long(resolved_authors))
    if HISTORICAL_IDENTIFIER in resolved_authors:
        frames.append(historical_identifier_long(resolved_authors))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty_history_frame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .astype("int64")
        .reset_index(drop=True)
    )


def selected_history_mapping_long(resolved_authors: pd.DataFrame) -> pd.DataFrame:
    """Expand selected-history mapping values into current/previous ID pairs."""
    if SELECTED_HISTORY not in resolved_authors:
        return empty_history_frame()
    history = resolved_authors[[SELECTED_HISTORY]].copy()
    history[_HISTORY] = history[SELECTED_HISTORY].map(parse_mapping)
    history = history[history[_HISTORY].map(bool)]
    if history.empty:
        return empty_history_frame()
    rows = [
        (current, previous)
        for mapping in history[_HISTORY]
        for current, previous_ids in mapping.items()
        for previous in previous_ids
    ]
    return pd.DataFrame(rows, columns=[_CURRENT_AUTHOR_ID, _HISTORICAL_AUTHOR_ID]).astype("int64")


def historical_identifier_long(resolved_authors: pd.DataFrame) -> pd.DataFrame:
    """Expand historical identifier arrays against each selected author ID."""
    column = resolved_author_id_column(resolved_authors)
    if column is None:
        return empty_history_frame()
    history = resolved_authors[[column, HISTORICAL_IDENTIFIER]].copy()
    history[_HISTORICAL_AUTHOR_ID] = history[HISTORICAL_IDENTIFIER].map(parse_sequence)
    history = history.explode(_HISTORICAL_AUTHOR_ID).dropna(subset=[_HISTORICAL_AUTHOR_ID])
    if history.empty:
        return empty_history_frame()
    return history.rename(columns={column: _CURRENT_AUTHOR_ID})[
        [_CURRENT_AUTHOR_ID, _HISTORICAL_AUTHOR_ID]
    ].astype("int64")


def selected_author_cluster_long(resolved_authors: pd.DataFrame) -> pd.DataFrame:
    """Build selected-author-ID to mobility-profile-ID rows."""
    prepared = add_mobility_author_profile_ids(resolved_authors)
    target = (
        MOBILITY_AUTHOR_PROFILE_ID
        if MOBILITY_AUTHOR_PROFILE_ID in prepared
        else (SELECTED_DISAMBIGUATED_CLUSTER_ID)
    )
    if not {target, SELECTED_AUTHOR_IDS}.issubset(prepared.columns):
        return empty_cluster_frame()
    clusters = prepared[[target, SELECTED_AUTHOR_IDS]].dropna().copy()
    clusters[_SELECTED_AUTHOR_ID] = clusters[SELECTED_AUTHOR_IDS].map(parse_sequence)
    clusters = clusters.explode(_SELECTED_AUTHOR_ID).dropna(subset=[_SELECTED_AUTHOR_ID])
    if clusters.empty:
        return empty_cluster_frame()
    return clusters.rename(columns={target: _DISAMBIGUATED_AUTHOR_ID})[
        [_DISAMBIGUATED_AUTHOR_ID, _SELECTED_AUTHOR_ID]
    ].astype("int64")


def resolved_author_ids(resolved_authors: pd.DataFrame) -> pd.Series:
    """Return distinct numeric resolved IDs in source order."""
    column = resolved_author_id_column(resolved_authors)
    if column is None:
        return pd.Series(dtype="int64")
    return (
        pd.to_numeric(resolved_authors[column], errors="coerce")
        .dropna()
        .astype("int64")
        .drop_duplicates()
    )


def resolved_author_id_column(resolved_authors: pd.DataFrame) -> str | None:
    """Select the mobility profile, selected cluster, or raw author ID field."""
    for column in (MOBILITY_AUTHOR_PROFILE_ID, SELECTED_DISAMBIGUATED_CLUSTER_ID, AUTHOR_ID):
        if column in resolved_authors:
            return column
    return None


def parse_mapping(value: Any) -> dict[str, list[int]]:
    """Parse a mapping of current profile IDs to historical IDs."""
    parsed = parse_literal(value)
    if not isinstance(parsed, dict):
        return {}
    return {
        str(identifier): ids
        for identifier, history in parsed.items()
        if (ids := parse_sequence(history))
    }


def parse_sequence(value: Any) -> list[int]:
    """Parse unique integral IDs from a scalar or sequence."""
    parsed = parse_literal(value)
    if parsed is None:
        return []
    values = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
    series = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna()
    return series.astype("int64").drop_duplicates().tolist()


def parse_literal(value: Any) -> Any:
    """Parse Python-literal serialized values, retaining unparseable text."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (dict, list, tuple, set)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def coerce_int_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy with a column numerically coerced."""
    result = frame.copy()
    result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def ensure_column(frame: pd.DataFrame, column: str, default: Any) -> None:
    """Add a missing schema column with its given default value."""
    if column not in frame:
        frame[column] = default


def empty_history_frame() -> pd.DataFrame:
    """Return an empty history mapping frame with the expected columns."""
    return pd.DataFrame(columns=[_CURRENT_AUTHOR_ID, _HISTORICAL_AUTHOR_ID])


def empty_cluster_frame() -> pd.DataFrame:
    """Return an empty selected-profile mapping frame."""
    return pd.DataFrame(columns=[_DISAMBIGUATED_AUTHOR_ID, _SELECTED_AUTHOR_ID])


def parallelize_dataframe(
    function: Callable[..., pd.DataFrame],
    frame: pd.DataFrame,
    *,
    chunk_size: int,
    max_workers: int,
    **kwargs: Any,
) -> pd.DataFrame:
    """Apply a DataFrame transform sequentially or over ordered process chunks.

    :param function: Row-preserving transform applied to each DataFrame chunk.
    :param frame: Input rows to transform.
    :param chunk_size: Target number of rows in each process chunk.
    :param max_workers: Maximum number of worker processes.
    :param kwargs: Additional keyword arguments forwarded to the transform.
    :return: Transformed rows in their original index order when chunk indices
        are present in the transformed frames.
    """
    if frame.empty:
        return frame
    number_of_chunks = ceil(len(frame) / chunk_size)
    if max_workers <= 1 or number_of_chunks <= 2:
        return function(frame, **kwargs)
    chunks = max(int(len(frame) / chunk_size), 1)
    size = len(frame) // chunks + (len(frame) % chunks > 0)
    pieces = [frame.iloc[start : start + size].copy() for start in range(0, len(frame), size)]
    for piece in pieces:
        piece[_PARALLEL_INDEX] = piece.index
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(function, piece, **kwargs) for piece in pieces]
        output = pd.concat(
            [future.result() for future in as_completed(futures)],
            ignore_index=True,
        )
    if _PARALLEL_INDEX in output.columns:
        return (
            output.set_index(output[_PARALLEL_INDEX])
            .sort_index()
            .drop(columns=[_PARALLEL_INDEX], errors="ignore")
            .copy()
        )
    return output.reset_index(drop=True)
