# Copyright © 2026 TU Wien

"""Collect, validate, and persist name-disambiguation artifacts."""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from typing import Any

import pandas as pd
import pandera.pandas as pa

from acib.common.titles import add_title_class
from acib.constants import (
    AFFILIATION_ID_CURRENT,
    AFFILIATION_IDS_CURRENT,
    AUTHOR_ID,
    HISTORICAL_IDENTIFIER,
    PERSON_TITLE,
    RESEARCHER_ID,
    TITLE,
    UNIVERSITY_NAME,
)
from acib.name_disambiguation.config import NameDisambiguationConfig
from acib.name_disambiguation.constants import (
    AUTHOR_HISTORY,
    AUTHOR_IDS,
    CURRENT_AFFILIATION_IDS,
    LEFT_AUTHOR_ID,
    LEFT_HISTORY,
    POSSIBLE_CLUSTER_HISTORY,
    POSSIBLE_CLUSTERS,
    RIGHT_AUTHOR_ID,
    RIGHT_HISTORY,
    SELECTED_AUTHOR_IDS,
    SELECTED_HISTORY,
)
from acib.name_disambiguation.models import BatchDisambiguationResult
from acib.name_disambiguation.reporting import render_json_report, render_text_report
from acib.name_disambiguation.scoring_evidence import (
    normalize_current_affiliation_id,
    parse_current_affiliation_id_values,
)
from acib.name_matching.constants import RESEARCHER_NAME
from acib.schemas.authors import (
    AuthorClusterMembershipSchema,
    AuthorClusterSelectionScoreSchema,
    AuthorPairDecisionSchema,
)
from acib.schemas.researchers import ResearcherDisambiguationSelectionSchema

_HISTORY_NULL_VALUES = {"", "nan", "none", "null", "[]"}
_REPORT_COLUMN = "report"


def write_all_outputs(
    batch_results: list[BatchDisambiguationResult],
    config: NameDisambiguationConfig,
    *,
    overwrite: bool = False,
) -> None:
    """Write all configured disambiguation artifacts.

    :param batch_results: Per-researcher decisions, clusters, and selections.
    :param config: Paths and output options for this run.
    :param overwrite: Whether to replace existing disambiguation outputs.
    :raises FileExistsError: If outputs already exist and replacement is not enabled.
    """
    existing = [path for path in config.outputs.all_paths() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Disambiguation outputs already exist; pass overwrite=True to replace: "
            + ", ".join(map(str, existing))
        )

    outputs = (
        (
            config.outputs.pair_decisions,
            collect_pair_decisions(batch_results),
            AuthorPairDecisionSchema,
        ),
        (
            config.outputs.clusters,
            collect_clusters(batch_results),
            AuthorClusterMembershipSchema,
        ),
        (
            config.outputs.final_author_selection,
            collect_final_selections(batch_results),
            ResearcherDisambiguationSelectionSchema,
        ),
        (
            config.outputs.final_author_selection_scores,
            collect_final_selection_scores(batch_results),
            AuthorClusterSelectionScoreSchema,
        ),
    )
    for path, frame, schema in outputs:
        validated = _validate_output_frame(frame, schema)
        path.parent.mkdir(parents=True, exist_ok=True)
        validated.to_csv(path, index=False)

    text_report = (
        "\n".join(
            render_text_report(
                result.decisions,
                result.clusters,
                result.final_selection,
                result.final_selection_scores,
            ).rstrip()
            for result in batch_results
        ).rstrip()
        + "\n"
    )
    json_rows = [
        {
            RESEARCHER_ID: result.sc_id,
            _REPORT_COLUMN: render_json_report(
                result.decisions,
                result.clusters,
                result.final_selection,
                result.final_selection_scores,
            ),
        }
        for result in batch_results
    ]
    config.outputs.report_txt.parent.mkdir(parents=True, exist_ok=True)
    config.outputs.report_txt.write_text(text_report, encoding="utf-8")
    config.outputs.report_jsonl.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(json_rows).to_json(
        config.outputs.report_jsonl,
        orient="records",
        lines=True,
        force_ascii=False,
    )


def _validate_output_frame(frame: pd.DataFrame, schema: type[pa.DataFrameModel]) -> pd.DataFrame:
    """Validate an artifact, adding the declared columns for empty results.

    :param frame: Collected output records.
    :param schema: Pandera model describing the artifact.
    :return: Validated dataframe.
    """
    if frame.empty:
        frame = frame.copy()
        for column_name, column in schema.to_schema().columns.items():
            if column_name not in frame:
                dtype = column.dtype
                pandas_dtype = getattr(dtype, "type", object) if dtype is not None else object
                frame[column_name] = pd.Series(index=frame.index, dtype=pandas_dtype)
    return schema.validate(frame, lazy=True)


def collect_pair_decisions(batch_results: list[BatchDisambiguationResult]) -> pd.DataFrame:
    """Combine pair decisions and attach historical author identifiers.

    :param batch_results: Per-researcher disambiguation results.
    :return: Combined pair-decision table with history fields.
    """
    frames = []
    for result in batch_results:
        if result.decisions.empty:
            continue
        frame = result.decisions.copy()
        frame[RESEARCHER_ID] = result.sc_id
        histories = author_history_map(result.authors)
        frame[LEFT_HISTORY] = frame[LEFT_AUTHOR_ID].map(
            lambda value, history_map=histories: history_for_author_ids([value], history_map)
        )
        frame[RIGHT_HISTORY] = frame[RIGHT_AUTHOR_ID].map(
            lambda value, history_map=histories: history_for_author_ids([value], history_map)
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_clusters(batch_results: list[BatchDisambiguationResult]) -> pd.DataFrame:
    """Combine cluster memberships and attach author history.

    :param batch_results: Per-researcher disambiguation results.
    :return: Combined author-cluster membership table.
    """
    frames = []
    for result in batch_results:
        if result.clusters.empty:
            continue
        frame = result.clusters.copy()
        frame[RESEARCHER_ID] = result.sc_id
        histories = author_history_map(result.authors)
        frame[AUTHOR_HISTORY] = frame[AUTHOR_ID].map(
            lambda value, history_map=histories: history_for_author_ids([value], history_map)
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_final_selections(batch_results: list[BatchDisambiguationResult]) -> pd.DataFrame:
    """Collect final selections with researcher metadata and author history.

    :param batch_results: Per-researcher disambiguation results.
    :return: Final-selection rows enriched with metadata and title classification.
    """
    rows: list[dict[str, object]] = []
    for result in batch_results:
        history = author_history_map(result.authors)
        current_affiliations = current_affiliation_ids_by_author(result.authors)
        row = asdict(result.final_selection)
        row[RESEARCHER_NAME] = first_author_value(result.authors, RESEARCHER_NAME)
        row[PERSON_TITLE] = first_author_value(result.authors, PERSON_TITLE)
        row[UNIVERSITY_NAME] = first_author_value(result.authors, UNIVERSITY_NAME)
        row[CURRENT_AFFILIATION_IDS] = selected_current_affiliation_ids(
            row[SELECTED_AUTHOR_IDS], current_affiliations
        )
        row[SELECTED_HISTORY] = history_for_author_ids(row[SELECTED_AUTHOR_IDS], history)
        row[POSSIBLE_CLUSTER_HISTORY] = [
            history_for_author_ids(author_ids, history) for author_ids in row[POSSIBLE_CLUSTERS]
        ]
        rows.append(row)
    selections = pd.DataFrame(rows)
    if selections.empty or PERSON_TITLE not in selections:
        return selections
    return add_title_class(selections, output_title_column=TITLE)


def collect_final_selection_scores(batch_results: list[BatchDisambiguationResult]) -> pd.DataFrame:
    """Combine candidate cluster scores and attach author histories.

    :param batch_results: Per-researcher disambiguation results.
    :return: Combined candidate-cluster score table.
    """
    frames = []
    for result in batch_results:
        if result.final_selection_scores.empty:
            continue
        frame = result.final_selection_scores.copy()
        frame[RESEARCHER_ID] = result.sc_id
        histories = author_history_map(result.authors)
        frame[AUTHOR_HISTORY] = frame[AUTHOR_IDS].map(
            lambda author_ids, history_map=histories: history_for_author_ids(
                author_ids, history_map
            )
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def current_affiliation_ids_by_author(authors: pd.DataFrame) -> dict[str, set[str]]:
    """Return the current affiliation IDs carried by each Scopus author profile.

    :param authors: Candidate author rows with optional current-affiliation columns.
    :return: Normalized affiliation IDs grouped by normalized author ID.
    """
    if AUTHOR_ID not in authors:
        return {}
    columns = [
        column for column in (AFFILIATION_IDS_CURRENT, AFFILIATION_ID_CURRENT) if column in authors
    ]
    if not columns:
        return {}
    result: dict[str, set[str]] = {}
    for row in authors[[AUTHOR_ID, *columns]].itertuples(index=False, name=None):
        author_id = normalize_author_lookup_id(row[0])
        if not author_id:
            continue
        affiliation_ids = {
            normalized
            for value in row[1:]
            for parsed in parse_current_affiliation_id_values(value)
            if (normalized := normalize_current_affiliation_id(parsed))
        }
        if affiliation_ids:
            result.setdefault(author_id, set()).update(affiliation_ids)
    return result


def selected_current_affiliation_ids(
    author_ids: list[str], current_affiliations: dict[str, set[str]]
) -> list[str]:
    """Return the ordered union of current affiliations for selected author IDs.

    :param author_ids: Selected Scopus author IDs.
    :param current_affiliations: Affiliation IDs keyed by normalized author ID.
    :return: Unique affiliation IDs sorted numerically where possible.
    """
    ids = {
        affiliation_id
        for author_id in author_ids
        for affiliation_id in current_affiliations.get(normalize_author_lookup_id(author_id), set())
    }
    return sorted(ids, key=current_affiliation_sort_key)


def normalize_author_lookup_id(value: Any) -> str:
    """Normalize integer-like author IDs for history and affiliation lookups.

    :param value: Scopus author ID representation.
    :return: Normalized lookup key.
    """
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return str(int(text.split(".", 1)[0]))
    return text


def current_affiliation_sort_key(value: str) -> tuple[int, int, str]:
    """Sort numeric affiliation IDs numerically before nonnumeric IDs.

    :param value: Normalized affiliation ID.
    :return: Sort tuple placing numeric IDs first.
    """
    try:
        numeric = int(value)
    except ValueError:
        return (1, 0, value)
    return (0, numeric, value)


def first_author_value(authors: pd.DataFrame, column: str) -> str:
    """Return the first distinct nonblank researcher metadata value.

    :param authors: Candidate author rows with researcher metadata.
    :param column: Metadata column to read.
    :return: First distinct nonblank value, or empty text.
    """
    if column not in authors:
        return ""
    values = authors[column].dropna().astype(str).str.strip()
    values = values[values.ne("")].drop_duplicates()
    return values.iloc[0] if not values.empty else ""


def author_history_map(authors: pd.DataFrame) -> dict[str, list[int | str]]:
    """Build a normalized Scopus-author-to-history lookup.

    :param authors: Candidate rows containing current and historical author IDs.
    :return: Nonempty history lists keyed by normalized author ID.
    """
    if AUTHOR_ID not in authors or HISTORICAL_IDENTIFIER not in authors:
        return {}
    result: dict[str, list[int | str]] = {}
    for author_id, raw_history in authors[[AUTHOR_ID, HISTORICAL_IDENTIFIER]].itertuples(
        index=False, name=None
    ):
        history = parse_historical_identifiers(raw_history)
        if history:
            result.setdefault(normalize_author_lookup_id(author_id), history)
    return result


def history_for_author_ids(
    author_ids: list[str], history_by_author: dict[str, list[int | str]]
) -> dict[str, list[int | str]]:
    """Return history entries for the supplied selected author IDs.

    :param author_ids: Selected current author IDs.
    :param history_by_author: Historical IDs keyed by current author ID.
    :return: Histories for IDs present in the lookup.
    """
    return {
        normalized: history
        for author_id in author_ids
        if (normalized := normalize_author_lookup_id(author_id))
        and (history := history_by_author.get(normalized))
    }


def parse_historical_identifiers(value: Any) -> list[int | str]:
    """Parse historical Scopus identifiers without changing malformed values silently.

    :param value: Serialized or structured history identifiers.
    :return: Unique identifiers normalized to integers where possible.
    """
    if value is None or value is pd.NA:
        return []
    if isinstance(value, (list, tuple, set)):
        values = value
    elif isinstance(value, bool):
        return []
    elif isinstance(value, int):
        return [value]
    elif isinstance(value, float):
        return [int(value)] if math.isfinite(value) and value.is_integer() else []
    else:
        text = str(value).strip()
        if not text or text.casefold() in _HISTORY_NULL_VALUES:
            return []
        matches = re.findall(r"\d+", text)
        values = matches if matches else re.split(r"[,;|\s]+", text)
    normalized = list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    return [int(item) if item.isdigit() else item for item in normalized]
