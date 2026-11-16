# Copyright © 2026 TU Wien

"""Adapt observed OpenAlex author-affiliation rows for name matching."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from acib.common.names import normalize_person_name
from acib.constants import AFFILIATION_ID, AUTHOR_ID
from acib.name_matching.constants import (
    AUTHOR_NAME,
    NAME,
    NAME_VARIANTS,
    OPENALEX_AUTHOR_DISPLAY_NAME,
    OPENALEX_AUTHOR_NAME_ALTERNATIVES,
    OPENALEX_INSTITUTION_ID,
    OPENALEX_OBSERVATIONS,
    OPENALEX_RAW_AUTHOR_NAME,
    OPENALEX_RAW_AUTHOR_NAMES,
    OPENALEX_WORK_ID,
)

OPENALEX_SOURCE = "openalex"
_PARTICLES = frozenset({"da", "de", "del", "den", "der", "di", "la", "le", "van", "von"})


def _alternatives(value: object) -> list[str] | None:
    """Decode the observed Arrow JSON scalar or an already-decoded list.

    The parquet's ``arrow.json`` extension exposes each non-null cell as JSON
    text with pyarrow's ``to_pylist``. A caller may instead supply decoded
    values. Other representations are deliberately rejected rather than
    guessed.
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, str):
        value = json.loads(value, strict=True)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("author_name_alternatives must be a JSON string array or list of strings")
    return value


def _raw_names(value: object) -> list[str] | None:
    """Return observed raw-name strings in order, without coercing values."""
    if value is None or value is pd.NA:
        return None
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ValueError("raw_author_names must be a list of strings")
    return list(value)


def _add_evidence(
    evidence: dict[tuple[str, str], list[str]],
    source: str,
    raw_name: str,
    work_id: str | None = None,
) -> None:
    """Add one raw name and retain each distinct associated work identifier.

    :param evidence: Evidence mapping for one author-institution observation.
    :param source: Provenance label for the raw name.
    :param raw_name: Original, unmodified name string.
    :param work_id: Optional stable work identifier.
    """
    work_ids = evidence.setdefault((source, raw_name), [])
    if work_id is not None and work_id not in work_ids:
        work_ids.append(work_id)


def _parse_hypotheses(value: str) -> list[tuple[str, str, str] | None]:
    """Create conservative surname/given hypotheses without altering evidence.

    A comma explicitly indicates family-first order. For ambiguous no-comma
    strings, all non-empty boundaries are considered in both given-first and
    family-first orientations.
    """
    text = " ".join(value.split())
    if not text:
        return [None]
    if "," in text:
        surname, given = text.split(",", 1)
        surname, given = surname.strip(), given.strip()
        return [("family_first_comma", surname, given)] if surname and given else [None]
    tokens = text.split()
    if len(tokens) < 2:
        return [None]
    hypotheses: list[tuple[str, str, str] | None] = []
    for boundary in range(1, len(tokens)):
        given_tokens = tokens[:boundary]
        surname_tokens = tokens[boundary:]
        # Include the preceding particle as part of the surname when present.
        particle_start = boundary
        while particle_start > 0 and tokens[particle_start - 1].casefold() in _PARTICLES:
            particle_start -= 1
        if particle_start < boundary:
            given_tokens = tokens[:particle_start]
            surname_tokens = tokens[particle_start:]
        if given_tokens and surname_tokens:
            hypotheses.append(("given_first", " ".join(surname_tokens), " ".join(given_tokens)))
            hypotheses.append(
                ("family_first", " ".join(tokens[:boundary]), " ".join(tokens[boundary:]))
            )
    if hypotheses:
        return hypotheses
    return [None]


def openalex_candidate_rows(
    rows: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Adapt observed author-affiliation records into matching evidence rows.

    Each evidence row remains associated with its stable author and institution
    IDs. Profile display names, alternative names, and aggregated raw names
    retain their original spelling and are labeled with explicit provenance.
    IDs are copied as strings. Identical normalized hypotheses are combined per
    author-institution pair while their distinct raw evidence remains attached.

    :param rows: Already-loaded and decoded records from the observed derived
        author-affiliation schema.
    :return: Candidate rows with raw evidence and separate normalized matching
        hypotheses. Unsplittable evidence remains as a row with an empty
        ``author_name`` and a null hypothesis.
    :raises ValueError: If required IDs or structured name fields are malformed.

    The observed schema has a JSON-encoded Arrow extension for
    ``author_name_alternatives``, list-valued ``raw_author_names``, and decoded
    publication ``observations``. Duplicate raw observations are grouped by
    author, institution, source, and exact name; their work IDs remain attached
    to the evidence. Null-versus-empty source lists are represented by explicit
    flags. Other OpenAlex exports are not assumed to share these field names or
    semantics.
    """
    grouped_evidence: dict[tuple[str, str, bool, bool], dict[tuple[str, str], list[str]]] = {}
    for row in rows:
        author_id = row.get("author_id")
        institution_id = row.get(OPENALEX_INSTITUTION_ID)
        if not isinstance(author_id, str) or not isinstance(institution_id, str):
            raise TypeError("author_id and institution_id must be strings")
        alternatives = _alternatives(row.get(OPENALEX_AUTHOR_NAME_ALTERNATIVES))
        raw_names = _raw_names(row.get(OPENALEX_RAW_AUTHOR_NAMES))
        group_key = (author_id, institution_id, alternatives is None, raw_names is None)
        evidence = grouped_evidence.setdefault(group_key, {})
        display_name = row.get(OPENALEX_AUTHOR_DISPLAY_NAME)
        if isinstance(display_name, str):
            _add_evidence(evidence, "author_display_name", display_name)
        for name in alternatives or []:
            _add_evidence(evidence, "author_name_alternatives", name)
        for name in raw_names or []:
            _add_evidence(evidence, "raw_author_names", name)
        observations = row.get(OPENALEX_OBSERVATIONS)
        if observations is not None:
            if not isinstance(observations, (list, tuple)):
                raise ValueError("observations must be a decoded list of records")
            for observation in observations:
                if not isinstance(observation, Mapping):
                    raise TypeError("each observation must be a decoded record")
                raw_name = observation.get(OPENALEX_RAW_AUTHOR_NAME)
                if raw_name is None:
                    continue
                if not isinstance(raw_name, str):
                    raise TypeError("observations[].raw_author_name must be a string or null")
                work_id = observation.get(OPENALEX_WORK_ID)
                if work_id is not None and not isinstance(work_id, str):
                    raise TypeError("observations[].work_id must be a string or null")
                _add_evidence(evidence, "observation_raw_author_name", raw_name, work_id)

    output: list[dict[str, object]] = []
    for (
        author_id,
        institution_id,
        alternatives_is_null,
        raw_names_is_null,
    ), evidence in grouped_evidence.items():
        normalized_evidence: dict[
            tuple[str | None, str | None],
            list[tuple[str, str, list[str], str | None]],
        ] = {}
        for (source, raw_name), work_ids in evidence.items():
            for hypothesis in _parse_hypotheses(raw_name):
                normalized = (
                    f"{normalize_person_name(hypothesis[1])}, "
                    f"{normalize_person_name(hypothesis[2])}"
                    if hypothesis is not None
                    else ""
                )
                orientation = hypothesis[0] if hypothesis is not None else None
                key = (normalized or None, orientation)
                variant = (source, raw_name, work_ids, orientation)
                variants = normalized_evidence.setdefault(key, [])
                if variant not in variants:
                    variants.append(variant)
        for (hypothesis_name, hypothesis_orientation), variants in normalized_evidence.items():
            work_ids = list(
                dict.fromkeys(
                    work_id
                    for _, _, variant_work_ids, _ in variants
                    for work_id in variant_work_ids
                )
            )
            source, raw_name, _, _ = variants[0]
            output.append(
                {
                    AUTHOR_ID: author_id,
                    AFFILIATION_ID: institution_id,
                    "source": OPENALEX_SOURCE,
                    "name_source": source,
                    "work_id": work_ids[0] if len(work_ids) == 1 else None,
                    "work_ids": work_ids,
                    "name_orientation": hypothesis_orientation,
                    "raw_name": raw_name,
                    # The matcher reparses NAME. Give it the explicit
                    # hypothesis while retaining all untouched evidence above.
                    NAME: hypothesis_name or "",
                    "name_hypothesis": hypothesis_name,
                    AUTHOR_NAME: hypothesis_name or "",
                    NAME_VARIANTS: json.dumps(
                        [
                            {
                                "source": source_variant,
                                "raw_name": raw_name_variant,
                                "work_ids": variant_work_ids,
                                "orientation": orientation_variant,
                            }
                            for (
                                source_variant,
                                raw_name_variant,
                                variant_work_ids,
                                orientation_variant,
                            ) in variants
                        ],
                        ensure_ascii=False,
                    ),
                    "author_name_alternatives_is_null": alternatives_is_null,
                    "raw_author_names_is_null": raw_names_is_null,
                }
            )
    return pd.DataFrame(output)
