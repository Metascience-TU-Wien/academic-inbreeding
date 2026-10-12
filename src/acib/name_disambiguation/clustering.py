# Copyright © 2026 TU Wien

from __future__ import annotations

import ast
from collections import defaultdict
from itertools import product
from typing import Any

import pandas as pd

from acib.constants import AUTHOR_ID, GIVEN_NAME, SUBJECT_AREAS, SURNAME
from acib.name_disambiguation.constants import (
    CANONICAL_GIVEN_NAME,
    CANONICAL_INDEXED_NAME,
    CANONICAL_INITIALS,
    CANONICAL_NAME,
    CANONICAL_SURNAME,
    CLUSTER_ID,
    CLUSTER_SIZE,
    COMPONENTS,
    DECISION_LABEL,
    DISAMBIGUATED_CLUSTER_ID,
    DOCUMENT_COUNT,
    LEFT_AUTHOR_ID,
    ORCID,
    PUBLICATION_END,
    PUBLICATION_START,
    RIGHT_AUTHOR_ID,
    ROW_ID,
    SUBJECT_AREA_LABELS,
    TEMPORAL_HARD_CONFLICT_LABELS,
)
from acib.name_matching.constants import INDEXED_NAME, NAME

from .models import AuthorRecord

_CLUSTER_AUTHOR_ID_COLUMN = "_cluster_author_id"
_DOCUMENT_COUNT_NUMERIC_COLUMN = "document_count_numeric"
_PUBLICATION_SPAN_NUMERIC_COLUMN = "publication_span_numeric"
_SOURCE_ROW_ID_COLUMN = "id"

_HARD_DISTINCT_COMPONENT_LABELS = (
    frozenset({"given_gender_conflict", "orcid_conflict"}) | TEMPORAL_HARD_CONFLICT_LABELS
)


def cluster_authors(
    authors: pd.DataFrame | list[AuthorRecord], decisions: pd.DataFrame
) -> pd.DataFrame:
    """Build identity clusters from author records or a prepared author frame.

    :param authors: Canonicalized author rows or typed author records.
    :param decisions: Pairwise decisions used to form and constrain clusters.
    :return: One row per author profile with its assigned cluster metadata.
    """
    if isinstance(authors, pd.DataFrame):
        return cluster_author_frame(authors=authors, decisions=decisions)

    author_records = ensure_author_records(authors=authors)
    parent = {author.author_id: author.author_id for author in author_records}
    members_by_root = {author.author_id: {author.author_id} for author in author_records}
    hard_distinct_pairs = hard_distinct_author_pairs(decisions)

    def find(author_id: str) -> str:
        while parent[author_id] != author_id:
            parent[author_id] = parent[parent[author_id]]
            author_id = parent[author_id]
        return author_id

    def union(left_id: str, right_id: str) -> None:
        if left_id not in parent or right_id not in parent:
            return
        left_root = find(left_id)
        right_root = find(right_id)
        if components_have_hard_distinct_pair(
            members_by_root[left_root],
            members_by_root[right_root],
            hard_distinct_pairs,
        ):
            return
        if left_root != right_root:
            parent[right_root] = left_root
            members_by_root[left_root].update(members_by_root.pop(right_root))

    for decision in decisions.itertuples(index=False):
        if getattr(decision, DECISION_LABEL) == "same_author":
            union(
                str(getattr(decision, LEFT_AUTHOR_ID)),
                str(getattr(decision, RIGHT_AUTHOR_ID)),
            )

    grouped: dict[str, list[AuthorRecord]] = defaultdict(list)
    for author in author_records:
        grouped[find(author.author_id)].append(author)

    cluster_rows: list[dict[str, object]] = []
    for index, members in enumerate(
        sorted(grouped.values(), key=lambda records: records[0].author_id), start=1
    ):
        disambiguated_cluster_id = select_disambiguated_cluster_id(members)
        members.sort(
            key=lambda author: (author.surname.lower(), author.given_name.lower(), author.author_id)
        )
        for member in members:
            cluster_rows.append(
                {
                    CLUSTER_ID: index,
                    DISAMBIGUATED_CLUSTER_ID: disambiguated_cluster_id,
                    CLUSTER_SIZE: len(members),
                    ROW_ID: member.row_id,
                    AUTHOR_ID: member.author_id,
                    NAME: member.name,
                    CANONICAL_NAME: member.canonical_name,
                    SURNAME: member.surname,
                    CANONICAL_SURNAME: member.canonical_surname,
                    GIVEN_NAME: member.given_name,
                    CANONICAL_GIVEN_NAME: member.canonical_given_name,
                    INDEXED_NAME: member.indexed_name,
                    CANONICAL_INDEXED_NAME: member.canonical_indexed_name,
                    CANONICAL_INITIALS: member.canonical_initials,
                    ORCID: member.orcid,
                    PUBLICATION_START: member.publication_range.start,
                    PUBLICATION_END: member.publication_range.end,
                    SUBJECT_AREAS: sorted(
                        {area.abbreviation or area.area for area in member.subject_areas}
                    ),
                }
            )
    return pd.DataFrame(cluster_rows)


def ensure_author_records(authors: list[AuthorRecord]) -> list[AuthorRecord]:
    """Return the already-typed author-record sequence unchanged.

    :param authors: Author records from a parsed author frame.
    :return: The input records in their existing order.
    """
    return authors


def hard_distinct_author_pairs(decisions: pd.DataFrame) -> set[frozenset[str]]:
    """Return author pairs that must never be connected transitively.

    :param decisions: Pairwise author identity decisions.
    :return: Pairs with a hard-distinct evidence component.
    """
    if decisions.empty or DECISION_LABEL not in decisions:
        return set()
    required_columns = {LEFT_AUTHOR_ID, RIGHT_AUTHOR_ID, COMPONENTS}
    if not required_columns.issubset(decisions.columns):
        return set()

    pairs: set[frozenset[str]] = set()
    distinct = decisions[decisions[DECISION_LABEL].astype(str).eq("distinct")]
    for row in distinct.itertuples(index=False):
        if has_hard_distinct_component(getattr(row, COMPONENTS, None)):
            pairs.add(
                frozenset({str(getattr(row, LEFT_AUTHOR_ID)), str(getattr(row, RIGHT_AUTHOR_ID))})
            )
    return pairs


def has_hard_distinct_component(value: Any) -> bool:
    """Check whether serialized evidence contains a hard-distinct component.

    :param value: A list or serialized list of scoring components.
    :return: Whether a hard-distinct component is present.
    """
    return bool(component_labels(value) & _HARD_DISTINCT_COMPONENT_LABELS)


def component_labels(value: Any) -> set[str]:
    """Extract component labels from parsed or serialized evidence.

    :param value: A list or serialized list of scoring components.
    :return: Distinct component label strings.
    """
    labels: set[str] = set()
    for component in parse_components(value):
        if isinstance(component, dict) and component.get("label"):
            labels.add(str(component["label"]))
        elif hasattr(component, "label"):
            labels.add(str(component.label))
    return labels


def parse_components(value: Any) -> list[Any]:
    """Parse a component collection stored as a Python-style string.

    :param value: Component list or serialized representation.
    :return: Parsed list, or an empty list for missing or malformed values.
    """
    if value is None or value is pd.NA:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def components_have_hard_distinct_pair(
    left_members: set[str],
    right_members: set[str],
    hard_distinct_pairs: set[frozenset[str]],
) -> bool:
    """Determine whether merging two components violates a hard exclusion.

    :param left_members: Author IDs in the left cluster component.
    :param right_members: Author IDs in the right cluster component.
    :param hard_distinct_pairs: Pairs that may not share a cluster.
    :return: Whether any cross-component pair is hard-distinct.
    """
    return any(
        frozenset({left_id, right_id}) in hard_distinct_pairs
        for left_id, right_id in product(left_members, right_members)
    )


def cluster_author_frame(authors: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Cluster canonical author rows while preserving dataframe metadata.

    :param authors: Prepared author rows.
    :param decisions: Pairwise identity decisions.
    :return: Cluster membership rows with representative profile metadata.
    """
    parent = {author_id: author_id for author_id in authors[AUTHOR_ID].astype(str).tolist()}
    members_by_root = {author_id: {author_id} for author_id in parent}
    hard_distinct_pairs = hard_distinct_author_pairs(decisions)

    def find(author_id: str) -> str:
        while parent[author_id] != author_id:
            parent[author_id] = parent[parent[author_id]]
            author_id = parent[author_id]
        return author_id

    def union(left_id: str, right_id: str) -> None:
        if left_id not in parent or right_id not in parent:
            return
        left_root = find(left_id)
        right_root = find(right_id)
        if components_have_hard_distinct_pair(
            members_by_root[left_root],
            members_by_root[right_root],
            hard_distinct_pairs,
        ):
            return
        if left_root != right_root:
            parent[right_root] = left_root
            members_by_root[left_root].update(members_by_root.pop(right_root))

    for decision in decisions.itertuples(index=False):
        if getattr(decision, DECISION_LABEL) == "same_author":
            union(
                str(getattr(decision, LEFT_AUTHOR_ID)),
                str(getattr(decision, RIGHT_AUTHOR_ID)),
            )

    members_by_cluster: dict[str, list[str]] = defaultdict(list)
    for author_id in authors[AUTHOR_ID].astype(str).tolist():
        members_by_cluster[find(author_id)].append(author_id)

    authors_indexed = authors.copy()
    authors_indexed[_CLUSTER_AUTHOR_ID_COLUMN] = authors_indexed[AUTHOR_ID].astype(str)
    authors_indexed = authors_indexed.set_index(_CLUSTER_AUTHOR_ID_COLUMN, drop=True)
    cluster_rows: list[dict[str, object]] = []
    for index, member_ids in enumerate(
        sorted(members_by_cluster.values(), key=lambda ids: ids[0]), start=1
    ):
        members = authors_indexed.loc[member_ids].reset_index(drop=True).copy()
        document_counts = (
            members[DOCUMENT_COUNT]
            if DOCUMENT_COUNT in members.columns
            else pd.Series("", index=members.index)
        )
        publication_end = (
            members[PUBLICATION_END]
            if PUBLICATION_END in members.columns
            else pd.Series("", index=members.index)
        )
        publication_start = (
            members[PUBLICATION_START]
            if PUBLICATION_START in members.columns
            else pd.Series("", index=members.index)
        )
        members[_DOCUMENT_COUNT_NUMERIC_COLUMN] = pd.to_numeric(
            document_counts, errors="coerce"
        ).fillna(-1)
        members[_PUBLICATION_SPAN_NUMERIC_COLUMN] = (
            pd.to_numeric(publication_end, errors="coerce")
            - pd.to_numeric(publication_start, errors="coerce")
        ).fillna(-1)
        representative = members.sort_values(
            [_DOCUMENT_COUNT_NUMERIC_COLUMN, _PUBLICATION_SPAN_NUMERIC_COLUMN, AUTHOR_ID],
            ascending=[False, False, False],
        ).iloc[0]
        disambiguated_cluster_id = str(representative[AUTHOR_ID])
        members = members.sort_values([SURNAME, GIVEN_NAME, AUTHOR_ID])
        for _, member in members.iterrows():
            cluster_rows.append(
                {
                    CLUSTER_ID: index,
                    DISAMBIGUATED_CLUSTER_ID: disambiguated_cluster_id,
                    CLUSTER_SIZE: len(members),
                    ROW_ID: member.get(_SOURCE_ROW_ID_COLUMN, ""),
                    AUTHOR_ID: member[AUTHOR_ID],
                    NAME: member.get(NAME, ""),
                    CANONICAL_NAME: member.get(CANONICAL_NAME, ""),
                    SURNAME: member.get(SURNAME, ""),
                    CANONICAL_SURNAME: member.get(CANONICAL_SURNAME, ""),
                    GIVEN_NAME: member.get(GIVEN_NAME, ""),
                    CANONICAL_GIVEN_NAME: member.get(CANONICAL_GIVEN_NAME, ""),
                    INDEXED_NAME: member.get(INDEXED_NAME, ""),
                    CANONICAL_INDEXED_NAME: member.get(CANONICAL_INDEXED_NAME, ""),
                    CANONICAL_INITIALS: member.get(CANONICAL_INITIALS, ""),
                    ORCID: member.get(ORCID, ""),
                    PUBLICATION_START: member.get(PUBLICATION_START),
                    PUBLICATION_END: member.get(PUBLICATION_END),
                    SUBJECT_AREAS: member.get(SUBJECT_AREA_LABELS, []),
                }
            )
    return pd.DataFrame(cluster_rows)


def select_disambiguated_cluster_id(members: list[AuthorRecord]) -> str:
    """Select the profile ID representing a cluster using legacy tie-breaks.

    :param members: Author profiles assigned to one identity cluster.
    :return: The representative Scopus author ID.
    """
    representative = max(
        members,
        key=lambda author: (
            author.document_count if author.document_count is not None else -1,
            publication_span(author),
            author.author_id,
        ),
    )
    return representative.author_id


def publication_span(author: AuthorRecord) -> int:
    """Return the difference between a profile's parsed publication years.

    :param author: Author profile with parsed publication-range evidence.
    :return: Year difference, or ``-1`` when either bound is missing.
    """
    start = author.publication_range.start
    end = author.publication_range.end
    if start is None or end is None:
        return -1
    return end - start
