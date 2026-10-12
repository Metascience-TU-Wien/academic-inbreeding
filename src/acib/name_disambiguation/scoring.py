# Copyright © 2026 TU Wien

"""Top-level orchestration for canonical author-pair scoring."""

from __future__ import annotations

import itertools

import pandas as pd

from .constants import (
    COMPONENTS,
    CONFIDENCE,
    DECISION_LABEL,
    GIVEN_CONFLICT_INITIAL_OVERLAP_LABEL,
    INITIAL_ONLY_GIVEN_NAME_MATCH_LABEL,
    INITIAL_ONLY_LATER_GIVEN_INITIAL_LABEL,
    LEFT_AUTHOR_ID,
    LEFT_CANONICAL_GIVEN_NAME,
    LEFT_CANONICAL_INDEXED_NAME,
    LEFT_CANONICAL_INITIALS,
    LEFT_CANONICAL_NAME,
    LEFT_CANONICAL_SURNAME,
    LEFT_NAME,
    LEFT_ROW_ID,
    RAW_SCORE,
    RIGHT_AUTHOR_ID,
    RIGHT_CANONICAL_GIVEN_NAME,
    RIGHT_CANONICAL_INDEXED_NAME,
    RIGHT_CANONICAL_INITIALS,
    RIGHT_CANONICAL_NAME,
    RIGHT_CANONICAL_SURNAME,
    RIGHT_NAME,
    RIGHT_ROW_ID,
    STRONG_INITIAL_ONLY_ANCHOR_LABELS,
    TEMPORAL_CONFLICT_OVERRIDE_ANCHOR_LABELS,
    TEMPORAL_HARD_CONFLICT_LABELS,
    TRANSPOSED_NAME_FIELDS_LABEL,
)
from .models import AuthorRecord, PairwiseDecision, ScoreComponent
from .parsing import author_records_from_frame
from .scoring_common import SEVERE_CONFLICT_LABELS, sigmoid
from .scoring_context import (
    FirstNameFrequencyIndex,
    PublicationEvidenceIndex,
    ScoringContext,
    build_first_name_frequency_index,
    build_publication_evidence_index,
    build_scoring_context,
)
from .scoring_evidence import (
    score_author_linked_address,
    score_citation_relationships,
    score_current_affiliation_id,
    score_email_evidence,
    score_grant_numbers,
    score_journal_overlap,
    score_publication_address,
    score_publication_range_consistency,
    score_shared_coauthors,
    score_subject_areas,
)
from .scoring_name import AuthorNames, build_author_names, score_name_evidence


def score_all_pairs(
    authors: pd.DataFrame | list[AuthorRecord],
    article_author_df: pd.DataFrame | None = None,
    name_frequency_df: pd.DataFrame | None = None,
    context: ScoringContext | None = None,
    *,
    same_author_score_threshold: float = 6.0,
    possible_author_score_threshold: float = 3.0,
) -> pd.DataFrame:
    """Score every unique author pair and return a tabular decision report.

    :param authors: Prepared author rows or parsed author records.
    :param article_author_df: Optional article-author evidence table.
    :param name_frequency_df: Optional corpus for common-name scoring.
    :param context: Prebuilt scoring indices reused across batches.
    :param same_author_score_threshold: Minimum score for a same-author decision.
    :param possible_author_score_threshold: Minimum score for a possible-match decision.
    :return: One row per unique author pair.
    """
    if isinstance(authors, pd.DataFrame):
        context = context or build_scoring_context(authors, article_author_df, name_frequency_df)
        author_records = author_records_from_frame(authors)
        decisions = [
            score_pair(
                left,
                right,
                context=context,
                same_author_score_threshold=same_author_score_threshold,
                possible_author_score_threshold=possible_author_score_threshold,
            )
            for left, right in itertools.combinations(author_records, 2)
        ]
        return pd.DataFrame(decision_to_record(decision) for decision in decisions)

    author_records = ensure_author_records(authors)
    context = context or build_scoring_context(author_records, article_author_df, name_frequency_df)
    decisions = [
        score_pair(
            left,
            right,
            context=context,
            same_author_score_threshold=same_author_score_threshold,
            possible_author_score_threshold=possible_author_score_threshold,
        )
        for left, right in itertools.combinations(author_records, 2)
    ]
    return pd.DataFrame(decision_to_record(decision) for decision in decisions)


def score_pair(
    left: AuthorRecord,
    right: AuthorRecord,
    *,
    context: ScoringContext | None = None,
    same_author_score_threshold: float = 6.0,
    possible_author_score_threshold: float = 3.0,
) -> PairwiseDecision:
    """Score one author pair using Table 2 evidence plus name-based blockers.

    :param left: First author profile.
    :param right: Second author profile.
    :param context: Optional reusable publication and name-frequency indices.
    :param same_author_score_threshold: Minimum score for a same-author decision.
    :param possible_author_score_threshold: Minimum score for a possible-match decision.
    :return: Pairwise score, confidence, classification, and evidence components.
    """
    components: list[ScoreComponent] = []

    if left.orcid and right.orcid:
        if left.orcid == right.orcid:
            components.append(ScoreComponent("orcid_match", 100.0, "matching ORCID identifiers"))
        else:
            components.append(
                ScoreComponent("orcid_conflict", -100.0, "different ORCID identifiers")
            )

    left_names = build_author_names(left)
    right_names = build_author_names(right)

    components.extend(score_name_evidence(left_names, right_names, context=context))
    components.extend(score_email_evidence(left, right))
    components.extend(score_author_linked_address(left, right))
    components.extend(score_current_affiliation_id(left, right))
    components.extend(score_publication_range_consistency(left, right))
    components.extend(score_shared_coauthors(left, right, context=context))
    components.extend(score_grant_numbers(left, right, context=context))
    components.extend(score_publication_address(left, right, context=context))
    components.extend(score_subject_areas(left, right, context=context))
    components.extend(score_journal_overlap(left, right, context=context))

    if context and context.publication_index is not None:
        components.extend(score_citation_relationships(left, right, context.publication_index))

    raw_score = round(sum(component.weight for component in components), 3)
    confidence = round(sigmoid(raw_score / 4.0), 3)
    label = classify_pair(
        components,
        raw_score,
        same_author_score_threshold=same_author_score_threshold,
        possible_author_score_threshold=possible_author_score_threshold,
    )
    return PairwiseDecision(
        left=left,
        right=right,
        raw_score=raw_score,
        confidence=confidence,
        label=label,
        components=sorted(components, key=lambda item: abs(item.weight), reverse=True),
    )


def classify_pair(
    components: list[ScoreComponent],
    raw_score: float,
    *,
    same_author_score_threshold: float = 6.0,
    possible_author_score_threshold: float = 3.0,
) -> str:
    """Map a scored pair to the public decision labels used by clustering.

    :param components: Evidence components contributing to the pair score.
    :param raw_score: Combined score for the pair.
    :param same_author_score_threshold: Minimum score for a same-author decision.
    :param possible_author_score_threshold: Minimum score for a possible-match decision.
    :return: ``same_author``, ``possible_same_author``, or ``distinct``.
    """
    if any(
        component.label in SEVERE_CONFLICT_LABELS and component.weight < 0.0
        for component in components
    ):
        return "distinct"
    if has_hard_temporal_conflict(components) and not has_temporal_conflict_override_anchor(
        components
    ):
        return "distinct"
    if has_initial_only_later_given_initial_risk(components) and not has_strong_initial_only_anchor(
        components
    ):
        if raw_score >= possible_author_score_threshold:
            return "possible_same_author"
        return "distinct"
    if has_weak_name_bridge_risk(components) and not has_strong_weak_name_bridge_anchor(components):
        if raw_score >= possible_author_score_threshold:
            return "possible_same_author"
        return "distinct"
    if raw_score >= same_author_score_threshold:
        return "same_author"
    if raw_score >= possible_author_score_threshold:
        return "possible_same_author"
    return "distinct"


def has_initial_only_later_given_initial_risk(components: list[ScoreComponent]) -> bool:
    """Check whether a one-letter given name bridges to a longer name.

    :param components: Pairwise evidence components.
    :return: Whether the temporal initial-only risk label is present.
    """
    return any(
        component.label == INITIAL_ONLY_LATER_GIVEN_INITIAL_LABEL for component in components
    )


def has_weak_name_bridge_risk(components: list[ScoreComponent]) -> bool:
    """Check whether the pair relies on one of the weak name-bridge labels.

    :param components: Pairwise evidence components.
    :return: Whether a weak name-bridge label is present.
    """
    weak_labels = {
        INITIAL_ONLY_GIVEN_NAME_MATCH_LABEL,
        GIVEN_CONFLICT_INITIAL_OVERLAP_LABEL,
        TRANSPOSED_NAME_FIELDS_LABEL,
    }
    return any(component.label in weak_labels for component in components)


def has_strong_initial_only_anchor(components: list[ScoreComponent]) -> bool:
    """Check for evidence strong enough to anchor an initial-only name match.

    :param components: Pairwise evidence components.
    :return: Whether a configured strong anchor is present.
    """
    for component in components:
        if component.label in STRONG_INITIAL_ONLY_ANCHOR_LABELS:
            return True
        if component.label == "shared_coauthors" and component.weight >= 7.0:
            return True
    return False


def has_strong_weak_name_bridge_anchor(components: list[ScoreComponent]) -> bool:
    """Check for an anchor that overrides a weak name bridge.

    :param components: Pairwise evidence components.
    :return: Whether a strong anchor is present.
    """
    return has_strong_initial_only_anchor(components)


def has_hard_temporal_conflict(components: list[ScoreComponent]) -> bool:
    """Check whether a publication timeline has a hard conflict.

    :param components: Pairwise evidence components.
    :return: Whether any hard temporal-conflict label is present.
    """
    return any(component.label in TEMPORAL_HARD_CONFLICT_LABELS for component in components)


def has_temporal_conflict_override_anchor(components: list[ScoreComponent]) -> bool:
    """Check whether configured evidence can override a temporal conflict.

    :param components: Pairwise evidence components.
    :return: Whether a temporal override anchor is present.
    """
    return any(
        component.label in TEMPORAL_CONFLICT_OVERRIDE_ANCHOR_LABELS for component in components
    )


def decision_to_record(decision: PairwiseDecision) -> dict[str, object]:
    """Serialize a pairwise decision into a DataFrame-friendly record.

    :param decision: Pairwise result and evidence details.
    :return: Column-keyed record for the decision table.
    """
    return {
        LEFT_ROW_ID: decision.left.row_id,
        RIGHT_ROW_ID: decision.right.row_id,
        LEFT_AUTHOR_ID: decision.left.author_id,
        RIGHT_AUTHOR_ID: decision.right.author_id,
        LEFT_NAME: decision.left.name,
        RIGHT_NAME: decision.right.name,
        LEFT_CANONICAL_NAME: decision.left.canonical_name,
        RIGHT_CANONICAL_NAME: decision.right.canonical_name,
        LEFT_CANONICAL_SURNAME: decision.left.canonical_surname,
        RIGHT_CANONICAL_SURNAME: decision.right.canonical_surname,
        LEFT_CANONICAL_GIVEN_NAME: decision.left.canonical_given_name,
        RIGHT_CANONICAL_GIVEN_NAME: decision.right.canonical_given_name,
        LEFT_CANONICAL_INDEXED_NAME: decision.left.canonical_indexed_name,
        RIGHT_CANONICAL_INDEXED_NAME: decision.right.canonical_indexed_name,
        LEFT_CANONICAL_INITIALS: decision.left.canonical_initials,
        RIGHT_CANONICAL_INITIALS: decision.right.canonical_initials,
        RAW_SCORE: decision.raw_score,
        CONFIDENCE: decision.confidence,
        DECISION_LABEL: decision.label,
        COMPONENTS: [
            {
                "label": component.label,
                "weight": component.weight,
                "reason": component.reason,
            }
            for component in decision.components
        ],
    }


def ensure_author_records(authors: pd.DataFrame | list[AuthorRecord]) -> list[AuthorRecord]:
    """Accept either a prepared DataFrame or concrete author records.

    :param authors: Prepared author rows or their parsed records.
    :return: Parsed author records in source order.
    """
    if isinstance(authors, pd.DataFrame):
        return author_records_from_frame(authors)
    return authors


__all__ = [
    "AuthorNames",
    "FirstNameFrequencyIndex",
    "PublicationEvidenceIndex",
    "ScoringContext",
    "build_author_names",
    "build_first_name_frequency_index",
    "build_publication_evidence_index",
    "build_scoring_context",
    "classify_pair",
    "decision_to_record",
    "ensure_author_records",
    "score_all_pairs",
    "score_name_evidence",
    "score_pair",
]
