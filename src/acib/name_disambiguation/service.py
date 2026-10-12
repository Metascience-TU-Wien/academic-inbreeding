# Copyright © 2026 TU Wien

"""Business logic for batched disambiguation and final author assignment."""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pandas as pd
import structlog

from acib.common.names import given_tokens, split_name
from acib.constants import (
    AFFILIATION_ID_CURRENT,
    AFFILIATION_IDS_CURRENT,
    AUTHOR_ID as AUTHOR_SCOPUS_ID_COLUMN,
    EID as EID_COLUMN,
    INFERRED_ASJC_CODES,
    RESEARCHER_ID as RESEARCHER_ID_COLUMN_NAME,
    UNIVERSITY_NAME as UNIVERSITY_NAME_COLUMN,
)
from acib.name_disambiguation.constants import (
    ARTICLE_PARENT_SUBJECT_CODES,
    ARTICLE_PARENT_SUBJECT_COVERAGE,
    ARTICLE_PARENT_SUBJECT_OVERLAP_CODES,
    ARTICLE_PARENT_SUBJECT_OVERLAP_COUNT,
    AUTHOR_IDS,
    CANONICAL_GIVEN_NAME,
    CLUSTER_CURRENT_AFFILIATION_IDS,
    CLUSTER_GENDER,
    CLUSTER_GENDER_SOURCE,
    CLUSTER_GIVEN_NAME_TOKENS,
    CLUSTER_ID,
    CLUSTER_LEVEL_TWO_SUBJECT_CODES,
    CLUSTER_SIZE,
    CLUSTER_SUBJECT_CODES,
    CLUSTER_SUBJECT_COUNT,
    CURRENT_AFFILIATION_OVERLAP_COUNT,
    CURRENT_AFFILIATION_OVERLAP_IDS,
    DISAMBIGUATED_CLUSTER_ID,
    EXACT_INFERRED_COVERAGE,
    EXACT_OVERLAP_CODES,
    EXACT_OVERLAP_COUNT,
    EXACT_SPECIFIC_COVERAGE,
    EXACT_SPECIFIC_OVERLAP_CODES,
    EXACT_SPECIFIC_OVERLAP_COUNT,
    GENDER_COMPATIBLE,
    INFERRED_ASJC_LABELS,
    INFERRED_CODE_COVERAGE,
    INFERRED_CODE_OVERLAP_CODES,
    INFERRED_CODE_OVERLAP_COUNT,
    INFERRED_COVERAGE,
    OVERLAP_CODES,
    OVERLAP_COUNT,
    PROFILE_PARENT_SUBJECT_CODES,
    PROFILE_PARENT_SUBJECT_COVERAGE,
    PROFILE_PARENT_SUBJECT_OVERLAP_CODES,
    PROFILE_PARENT_SUBJECT_OVERLAP_COUNT,
    PUBLICATION_END,
    PUBLICATION_START,
    RESEARCHER_GENDER,
    RESEARCHER_GENDER_SOURCE,
    RESEARCHER_GIVEN_NAME_COVERAGE,
    RESEARCHER_GIVEN_NAME_OVERLAP_COUNT,
    RESEARCHER_GIVEN_NAME_OVERLAP_TOKENS,
    RESEARCHER_GIVEN_NAME_TOKENS,
    SUBJECT_AREAS_PARSED,
    TEMPORAL_HARD_SPAN_YEARS,
    UNIVERSITY_AFFILIATION_IDS,
)
from acib.name_matching.constants import RESEARCHER_NAME, SCOPUS_ID_MAPPING

from .asjc import is_level_two_asjc_code, level_two_asjc_code, level_two_asjc_codes
from .clustering import cluster_authors
from .data import prepare_disambiguation_author_frame
from .gender import (
    GenderEvidence,
    combine_gender_evidence,
    genders_conflict,
    infer_gender_from_full_name,
    infer_gender_from_given_name,
)
from .models import BatchDisambiguationResult, FinalAuthorSelection
from .scoring import score_all_pairs
from .scoring_common import normalize_person_name, parse_collection_field
from .scoring_context import build_scoring_context
from .scoring_evidence import normalize_current_affiliation_id

_logger = structlog.getLogger(__name__)
_WORKER_CONTEXT: Any = None
_INFERRED_ASJC_PARENT_CODES_COLUMN = "inferred_asjc_parent_codes"
_INFERRED_ASJC_PARENT_CODES_JSON_COLUMN = "inferred_asjc_parent_codes_json"


def initialize_disambiguation_worker(context: Any) -> None:
    """Set the process-local scoring context for batch workers.

    :param context: Shared publication and first-name scoring context.
    """
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def disambiguate_batch_worker(
    payload: tuple[str, pd.DataFrame, float, float],
) -> BatchDisambiguationResult:
    """Run one process-pool payload using the initialized scoring context.

    :param payload: Researcher ID, author rows, and pair-score thresholds.
    :return: Results for the supplied researcher batch.
    """
    sc_id, authors, same_author_score_threshold, possible_author_score_threshold = payload
    return disambiguate_batch(
        sc_id=sc_id,
        authors=authors,
        context=_WORKER_CONTEXT,
        same_author_score_threshold=same_author_score_threshold,
        possible_author_score_threshold=possible_author_score_threshold,
    )


def disambiguate_batches(
    author_df: pd.DataFrame,
    *,
    article_author_df: pd.DataFrame | None = None,
    article_reference_df: pd.DataFrame | None = None,
    name_frequency_df: pd.DataFrame | None = None,
    same_author_score_threshold: float = 6.0,
    possible_author_score_threshold: float = 3.0,
    max_workers: int = 1,
) -> list[BatchDisambiguationResult]:
    """Run pair scoring, clustering, and selection for each researcher batch.

    :param author_df: Matched author candidate rows.
    :param article_author_df: Optional publication-author evidence table.
    :param article_reference_df: Optional citation-edge table.
    :param name_frequency_df: Optional broad author corpus.
    :param same_author_score_threshold: Minimum score for a same-author decision.
    :param possible_author_score_threshold: Minimum score for a possible-match decision.
    :param max_workers: Maximum parallel worker processes.
    :return: One result per researcher batch.
    :raises ValueError: If no author candidates are supplied.
    """
    author_df = prepare_disambiguation_author_frame(author_df)
    _logger.info("Prepared author candidates", author_rows=len(author_df))

    if author_df.empty:
        raise ValueError("No author candidates were found in the configured author file.")

    _logger.info(
        "Building disambiguation scoring context",
        article_author_rows=0 if article_author_df is None else len(article_author_df),
        article_reference_rows=0 if article_reference_df is None else len(article_reference_df),
        name_frequency_rows=0 if name_frequency_df is None else len(name_frequency_df),
    )
    shared_context = build_scoring_context(
        authors=author_df,
        article_author_df=article_author_df,
        article_reference_df=article_reference_df,
        name_frequency_df=name_frequency_df,
    )
    _logger.info("Built disambiguation scoring context")

    if RESEARCHER_ID_COLUMN_NAME not in author_df.columns:
        _logger.info(
            "Disambiguating one author batch without researcher ID grouping",
            author_rows=len(author_df),
        )
        return [disambiguate_batch(sc_id="", authors=author_df, context=shared_context)]

    results: list[BatchDisambiguationResult] = []
    group_sizes = author_df.groupby(RESEARCHER_ID_COLUMN_NAME)[EID_COLUMN].transform("size")
    ambiguous_authors = author_df.loc[group_sizes.gt(1)].reset_index(drop=True)
    singleton_authors = author_df.loc[group_sizes.eq(1)].reset_index(drop=True)
    _logger.info(
        "Partitioned author candidates",
        ambiguous_batch_count=ambiguous_authors[RESEARCHER_ID_COLUMN_NAME].nunique(),
        ambiguous_author_rows=len(ambiguous_authors),
        singleton_batch_count=singleton_authors[RESEARCHER_ID_COLUMN_NAME].nunique(),
        singleton_author_rows=len(singleton_authors),
    )

    max_workers = max(1, int(max_workers))
    _logger.info(
        "Processing ambiguous disambiguation batches",
        batch_count=ambiguous_authors[RESEARCHER_ID_COLUMN_NAME].nunique(),
        max_workers=max_workers,
    )
    if max_workers == 1:
        for sc_id, batch in ambiguous_authors.groupby(RESEARCHER_ID_COLUMN_NAME, sort=True):
            results.append(
                disambiguate_batch(
                    sc_id=str(sc_id),
                    authors=batch.reset_index(drop=True),
                    context=shared_context,
                    same_author_score_threshold=same_author_score_threshold,
                    possible_author_score_threshold=possible_author_score_threshold,
                )
            )
    else:
        ambiguous_payloads = [
            (
                str(sc_id),
                batch.reset_index(drop=True),
                same_author_score_threshold,
                possible_author_score_threshold,
            )
            for sc_id, batch in ambiguous_authors.groupby(RESEARCHER_ID_COLUMN_NAME, sort=True)
        ]
        if len(ambiguous_payloads) <= 1:
            for sc_id, batch, same_threshold, possible_threshold in ambiguous_payloads:
                results.append(
                    disambiguate_batch(
                        sc_id=sc_id,
                        authors=batch,
                        context=shared_context,
                        same_author_score_threshold=same_threshold,
                        possible_author_score_threshold=possible_threshold,
                    )
                )
            ambiguous_payloads = []
        if ambiguous_payloads:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=initialize_disambiguation_worker,
                initargs=(shared_context,),
            ) as executor:
                results.extend(executor.map(disambiguate_batch_worker, ambiguous_payloads))

    _logger.info(
        "Processing single-candidate disambiguation batches",
        batch_count=singleton_authors[RESEARCHER_ID_COLUMN_NAME].nunique(),
    )
    for sc_id, batch in singleton_authors.groupby(RESEARCHER_ID_COLUMN_NAME, sort=True):
        results.append(
            single_candidate_batch(sc_id=str(sc_id), authors=batch.reset_index(drop=True))
        )
    _logger.info("Finished disambiguation batches", batch_count=len(results))
    return results


def single_candidate_batch(sc_id: str, authors: pd.DataFrame) -> BatchDisambiguationResult:
    """Create a final selection for a researcher with one candidate author.

    :param sc_id: Researcher identifier.
    :param authors: Prepared candidate frame containing exactly one row.
    :return: Single-candidate selection and cluster result.
    """
    clusters = cluster_authors(authors=authors, decisions=pd.DataFrame())
    if not clusters.empty:
        clusters = clusters.copy()
        clusters[RESEARCHER_ID_COLUMN_NAME] = sc_id

    author_id = str(authors[AUTHOR_SCOPUS_ID_COLUMN].iloc[0])
    final_selection = FinalAuthorSelection(
        sc_id=sc_id,
        status="assigned_single_candidate",
        selected_cluster_id=1,
        selected_disambiguated_cluster_id=author_id,
        selected_author_ids=[author_id],
        possible_clusters=[[author_id]],
        inferred_asjc_codes=sorted(extract_inferred_asjc_codes(authors=authors)),
        scopus_asjc_codes=sorted(extract_scopus_asjc_codes(authors=authors)),
        inferred_asjc_labels=sorted(extract_inferred_asjc_labels(authors=authors)),
        reason="only one candidate author exists for this sc_id; disambiguation not required",
    )
    final_selection = apply_publication_span_quality_control(final_selection, clusters)
    return BatchDisambiguationResult(
        sc_id=sc_id,
        authors=authors,
        decisions=pd.DataFrame(),
        clusters=clusters,
        final_selection=final_selection,
        final_selection_scores=pd.DataFrame(),
    )


def disambiguate_batch(
    sc_id: str,
    authors: pd.DataFrame,
    context: Any,
    *,
    same_author_score_threshold: float = 6.0,
    possible_author_score_threshold: float = 3.0,
) -> BatchDisambiguationResult:
    """Disambiguate one candidate batch and select a linked author when possible.

    :param sc_id: Researcher identifier for this batch.
    :param authors: Prepared author candidate rows.
    :param context: Shared publication and name-frequency evidence.
    :param same_author_score_threshold: Minimum score for merging a pair.
    :param possible_author_score_threshold: Minimum score for a possible match.
    :return: Pair decisions, clusters, and final-selection evidence.
    """
    decisions = score_all_pairs(
        authors,
        context=context,
        same_author_score_threshold=same_author_score_threshold,
        possible_author_score_threshold=possible_author_score_threshold,
    )
    clusters = cluster_authors(authors=authors, decisions=decisions)
    if not clusters.empty:
        clusters = clusters.copy()
        clusters[RESEARCHER_ID_COLUMN_NAME] = sc_id

    final_selection, selection_scores = select_final_author(
        authors=authors,
        clusters=clusters,
        sc_id=sc_id,
        context=context,
    )
    final_selection = apply_publication_span_quality_control(final_selection, clusters)
    return BatchDisambiguationResult(
        sc_id=sc_id,
        authors=authors,
        decisions=decisions,
        clusters=clusters,
        final_selection=final_selection,
        final_selection_scores=selection_scores,
    )


def apply_publication_span_quality_control(
    selection: FinalAuthorSelection,
    clusters: pd.DataFrame,
) -> FinalAuthorSelection:
    """Suppress automatic selections with implausibly long publication histories.

    :param selection: Current final author selection.
    :param clusters: Cluster rows used to inspect the selected publication span.
    :return: Original selection or a manual-review selection.
    """
    if selection.selected_cluster_id is None or selection.status == "manual_selection_required":
        return selection
    if clusters.empty or CLUSTER_ID not in clusters:
        return selection

    cluster = clusters[clusters[CLUSTER_ID].astype(str).eq(str(selection.selected_cluster_id))]
    if cluster.empty:
        return selection
    span = publication_span(cluster)
    if span is None or span <= TEMPORAL_HARD_SPAN_YEARS:
        return selection

    start = int(pd.to_numeric(cluster[PUBLICATION_START], errors="coerce").min())
    end = int(pd.to_numeric(cluster[PUBLICATION_END], errors="coerce").max())
    return FinalAuthorSelection(
        sc_id=selection.sc_id,
        status="manual_selection_required",
        selected_cluster_id=None,
        selected_disambiguated_cluster_id="",
        selected_author_ids=[],
        possible_clusters=selection.possible_clusters or [selection.selected_author_ids],
        inferred_asjc_codes=selection.inferred_asjc_codes,
        scopus_asjc_codes=selection.scopus_asjc_codes,
        inferred_asjc_labels=selection.inferred_asjc_labels,
        reason=(
            "selected cluster has an implausible publication span "
            f"of {span} years ({start}-{end}); automatic selection suppressed for manual review. "
            f"Original reason: {selection.reason}"
        ),
    )


def publication_span(cluster: pd.DataFrame) -> int | None:
    """Calculate the span covered by cluster publication-year bounds.

    :param cluster: Cluster membership rows.
    :return: Difference between latest end and earliest start, or ``None``.
    """
    if PUBLICATION_START not in cluster or PUBLICATION_END not in cluster:
        return None
    starts = pd.to_numeric(cluster[PUBLICATION_START], errors="coerce").dropna()
    ends = pd.to_numeric(cluster[PUBLICATION_END], errors="coerce").dropna()
    if starts.empty or ends.empty:
        return None
    return int(ends.max() - starts.min())


def select_final_author(
    authors: pd.DataFrame,
    clusters: pd.DataFrame,
    sc_id: str,
    context: Any = None,
) -> tuple[FinalAuthorSelection, pd.DataFrame]:
    """Choose a final cluster using evidence, or request manual selection.

    :param authors: Prepared author rows for one researcher.
    :param clusters: Cluster memberships for those authors.
    :param sc_id: Researcher identifier.
    :param context: Optional shared scoring context.
    :return: Final selection and the per-cluster score table.
    """
    inferred_codes = sorted(extract_inferred_asjc_codes(authors=authors))
    inferred_labels = sorted(extract_inferred_asjc_labels(authors=authors))
    researcher_gender = infer_researcher_gender(authors=authors)

    if clusters.empty:
        return (
            FinalAuthorSelection(
                sc_id=sc_id,
                status="manual_selection_required",
                selected_cluster_id=None,
                selected_disambiguated_cluster_id="",
                selected_author_ids=[],
                possible_clusters=[],
                inferred_asjc_codes=inferred_codes,
                scopus_asjc_codes=sorted(extract_scopus_asjc_codes(authors=authors)),
                inferred_asjc_labels=inferred_labels,
                reason="no clusters available",
            ),
            pd.DataFrame(),
        )

    selection_scores = build_final_selection_scores(
        authors=authors,
        clusters=clusters,
        inferred_codes=inferred_codes,
        researcher_gender=researcher_gender,
        context=context,
    )
    cluster_count = clusters[CLUSTER_ID].nunique()
    selectable_scores = selection_scores[selection_scores[GENDER_COMPATIBLE]].reset_index(drop=True)

    if selectable_scores.empty:
        return (
            FinalAuthorSelection(
                sc_id=sc_id,
                status="manual_selection_required",
                selected_cluster_id=None,
                selected_disambiguated_cluster_id="",
                selected_author_ids=[],
                possible_clusters=selection_scores[AUTHOR_IDS].tolist(),
                inferred_asjc_codes=inferred_codes,
                scopus_asjc_codes=cluster_subject_codes_by_cluster(selection_scores),
                inferred_asjc_labels=inferred_labels,
                reason="no cluster has compatible high-confidence researcher gender evidence",
            ),
            selection_scores,
        )

    if cluster_count == 1:
        row = selectable_scores.iloc[0]
        return (
            FinalAuthorSelection(
                sc_id=sc_id,
                status="assigned_single_cluster",
                selected_cluster_id=int(row[CLUSTER_ID]),
                selected_disambiguated_cluster_id=str(row[DISAMBIGUATED_CLUSTER_ID]),
                selected_author_ids=row[AUTHOR_IDS],
                possible_clusters=[row[AUTHOR_IDS]],
                inferred_asjc_codes=inferred_codes,
                scopus_asjc_codes=row[CLUSTER_SUBJECT_CODES],
                inferred_asjc_labels=inferred_labels,
                reason="only one cluster remained after disambiguation",
            ),
            selection_scores,
        )

    current_affiliation_selection = select_by_current_affiliation(
        selectable_scores=selectable_scores,
        selection_scores=selection_scores,
        sc_id=sc_id,
        inferred_codes=inferred_codes,
        inferred_labels=inferred_labels,
    )
    if current_affiliation_selection is not None:
        return current_affiliation_selection, selection_scores

    if not inferred_codes:
        if researcher_gender.reliable and len(selectable_scores) == 1:
            row = selectable_scores.iloc[0]
            return (
                FinalAuthorSelection(
                    sc_id=sc_id,
                    status="assigned_by_gender",
                    selected_cluster_id=int(row[CLUSTER_ID]),
                    selected_disambiguated_cluster_id=str(row[DISAMBIGUATED_CLUSTER_ID]),
                    selected_author_ids=row[AUTHOR_IDS],
                    possible_clusters=selection_scores[AUTHOR_IDS].tolist(),
                    inferred_asjc_codes=inferred_codes,
                    scopus_asjc_codes=row[CLUSTER_SUBJECT_CODES],
                    inferred_asjc_labels=inferred_labels,
                    reason="only one cluster remained after reliable researcher-gender filtering",
                ),
                selection_scores,
            )
        researcher_given_name_selection = select_by_researcher_given_name(
            selectable_scores=selectable_scores,
            selection_scores=selection_scores,
            sc_id=sc_id,
            inferred_codes=inferred_codes,
            inferred_labels=inferred_labels,
        )
        if researcher_given_name_selection is not None:
            return researcher_given_name_selection, selection_scores
        return (
            FinalAuthorSelection(
                sc_id=sc_id,
                status="manual_selection_required",
                selected_cluster_id=None,
                selected_disambiguated_cluster_id="",
                selected_author_ids=[],
                possible_clusters=selection_scores[AUTHOR_IDS].tolist(),
                inferred_asjc_codes=inferred_codes,
                scopus_asjc_codes=cluster_subject_codes_by_cluster(selection_scores),
                inferred_asjc_labels=inferred_labels,
                reason="multiple clusters remain and no inferred ASJC codes are available",
            ),
            selection_scores,
        )

    ranked = selectable_scores.sort_values(
        [
            ARTICLE_PARENT_SUBJECT_OVERLAP_COUNT,
            ARTICLE_PARENT_SUBJECT_COVERAGE,
            INFERRED_CODE_OVERLAP_COUNT,
            INFERRED_CODE_COVERAGE,
            EXACT_SPECIFIC_OVERLAP_COUNT,
            EXACT_SPECIFIC_COVERAGE,
            OVERLAP_COUNT,
            INFERRED_COVERAGE,
            CLUSTER_SUBJECT_COUNT,
            CLUSTER_SIZE,
        ],
        ascending=[False, False, False, False, False, False, False, False, True, False],
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    second = ranked.iloc[1] if len(ranked) > 1 else None

    best_core_tuple = (
        best[ARTICLE_PARENT_SUBJECT_OVERLAP_COUNT],
        best[ARTICLE_PARENT_SUBJECT_COVERAGE],
        best[INFERRED_CODE_OVERLAP_COUNT],
        best[INFERRED_CODE_COVERAGE],
        best[EXACT_SPECIFIC_OVERLAP_COUNT],
        best[EXACT_SPECIFIC_COVERAGE],
        best[OVERLAP_COUNT],
        best[INFERRED_COVERAGE],
    )
    second_core_tuple = (
        (
            second[ARTICLE_PARENT_SUBJECT_OVERLAP_COUNT],
            second[ARTICLE_PARENT_SUBJECT_COVERAGE],
            second[INFERRED_CODE_OVERLAP_COUNT],
            second[INFERRED_CODE_COVERAGE],
            second[EXACT_SPECIFIC_OVERLAP_COUNT],
            second[EXACT_SPECIFIC_COVERAGE],
            second[OVERLAP_COUNT],
            second[INFERRED_COVERAGE],
        )
        if second is not None
        else (-1, -1.0, -1, -1.0, -1, -1.0, -1, -1.0)
    )
    best_is_unique_by_asjc = best_core_tuple > second_core_tuple
    has_decisive_asjc_overlap = (
        best[ARTICLE_PARENT_SUBJECT_OVERLAP_COUNT] > 0
        or best[INFERRED_CODE_OVERLAP_COUNT] > 0
        or best[OVERLAP_COUNT] > 0
    )
    if has_decisive_asjc_overlap and best_is_unique_by_asjc:
        status = asjc_assignment_status(best)
        return (
            FinalAuthorSelection(
                sc_id=sc_id,
                status=status,
                selected_cluster_id=int(best[CLUSTER_ID]),
                selected_disambiguated_cluster_id=str(best[DISAMBIGUATED_CLUSTER_ID]),
                selected_author_ids=best[AUTHOR_IDS],
                possible_clusters=ranked[AUTHOR_IDS].tolist(),
                inferred_asjc_codes=inferred_codes,
                scopus_asjc_codes=best[CLUSTER_SUBJECT_CODES],
                inferred_asjc_labels=inferred_labels,
                reason=f"best cluster has a unique strongest {status.replace('_', ' ')} overlap",
            ),
            selection_scores,
        )

    researcher_given_name_selection = select_by_researcher_given_name(
        selectable_scores=selectable_scores,
        selection_scores=selection_scores,
        sc_id=sc_id,
        inferred_codes=inferred_codes,
        inferred_labels=inferred_labels,
    )
    if researcher_given_name_selection is not None:
        return researcher_given_name_selection, selection_scores

    return (
        FinalAuthorSelection(
            sc_id=sc_id,
            status="manual_selection_required",
            selected_cluster_id=None,
            selected_disambiguated_cluster_id="",
            selected_author_ids=[],
            possible_clusters=selection_scores[AUTHOR_IDS].tolist(),
            inferred_asjc_codes=inferred_codes,
            scopus_asjc_codes=cluster_subject_codes_by_cluster(selection_scores),
            inferred_asjc_labels=inferred_labels,
            reason="multiple clusters remain and inferred-ASJC evidence is not decisive",
        ),
        selection_scores,
    )


def select_by_current_affiliation(
    *,
    selectable_scores: pd.DataFrame,
    selection_scores: pd.DataFrame,
    sc_id: str,
    inferred_codes: list[int],
    inferred_labels: list[str],
) -> FinalAuthorSelection | None:
    """Select a unique cluster matching the researcher's current institution.

    :param selectable_scores: Scores restricted to gender-compatible clusters.
    :param selection_scores: Scores for all candidate clusters.
    :param sc_id: Researcher identifier.
    :param inferred_codes: Inferred subject codes for the researcher.
    :param inferred_labels: Human-readable inferred subject labels.
    :return: Assignment when one cluster uniquely matches, otherwise ``None``.
    """
    if CURRENT_AFFILIATION_OVERLAP_COUNT not in selectable_scores.columns:
        return None

    ranked = selectable_scores.sort_values(
        [
            CURRENT_AFFILIATION_OVERLAP_COUNT,
            INFERRED_CODE_OVERLAP_COUNT,
            INFERRED_CODE_COVERAGE,
            OVERLAP_COUNT,
            INFERRED_COVERAGE,
            CLUSTER_SIZE,
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    second = ranked.iloc[1] if len(ranked) > 1 else None
    best_overlap_count = int(best[CURRENT_AFFILIATION_OVERLAP_COUNT])
    second_overlap_count = (
        int(second[CURRENT_AFFILIATION_OVERLAP_COUNT]) if second is not None else 0
    )

    if best_overlap_count <= 0 or best_overlap_count <= second_overlap_count:
        return None

    return FinalAuthorSelection(
        sc_id=sc_id,
        status="assigned_by_current_affiliation",
        selected_cluster_id=int(best[CLUSTER_ID]),
        selected_disambiguated_cluster_id=str(best[DISAMBIGUATED_CLUSTER_ID]),
        selected_author_ids=best[AUTHOR_IDS],
        possible_clusters=selection_scores[AUTHOR_IDS].tolist(),
        inferred_asjc_codes=inferred_codes,
        scopus_asjc_codes=best[CLUSTER_SUBJECT_CODES],
        inferred_asjc_labels=inferred_labels,
        reason="best cluster has a unique current-affiliation match to the researcher university",
    )


def select_by_researcher_given_name(
    *,
    selectable_scores: pd.DataFrame,
    selection_scores: pd.DataFrame,
    sc_id: str,
    inferred_codes: list[int],
    inferred_labels: list[str],
) -> FinalAuthorSelection | None:
    """Select a unique cluster matching the researcher's full given-name tokens.

    :param selectable_scores: Scores restricted to gender-compatible clusters.
    :param selection_scores: Scores for all candidate clusters.
    :param sc_id: Researcher identifier.
    :param inferred_codes: Inferred subject codes for the researcher.
    :param inferred_labels: Human-readable inferred subject labels.
    :return: Assignment when one cluster uniquely matches, otherwise ``None``.
    """
    required_columns = {
        RESEARCHER_GIVEN_NAME_OVERLAP_COUNT,
        RESEARCHER_GIVEN_NAME_COVERAGE,
    }
    if not required_columns.issubset(selectable_scores.columns):
        return None

    ranked = selectable_scores.sort_values(
        [
            RESEARCHER_GIVEN_NAME_OVERLAP_COUNT,
            RESEARCHER_GIVEN_NAME_COVERAGE,
            CLUSTER_SIZE,
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    second = ranked.iloc[1] if len(ranked) > 1 else None
    best_tuple = (
        int(best[RESEARCHER_GIVEN_NAME_OVERLAP_COUNT]),
        float(best[RESEARCHER_GIVEN_NAME_COVERAGE]),
    )
    second_tuple = (
        (
            int(second[RESEARCHER_GIVEN_NAME_OVERLAP_COUNT]),
            float(second[RESEARCHER_GIVEN_NAME_COVERAGE]),
        )
        if second is not None
        else (-1, -1.0)
    )
    if best_tuple[0] <= 0 or best_tuple <= second_tuple:
        return None

    return FinalAuthorSelection(
        sc_id=sc_id,
        status="assigned_by_researcher_given_name",
        selected_cluster_id=int(best[CLUSTER_ID]),
        selected_disambiguated_cluster_id=str(best[DISAMBIGUATED_CLUSTER_ID]),
        selected_author_ids=best[AUTHOR_IDS],
        possible_clusters=selection_scores[AUTHOR_IDS].tolist(),
        inferred_asjc_codes=inferred_codes,
        scopus_asjc_codes=best[CLUSTER_SUBJECT_CODES],
        inferred_asjc_labels=inferred_labels,
        reason="best cluster has a unique full given-name match to the researcher name",
    )


def asjc_assignment_status(row: pd.Series) -> str:
    """Choose the assignment status matching available article/profile overlap.

    :param row: Candidate-cluster score row.
    :return: Combined, article-based, or profile-based assignment status.
    """
    article_overlap = int(row.get(ARTICLE_PARENT_SUBJECT_OVERLAP_COUNT, 0))
    profile_overlap = int(row.get(PROFILE_PARENT_SUBJECT_OVERLAP_COUNT, 0))
    if article_overlap > 0 and profile_overlap > 0:
        return "assigned_by_combined_asjc"
    if article_overlap > 0:
        return "assigned_by_article_asjc"
    return "assigned_by_profile_asjc"


def build_final_selection_scores(
    authors: pd.DataFrame,
    clusters: pd.DataFrame,
    inferred_codes: list[int],
    *,
    researcher_gender: GenderEvidence | None = None,
    context: Any = None,
) -> pd.DataFrame:
    """Summarize cluster-level evidence against inferred ASJC codes.

    :param authors: Prepared author candidates for one researcher.
    :param clusters: Cluster membership rows.
    :param inferred_codes: Inferred ASJC codes for the researcher.
    :param researcher_gender: Optional precomputed researcher gender evidence.
    :param context: Optional publication evidence context.
    :return: One evidence-score row per candidate cluster.
    """
    inferred_specific_code_set = {
        code
        for code in inferred_codes
        if level_two_asjc_code(code) is not None and not is_level_two_asjc_code(code)
    }
    inferred_parent_code_set = extract_inferred_parent_asjc_codes(authors=authors)
    if not inferred_parent_code_set:
        inferred_parent_code_set = level_two_asjc_codes(codes=inferred_codes)
    researcher_gender = researcher_gender or infer_researcher_gender(authors=authors)
    university_name = extract_researcher_university_name(authors=authors)
    university_affiliation_ids = scopus_affiliation_ids_for_university(university_name)
    researcher_given_name_tokens = extract_researcher_given_name_tokens(authors=authors)
    authors = authors.copy()
    if CANONICAL_GIVEN_NAME not in authors.columns:
        authors[CANONICAL_GIVEN_NAME] = ""
    author_subjects = (
        authors[[AUTHOR_SCOPUS_ID_COLUMN, SUBJECT_AREAS_PARSED, CANONICAL_GIVEN_NAME]]
        .assign(
            **{AUTHOR_SCOPUS_ID_COLUMN: lambda frame: frame[AUTHOR_SCOPUS_ID_COLUMN].astype(str)}
        )
        .drop_duplicates(subset=[AUTHOR_SCOPUS_ID_COLUMN])
    )
    author_current_affiliations = current_affiliation_ids_by_author(authors=authors)

    rows: list[dict[str, object]] = []
    for cluster_id, members in clusters.groupby(CLUSTER_ID, sort=True):
        member_ids = members[AUTHOR_SCOPUS_ID_COLUMN].astype(str).tolist()
        article_parent_subject_codes = article_parent_subject_codes_for_members(
            member_ids=member_ids,
            context=context,
        )
        cluster_current_affiliation_ids = sorted(
            {
                affiliation_id
                for member_id in member_ids
                for affiliation_id in author_current_affiliations.get(member_id, set())
            }
        )
        current_affiliation_overlap_ids = sorted(
            set(cluster_current_affiliation_ids) & set(university_affiliation_ids)
        )
        matched_subjects = author_subjects[
            author_subjects[AUTHOR_SCOPUS_ID_COLUMN].isin(member_ids)
        ]
        cluster_gender = combine_gender_evidence(
            infer_gender_from_given_name(given_name)
            for given_name in matched_subjects[CANONICAL_GIVEN_NAME].tolist()
        )
        cluster_given_name_tokens = sorted(
            {
                token
                for given_name in matched_subjects[CANONICAL_GIVEN_NAME].dropna().astype(str)
                for token in full_given_name_tokens(given_name)
            }
        )
        researcher_given_name_overlap_tokens = sorted(
            set(researcher_given_name_tokens) & set(cluster_given_name_tokens)
        )
        researcher_given_name_coverage = (
            len(researcher_given_name_overlap_tokens) / len(researcher_given_name_tokens)
            if researcher_given_name_tokens
            else 0.0
        )
        profile_subject_codes: set[int] = set()
        for areas in matched_subjects[SUBJECT_AREAS_PARSED].tolist():
            profile_subject_codes.update(
                int(area.code) for area in areas if getattr(area, "code", None) not in {None, 0}
            )
        profile_parent_subject_codes = level_two_asjc_codes(codes=profile_subject_codes)
        combined_parent_subject_codes = article_parent_subject_codes | profile_parent_subject_codes

        article_overlap_codes = sorted(inferred_parent_code_set & article_parent_subject_codes)
        profile_overlap_codes = sorted(inferred_parent_code_set & profile_parent_subject_codes)
        article_overlap_count = len(article_overlap_codes)
        profile_overlap_count = len(profile_overlap_codes)
        article_coverage = (
            article_overlap_count / len(inferred_parent_code_set)
            if inferred_parent_code_set
            else 0.0
        )
        profile_coverage = (
            profile_overlap_count / len(inferred_parent_code_set)
            if inferred_parent_code_set
            else 0.0
        )
        inferred_code_overlap_codes = sorted(
            code
            for code in inferred_codes
            if asjc_code_matches_cluster(code, profile_subject_codes)
        )
        inferred_code_overlap_count = len(inferred_code_overlap_codes)
        inferred_code_coverage = (
            inferred_code_overlap_count / len(inferred_codes) if inferred_codes else 0.0
        )
        exact_specific_overlap_codes = sorted(inferred_specific_code_set & profile_subject_codes)
        exact_specific_overlap_count = len(exact_specific_overlap_codes)
        exact_specific_coverage = (
            exact_specific_overlap_count / len(inferred_specific_code_set)
            if inferred_specific_code_set
            else 0.0
        )
        rows.append(
            {
                CLUSTER_ID: int(cluster_id),
                DISAMBIGUATED_CLUSTER_ID: members[DISAMBIGUATED_CLUSTER_ID].iloc[0],
                CLUSTER_SIZE: int(members[CLUSTER_SIZE].iloc[0]),
                AUTHOR_IDS: member_ids,
                UNIVERSITY_NAME_COLUMN: university_name,
                UNIVERSITY_AFFILIATION_IDS: university_affiliation_ids,
                CLUSTER_CURRENT_AFFILIATION_IDS: cluster_current_affiliation_ids,
                CURRENT_AFFILIATION_OVERLAP_IDS: current_affiliation_overlap_ids,
                CURRENT_AFFILIATION_OVERLAP_COUNT: len(current_affiliation_overlap_ids),
                RESEARCHER_GIVEN_NAME_TOKENS: researcher_given_name_tokens,
                CLUSTER_GIVEN_NAME_TOKENS: cluster_given_name_tokens,
                RESEARCHER_GIVEN_NAME_OVERLAP_TOKENS: researcher_given_name_overlap_tokens,
                RESEARCHER_GIVEN_NAME_OVERLAP_COUNT: len(researcher_given_name_overlap_tokens),
                RESEARCHER_GIVEN_NAME_COVERAGE: round(researcher_given_name_coverage, 3),
                ARTICLE_PARENT_SUBJECT_CODES: sorted(article_parent_subject_codes),
                ARTICLE_PARENT_SUBJECT_OVERLAP_CODES: article_overlap_codes,
                ARTICLE_PARENT_SUBJECT_OVERLAP_COUNT: article_overlap_count,
                ARTICLE_PARENT_SUBJECT_COVERAGE: round(article_coverage, 3),
                PROFILE_PARENT_SUBJECT_CODES: sorted(profile_parent_subject_codes),
                PROFILE_PARENT_SUBJECT_OVERLAP_CODES: profile_overlap_codes,
                PROFILE_PARENT_SUBJECT_OVERLAP_COUNT: profile_overlap_count,
                PROFILE_PARENT_SUBJECT_COVERAGE: round(profile_coverage, 3),
                CLUSTER_SUBJECT_CODES: sorted(combined_parent_subject_codes),
                CLUSTER_SUBJECT_COUNT: len(combined_parent_subject_codes),
                CLUSTER_LEVEL_TWO_SUBJECT_CODES: sorted(combined_parent_subject_codes),
                EXACT_OVERLAP_CODES: inferred_code_overlap_codes,
                EXACT_OVERLAP_COUNT: inferred_code_overlap_count,
                EXACT_INFERRED_COVERAGE: round(inferred_code_coverage, 3),
                EXACT_SPECIFIC_OVERLAP_CODES: exact_specific_overlap_codes,
                EXACT_SPECIFIC_OVERLAP_COUNT: exact_specific_overlap_count,
                EXACT_SPECIFIC_COVERAGE: round(exact_specific_coverage, 3),
                INFERRED_CODE_OVERLAP_CODES: inferred_code_overlap_codes,
                INFERRED_CODE_OVERLAP_COUNT: inferred_code_overlap_count,
                INFERRED_CODE_COVERAGE: round(inferred_code_coverage, 3),
                OVERLAP_CODES: profile_overlap_codes,
                OVERLAP_COUNT: profile_overlap_count,
                INFERRED_COVERAGE: round(profile_coverage, 3),
                RESEARCHER_GENDER: researcher_gender.gender,
                RESEARCHER_GENDER_SOURCE: researcher_gender.source_name,
                CLUSTER_GENDER: cluster_gender.gender,
                CLUSTER_GENDER_SOURCE: cluster_gender.source_name,
                GENDER_COMPATIBLE: not genders_conflict(researcher_gender, cluster_gender),
            }
        )
    return pd.DataFrame(rows)


def extract_researcher_given_name_tokens(*, authors: pd.DataFrame) -> list[str]:
    """Collect normalized, fully spelled tokens from researcher names.

    :param authors: Candidate rows carrying the researcher-name field.
    :return: Sorted distinct given-name tokens.
    """
    if RESEARCHER_NAME not in authors.columns:
        return []

    tokens: set[str] = set()
    names = authors[RESEARCHER_NAME].dropna().astype(str).unique()
    for name in names:
        _, given_name = split_name(name)
        if given_name:
            tokens.update(full_given_name_tokens(given_name))
    return sorted(tokens)


def full_given_name_tokens(value: str) -> set[str]:
    """Return normalized given-name tokens longer than initials.

    :param value: Given-name text.
    :return: Unique full given-name tokens.
    """
    return {token for token in given_tokens(normalize_person_name(value)) if len(token) > 1}


def cluster_subject_codes_by_cluster(selection_scores: pd.DataFrame) -> list[list[int]]:
    """Return sorted subject-code lists for each scored cluster.

    :param selection_scores: Final-selection score table.
    :return: One normalized subject-code list per row.
    """
    if CLUSTER_SUBJECT_CODES not in selection_scores.columns:
        return []
    return [
        sorted(int(code) for code in cluster_codes if code not in {None, 0})
        for cluster_codes in selection_scores[CLUSTER_SUBJECT_CODES].tolist()
    ]


def asjc_code_matches_cluster(inferred_code: int, cluster_codes: set[int]) -> bool:
    """Return whether a cluster covers one inferred ASJC code hierarchically.

    :param inferred_code: Inferred child or parent ASJC code.
    :param cluster_codes: Codes represented by a candidate cluster.
    :return: Whether the cluster covers the inferred code.
    """
    inferred_parent = level_two_asjc_code(inferred_code)
    if inferred_parent is None:
        return inferred_code in cluster_codes
    if is_level_two_asjc_code(inferred_code):
        return any(
            level_two_asjc_code(cluster_code) == inferred_code for cluster_code in cluster_codes
        )
    for cluster_code in cluster_codes:
        if cluster_code == inferred_code:
            return True
        if cluster_code == inferred_parent:
            return True
    return False


def infer_researcher_gender(authors: pd.DataFrame) -> GenderEvidence:
    """Infer combined gender evidence from distinct researcher-name values.

    :param authors: Candidate rows carrying researcher-name metadata.
    :return: Combined gender evidence.
    """
    if RESEARCHER_NAME not in authors.columns:
        return combine_gender_evidence([])
    return combine_gender_evidence(
        infer_gender_from_full_name(name)
        for name in authors[RESEARCHER_NAME].dropna().astype(str).unique()
    )


def extract_researcher_university_name(authors: pd.DataFrame) -> str:
    """Return the first nonblank university name in a candidate batch.

    :param authors: Candidate author rows with researcher metadata.
    :return: University name or empty text.
    """
    if UNIVERSITY_NAME_COLUMN not in authors.columns:
        return ""
    values = authors[UNIVERSITY_NAME_COLUMN].dropna().astype(str).str.strip()
    values = values[values.ne("") & ~values.str.lower().isin({"nan", "none", "null"})]
    return values.iloc[0] if not values.empty else ""


def scopus_affiliation_ids_for_university(university_name: str) -> list[str]:
    """Look up normalized Scopus affiliation IDs for a university name.

    :param university_name: Researcher university label.
    :return: Sorted matching affiliation ID strings.
    """
    if not university_name:
        return []

    mapped_ids = SCOPUS_ID_MAPPING.get(university_name)
    if mapped_ids is None:
        normalized_university_name = university_name.casefold()
        mapped_ids = next(
            (
                ids
                for mapped_university_name, ids in SCOPUS_ID_MAPPING.items()
                if mapped_university_name.casefold() == normalized_university_name
            ),
            [],
        )

    normalized_ids = {
        normalized_id
        for affiliation_id in mapped_ids
        if (normalized_id := normalize_current_affiliation_id(affiliation_id))
    }
    return sorted(normalized_ids)


def current_affiliation_ids_by_author(authors: pd.DataFrame) -> dict[str, set[str]]:
    """Collect normalized current-affiliation IDs for each author profile.

    :param authors: Candidate author rows with current-affiliation fields.
    :return: Affiliation ID sets keyed by Scopus author ID.
    """
    if (
        AFFILIATION_IDS_CURRENT not in authors.columns
        and AFFILIATION_ID_CURRENT not in authors.columns
    ):
        return {}

    result: dict[str, set[str]] = {}
    for row in authors.itertuples(index=False):
        author_id = str(getattr(row, AUTHOR_SCOPUS_ID_COLUMN))
        raw_values = parse_collection_field(getattr(row, AFFILIATION_IDS_CURRENT, None))
        if not raw_values:
            raw_values = parse_collection_field(getattr(row, AFFILIATION_ID_CURRENT, None))
        affiliation_ids = {
            normalized_id
            for value in raw_values
            if (normalized_id := normalize_current_affiliation_id(value))
        }
        if affiliation_ids:
            result.setdefault(author_id, set()).update(affiliation_ids)
    return result


def article_parent_subject_codes_for_members(
    *,
    member_ids: list[str],
    context: Any = None,
) -> set[int]:
    """Collect article-derived parent subject codes for cluster members.

    :param member_ids: Scopus author IDs in the cluster.
    :param context: Optional publication evidence context.
    :return: Unique parent subject codes.
    """
    if context is None or getattr(context, "publication_index", None) is None:
        return set()
    subject_index = getattr(context.publication_index, "author_article_subject_codes", {})
    return {
        int(code)
        for member_id in member_ids
        for code in subject_index.get(str(member_id), set())
        if code not in {None, 0}
    }


def extract_inferred_asjc_codes(authors: pd.DataFrame) -> set[int]:
    """Parse inferred ASJC code values from enrichment columns.

    :param authors: Candidate rows carrying inferred-code fields.
    :return: Unique inferred ASJC codes.
    """
    if INFERRED_ASJC_CODES not in authors.columns:
        return set()
    codes: set[int] = set()
    for value in authors[INFERRED_ASJC_CODES].dropna().astype(str):
        for match in re.findall(r"\d+", value):
            codes.add(int(match))
    return codes


def extract_inferred_parent_asjc_codes(authors: pd.DataFrame) -> set[int]:
    """Prefer explicit parent codes, falling back to collapsed child codes.

    :param authors: Candidate rows with inferred ASJC fields.
    :return: Inferred level-two parent codes.
    """
    for column in (
        _INFERRED_ASJC_PARENT_CODES_COLUMN,
        _INFERRED_ASJC_PARENT_CODES_JSON_COLUMN,
    ):
        if column not in authors.columns:
            continue
        codes: set[int] = set()
        for value in authors[column].dropna().astype(str):
            for match in re.findall(r"\d+", value):
                parent = level_two_asjc_code(int(match))
                if parent is not None:
                    codes.add(parent)
        if codes:
            return codes
    return level_two_asjc_codes(codes=extract_inferred_asjc_codes(authors=authors))


def extract_scopus_asjc_codes(authors: pd.DataFrame) -> set[int]:
    """Collect ASJC codes from parsed Scopus author subject areas.

    :param authors: Prepared author rows.
    :return: Unique nonzero subject-area codes.
    """
    if SUBJECT_AREAS_PARSED not in authors.columns:
        return set()
    codes: set[int] = set()
    for areas in authors[SUBJECT_AREAS_PARSED].tolist():
        for area in areas:
            code = getattr(area, "code", None)
            if code is not None and code != 0:
                codes.add(int(code))
    return codes


def extract_inferred_asjc_labels(authors: pd.DataFrame) -> set[str]:
    """Collect inferred ASJC labels for reports and manual review.

    :param authors: Candidate rows carrying inferred-label fields.
    :return: Unique nonblank labels.
    """
    if INFERRED_ASJC_LABELS not in authors.columns:
        return set()
    labels: set[str] = set()
    for value in authors[INFERRED_ASJC_LABELS].dropna().astype(str):
        for label in re.split(r"[;|,]", value):
            cleaned = label.strip()
            if cleaned:
                labels.add(cleaned)
    return labels
