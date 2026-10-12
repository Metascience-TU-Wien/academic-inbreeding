# Copyright © 2026 TU Wien
"""Render detailed text and JSON reports for disambiguation batches."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pandas as pd

from acib.constants import AUTHOR_ID, SUBJECT_AREAS, UNIVERSITY_NAME
from acib.name_disambiguation.asjc import describe_codes, level_two_asjc_codes
from acib.name_disambiguation.constants import (
    AUTHOR_IDS,
    CANONICAL_INDEXED_NAME,
    CANONICAL_INITIALS,
    CANONICAL_NAME,
    CLUSTER_CURRENT_AFFILIATION_IDS,
    CLUSTER_GENDER,
    CLUSTER_GENDER_SOURCE,
    CLUSTER_ID,
    CLUSTER_LEVEL_TWO_SUBJECT_CODES,
    CLUSTER_SIZE,
    CLUSTER_SUBJECT_CODES,
    COMPONENTS,
    CONFIDENCE,
    CURRENT_AFFILIATION_OVERLAP_COUNT,
    CURRENT_AFFILIATION_OVERLAP_IDS,
    DECISION_LABEL,
    DISAMBIGUATED_CLUSTER_ID,
    GENDER_COMPATIBLE,
    INFERRED_COVERAGE,
    LEFT_AUTHOR_ID,
    LEFT_CANONICAL_NAME,
    LEFT_NAME,
    ORCID,
    OVERLAP_CODES,
    OVERLAP_COUNT,
    PUBLICATION_END,
    PUBLICATION_START,
    RAW_SCORE,
    RESEARCHER_GENDER,
    RESEARCHER_GENDER_SOURCE,
    RIGHT_AUTHOR_ID,
    RIGHT_CANONICAL_NAME,
    RIGHT_NAME,
    UNIVERSITY_AFFILIATION_IDS,
)
from acib.name_disambiguation.models import FinalAuthorSelection
from acib.name_matching.constants import INDEXED_NAME, NAME

_SELECTED_COLUMN = "selected"

_TABLE2_COMPONENTS: list[tuple[str, str | tuple[str, ...], str]] = [
    ("email_exact", "email_exact", "no exact email match"),
    (
        "shared_initials",
        ("shared_initials_2", "shared_initials_gt2", "initials_conflict"),
        "no scored initials evidence",
    ),
    (
        "shared_first_name",
        ("shared_first_name_general", "shared_first_name_nongeneral"),
        "no scored shared first-name evidence",
    ),
    ("author_address_match", "author_address_match", "no author-linked city/country match"),
    ("shared_coauthors", "shared_coauthors", "no scored shared coauthor evidence"),
    ("grant_number_match", "grant_number_match", "no shared grant number"),
    (
        "publication_address_match",
        "publication_address_match",
        "no publication-linked city/country match",
    ),
    (
        "article_subject_overlap",
        "article_subject_overlap",
        "no article parent subject-category overlap",
    ),
    (
        "profile_subject_overlap",
        "profile_subject_overlap",
        "no profile parent subject-category overlap",
    ),
    ("journal_match", "journal_match", "no matching journal"),
    ("self_citation", "self_citation", "no self-citation relation found"),
    ("bibliographic_coupling", "bibliographic_coupling", "no shared cited references"),
    ("co_citation", "co_citation", "no shared citing papers"),
]


def render_text_report(
    decisions: pd.DataFrame,
    clusters: pd.DataFrame,
    final_selection: FinalAuthorSelection | None = None,
    final_selection_scores: pd.DataFrame | None = None,
) -> str:
    """Render the legacy human-readable pair, cluster, and selection report.

    :param decisions: Scored pairwise decisions for one researcher.
    :param clusters: Cluster membership rows for the same researcher.
    :param final_selection: Optional final-selection decision.
    :param final_selection_scores: Optional candidate-cluster evidence table.
    :return: Complete report text ending in a newline.
    """
    lines = ["Author Name Disambiguation", "=" * 28, ""]
    if final_selection is not None and final_selection.sc_id:
        lines.extend([f"sc_id: {final_selection.sc_id}", ""])
    lines.extend(
        [
            f"Pair comparisons: {len(decisions)}",
            f"Identity clusters: {clusters[CLUSTER_ID].nunique()}",
            "",
            "Pairwise evidence",
            "-" * 17,
        ]
    )
    for decision in decisions.itertuples(index=False):
        lines.extend(
            [
                (
                    f"[{getattr(decision, DECISION_LABEL)}] {getattr(decision, LEFT_NAME)} "
                    f"({getattr(decision, LEFT_AUTHOR_ID)}) "
                    f"<-> {getattr(decision, RIGHT_NAME)} ({getattr(decision, RIGHT_AUTHOR_ID)})"
                ),
                (
                    f"  processed_names={getattr(decision, LEFT_CANONICAL_NAME)} "
                    f"<-> {getattr(decision, RIGHT_CANONICAL_NAME)}"
                ),
                (
                    f"  raw_score={getattr(decision, RAW_SCORE):.3f} "
                    f"confidence={getattr(decision, CONFIDENCE):.3f}"
                ),
            ]
        )
        for component in summarize_table2_components(getattr(decision, COMPONENTS)):
            lines.append(
                f"  {component['weight']:+.2f} {component['label']}: {component['reason']}"
            )
        extras = remaining_components(getattr(decision, COMPONENTS))
        if extras:
            lines.append("  additional_scoring:")
            lines.extend(f"  {c['weight']:+.2f} {c['label']}: {c['reason']}" for c in extras)
        lines.append("")
    lines.extend(["Clusters", "-" * 8])
    selected_ids = set(final_selection.selected_author_ids) if final_selection else set()
    manual = final_selection is not None and final_selection.status == "manual_selection_required"
    for cluster_id, members in clusters.groupby(CLUSTER_ID, sort=True):
        disambiguated = members[DISAMBIGUATED_CLUSTER_ID].iloc[0]
        selected = any(author_id in selected_ids for author_id in members[AUTHOR_ID].astype(str))
        lines.append(
            f"Cluster {cluster_id}: {len(members)} record(s) | "
            f"disambiguated_cluster_id={disambiguated} | "
            f"selected={selection_flag(selected, manual)}"
        )
        for member in members.itertuples(index=False):
            years = format_years(
                getattr(member, PUBLICATION_START), getattr(member, PUBLICATION_END)
            )
            areas = ", ".join(getattr(member, SUBJECT_AREAS)) or "n/a"
            lines.append(
                f"  - {getattr(member, NAME)} | author_id={getattr(member, AUTHOR_ID)} | "
                f"selected={selection_flag(str(getattr(member, AUTHOR_ID)) in selected_ids,
                                           manual)} | "
                f"indexed={getattr(member, INDEXED_NAME) or 'n/a'} | years={years} | areas={areas}"
            )
        lines.append("")
    if final_selection is not None:
        lines.extend(["Final Selection", "-" * 15, f"status={final_selection.status}"])
        if final_selection.selected_disambiguated_cluster_id:
            lines.extend(
                [
                    (
                        "selected_disambiguated_cluster_id="
                        f"{final_selection.selected_disambiguated_cluster_id}"
                    ),
                    f"selected_author_ids={final_selection.selected_author_ids}",
                ]
            )
        if final_selection.inferred_asjc_codes:
            level_two_codes = sorted(level_two_asjc_codes(final_selection.inferred_asjc_codes))
            lines.extend(
                [
                    f"inferred_asjc_codes={final_selection.inferred_asjc_codes}",
                    (
                        "target_subject_areas="
                        f"{format_asjc_codes(final_selection.inferred_asjc_codes)}"
                    ),
                    ("target_level2_subject_areas=" f"{format_asjc_codes(level_two_codes)}"),
                ]
            )
        if final_selection.inferred_asjc_labels:
            lines.append(f"inferred_asjc_labels={final_selection.inferred_asjc_labels}")
        lines.append(f"reason={final_selection.reason}")
        if final_selection_scores is not None and not final_selection_scores.empty:
            lines.append("candidate_clusters:")
            sort_cols = [
                c
                for c in [
                    CURRENT_AFFILIATION_OVERLAP_COUNT,
                    OVERLAP_COUNT,
                    INFERRED_COVERAGE,
                    CLUSTER_SIZE,
                ]
                if c in final_selection_scores
            ]
            scores = final_selection_scores.sort_values(
                sort_cols, ascending=[False] * len(sort_cols)
            )
            for row in scores.itertuples(index=False):
                lines.append(
                    f"  cluster_id={getattr(row, CLUSTER_ID)} "
                    f"disambiguated_cluster_id={getattr(row, DISAMBIGUATED_CLUSTER_ID)} "
                    f"overlap_count={getattr(row, OVERLAP_COUNT)} "
                    f"inferred_coverage={getattr(row, INFERRED_COVERAGE):.3f} "
                    f"author_ids={getattr(row, AUTHOR_IDS)} "
                    f"overlap_codes={getattr(row, OVERLAP_CODES)}"
                )
                if hasattr(row, GENDER_COMPATIBLE):
                    lines.append(f"    gender_compatible={getattr(row, GENDER_COMPATIBLE)}")
                if hasattr(row, CURRENT_AFFILIATION_OVERLAP_IDS):
                    lines.append(
                        f"    current_affiliation=university={getattr(row, UNIVERSITY_NAME) or
                                                              'n/a'}, "
                        f"mapped_ids={getattr(row, UNIVERSITY_AFFILIATION_IDS)}, "
                        f"cluster_ids={getattr(row, CLUSTER_CURRENT_AFFILIATION_IDS)}, "
                        f"overlap_ids={getattr(row, CURRENT_AFFILIATION_OVERLAP_IDS)}"
                    )
                if hasattr(row, RESEARCHER_GENDER):
                    lines.append(
                        f"    gender=researcher:{getattr(row, RESEARCHER_GENDER)}"
                        f"({getattr(row, RESEARCHER_GENDER_SOURCE) or 'n/a'}), "
                        f"cluster:{getattr(row, CLUSTER_GENDER)}"
                        f"({getattr(row, CLUSTER_GENDER_SOURCE) or 'n/a'})"
                    )
                lines.append(
                    f"    cluster_subject_areas={format_asjc_codes(getattr(row,
                                                                           CLUSTER_SUBJECT_CODES))}"
                )
                if hasattr(row, CLUSTER_LEVEL_TWO_SUBJECT_CODES):
                    lines.append(
                        "    cluster_level2_subject_areas="
                        f"{format_asjc_codes(getattr(row, CLUSTER_LEVEL_TWO_SUBJECT_CODES))}"
                    )
                lines.append(
                    f"    overlap_subject_areas={format_asjc_codes(getattr(row, OVERLAP_CODES))}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_json_report(
    decisions: pd.DataFrame,
    clusters: pd.DataFrame,
    final_selection: FinalAuthorSelection | None = None,
    final_selection_scores: pd.DataFrame | None = None,
) -> str:
    """Render one legacy-format JSON report as an indented JSON string.

    :param decisions: Scored pairwise decisions for one researcher.
    :param clusters: Cluster membership rows for the same researcher.
    :param final_selection: Optional final-selection decision.
    :param final_selection_scores: Optional candidate-cluster evidence table.
    :return: Indented JSON text ending in a newline.
    """
    selected = set(final_selection.selected_author_ids) if final_selection else set()
    manual = final_selection is not None and final_selection.status == "manual_selection_required"
    cluster_rows = []
    member_columns = [
        DISAMBIGUATED_CLUSTER_ID,
        AUTHOR_ID,
        NAME,
        CANONICAL_NAME,
        INDEXED_NAME,
        CANONICAL_INDEXED_NAME,
        CANONICAL_INITIALS,
        ORCID,
    ]
    for cluster_id, members in clusters.groupby(CLUSTER_ID, sort=True):
        member_data = members[member_columns].copy()
        member_data[_SELECTED_COLUMN] = (
            member_data[AUTHOR_ID]
            .astype(str)
            .map(lambda author_id: selection_flag(author_id in selected, manual))
        )
        cluster_rows.append(
            {
                CLUSTER_ID: cluster_id,
                DISAMBIGUATED_CLUSTER_ID: members[DISAMBIGUATED_CLUSTER_ID].iloc[0],
                _SELECTED_COLUMN: selection_flag(
                    any(a in selected for a in members[AUTHOR_ID].astype(str)), manual
                ),
                "members": member_data.to_dict(orient="records"),
            }
        )
    payload = {
        "pairs": decisions.to_dict(orient="records"),
        "clusters": cluster_rows,
        "final_selection": asdict(final_selection) if final_selection is not None else None,
        "final_selection_scores": (
            final_selection_scores.to_dict(orient="records")
            if final_selection_scores is not None
            else []
        ),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def format_years(start: int | None, end: int | None) -> str:
    """Format inclusive year bounds as a range or ``n/a``.

    :param start: First publication year.
    :param end: Last publication year.
    :return: One year, an inclusive range, or ``n/a``.
    """
    if start is None:
        return "n/a"
    return str(start) if start == end else f"{start}-{end}"


def selection_flag(selected: bool, manual_selection_required: bool) -> str:
    """Format selected state using the legacy yes/manual/no labels.

    :param selected: Whether the author or cluster was selected.
    :param manual_selection_required: Whether a nonselected candidate needs review.
    :return: Legacy selection label.
    """
    return "yes" if selected else "manual" if manual_selection_required else "no"


def format_asjc_codes(codes: list[int] | set[int]) -> str:
    """Format sorted ASJC codes with taxonomy descriptions.

    :param codes: ASJC codes to describe.
    :return: Bracketed, taxonomy-labeled code list.
    """
    if not codes:
        return "[]"
    ordered = sorted(int(code) for code in codes)
    return f"[{describe_codes(set(ordered))}]"


def summarize_table2_components(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the fixed legacy evidence summary, adding zero-evidence reasons.

    :param components: Scoring components for one author pair.
    :return: Ordered evidence rows for the human-readable report.
    """
    by_label = {str(component["label"]): component for component in components}
    summary = []
    for label, sources, reason in _TABLE2_COMPONENTS:
        labels = (sources,) if isinstance(sources, str) else sources
        summary.append(
            next(
                (by_label[item] for item in labels if item in by_label),
                {"label": label, "weight": 0.0, "reason": reason},
            )
        )
    return summary


def remaining_components(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return scored components not represented in the fixed evidence summary.

    :param components: Scoring components for one author pair.
    :return: Components omitted from the fixed evidence summary.
    """
    shown = {
        source
        for _, sources, _ in _TABLE2_COMPONENTS
        for source in ((sources,) if isinstance(sources, str) else sources)
    }
    return [component for component in components if str(component["label"]) not in shown]
