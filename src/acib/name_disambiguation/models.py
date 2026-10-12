# Copyright © 2026 TU Wien

"""Data models used by author name disambiguation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class NameVariant:
    """A source-provided author-name variant and its canonical form.

    :param indexed_name: Source-indexed name string.
    :param initials: Source initials.
    :param surname: Source surname.
    :param given_name: Source given name.
    :param canonical_surname: Normalized surname.
    :param canonical_given_name: Normalized given name.
    :param canonical_full_name: Normalized surname-first name.
    :param canonical_indexed_name: Normalized indexed name.
    :param canonical_initials: Normalized initials.
    :param doc_count: Optional source document count.
    """

    indexed_name: str = ""
    initials: str = ""
    surname: str = ""
    given_name: str = ""
    canonical_surname: str = ""
    canonical_given_name: str = ""
    canonical_full_name: str = ""
    canonical_indexed_name: str = ""
    canonical_initials: str = ""
    doc_count: int | None = None


@dataclass(slots=True)
class SubjectArea:
    """A Scopus subject-area label and optional numeric code.

    :param area: Full subject-area label.
    :param abbreviation: Source abbreviation.
    :param code: Optional numeric ASJC code.
    """

    area: str
    abbreviation: str = ""
    code: int | None = None


@dataclass(slots=True)
class PublicationRange:
    """Inclusive start and end publication years.

    :param start: Earliest publication year.
    :param end: Latest publication year.
    """

    start: int | None = None
    end: int | None = None


@dataclass(slots=True)
class AuthorRecord:
    """Canonical author profile and its original metadata.

    :param row_id: Source row identifier.
    :param author_id: Scopus author identifier.
    :param name: Source display name.
    :param surname: Source surname.
    :param given_name: Source given name.
    :param indexed_name: Scopus indexed name.
    :param initials: Source initials.
    :param canonical_name: Normalized full name.
    :param canonical_surname: Normalized surname.
    :param canonical_given_name: Normalized given name.
    :param canonical_indexed_name: Normalized indexed name.
    :param canonical_initials: Normalized initials.
    :param orcid: ORCID identifier.
    :param document_count: Optional publication count.
    :param citation_count: Optional citation count.
    :param cited_by_count: Optional cited-by count.
    :param h_index: Optional h-index.
    :param coauthor_count: Optional coauthor count.
    :param publication_range: Parsed publication-year range.
    :param name_variants: Source name variants.
    :param subject_areas: Parsed Scopus subject areas.
    :param parent_id: Source parent identifier.
    :param historical_identifier: Historical Scopus IDs.
    :param raw: String-valued original source fields.
    """

    row_id: str
    author_id: str
    name: str
    surname: str
    given_name: str
    indexed_name: str
    initials: str
    canonical_name: str
    canonical_surname: str
    canonical_given_name: str
    canonical_indexed_name: str
    canonical_initials: str
    orcid: str
    document_count: int | None
    citation_count: int | None
    cited_by_count: int | None
    h_index: int | None
    coauthor_count: int | None
    publication_range: PublicationRange
    name_variants: list[NameVariant] = field(default_factory=list)
    subject_areas: list[SubjectArea] = field(default_factory=list)
    parent_id: str = ""
    historical_identifier: str = ""
    raw: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreComponent:
    """An individual contribution to a pairwise score.

    :param label: Evidence component label.
    :param weight: Score contribution.
    :param reason: Human-readable evidence explanation.
    """

    label: str
    weight: float
    reason: str


@dataclass(slots=True)
class PairwiseDecision:
    """Scored relationship between two author profiles.

    :param left: First compared author.
    :param right: Second compared author.
    :param raw_score: Combined pair score.
    :param confidence: Logistic confidence value.
    :param label: Pair classification.
    :param components: Scoring evidence components.
    """

    left: AuthorRecord
    right: AuthorRecord
    raw_score: float
    confidence: float
    label: str
    components: list[ScoreComponent]


@dataclass(slots=True)
class Cluster:
    """A cluster of author profiles believed to represent one person.

    :param cluster_id: Researcher-local cluster number.
    :param members: Author profiles assigned to the cluster.
    """

    cluster_id: int
    members: list[AuthorRecord]


@dataclass(slots=True)
class FinalAuthorSelection:
    """Final profile selection and subject-area evidence for one researcher.

    :param sc_id: Researcher identifier.
    :param status: Selection status label.
    :param selected_cluster_id: Selected cluster, if assigned.
    :param selected_disambiguated_cluster_id: Representative author ID.
    :param selected_author_ids: Selected Scopus author IDs.
    :param possible_clusters: Candidate author-ID groups considered.
    :param inferred_asjc_codes: Researcher-side inferred subject codes.
    :param scopus_asjc_codes: Candidate Scopus subject codes.
    :param inferred_asjc_labels: Human-readable inferred subject labels.
    :param reason: Selection explanation.
    """

    sc_id: str
    status: str
    selected_cluster_id: int | None
    selected_disambiguated_cluster_id: str
    selected_author_ids: list[str]
    possible_clusters: list[list[str]]
    inferred_asjc_codes: list[int]
    scopus_asjc_codes: list[int] | list[list[int]]
    inferred_asjc_labels: list[str]
    reason: str


@dataclass(slots=True)
class BatchDisambiguationResult:
    """DataFrames and final selection produced for one researcher batch.

    :param sc_id: Researcher identifier.
    :param authors: Prepared author candidate rows.
    :param decisions: Pairwise decision table.
    :param clusters: Cluster membership table.
    :param final_selection: Final researcher selection.
    :param final_selection_scores: Candidate-cluster score table.
    """

    sc_id: str
    authors: pd.DataFrame
    decisions: pd.DataFrame
    clusters: pd.DataFrame
    final_selection: FinalAuthorSelection
    final_selection_scores: pd.DataFrame
