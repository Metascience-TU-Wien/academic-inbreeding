# Copyright © 2026 TU Wien

"""Stage-specific columns and labels used by name disambiguation."""

from __future__ import annotations

# Publication-evidence input columns.
AUTHOR_IDS = "author_ids"
CITING_EID = "citing_eid"
CITED_EID = "cited_eid"
ARTICLE_AFFILIATION_CITY = "affiliation_city"
ARTICLE_AFFILIATION_COUNTRY = "affiliation_country"
ARTICLE_CITY = "city"
ARTICLE_COUNTRY = "country"
ARTICLE_COUNTRY_CODE = "country_code"
ARTICLE_AUTHOR_NAMES = "author_names"
ARTICLE_PUBLICATION_CITY = "publication_city"
ARTICLE_PUBLICATION_COUNTRY = "publication_country"
ISO3_COUNTRY_CODE = "iso3_code"
ARTICLE_JOURNAL = "journal"
ARTICLE_JOURNALS = "journals"
ARTICLE_SOURCE_TITLE = "source_title"
ARTICLE_SOURCE_TITLES = "source_titles"
ARTICLE_SOURCE_TITLE_SHORT = "srctitle"
ARTICLE_PUBLICATION_NAME = "publication_name"
ARTICLE_GRANT_NUMBER = "grant_number"
ARTICLE_GRANT_NUMBERS = "grant_numbers"
ARTICLE_GRANT_ID = "grant_id"
ARTICLE_FUND_NUMBER = "fund_no"

# Prepared author-frame fields.
CANONICAL_NAME = "canonical_name"
CANONICAL_SURNAME = "canonical_surname"
CANONICAL_GIVEN_NAME = "canonical_given_name"
CANONICAL_INDEXED_NAME = "canonical_indexed_name"
CANONICAL_INITIALS = "canonical_initials"
PUBLICATION_START = "publication_start"
PUBLICATION_END = "publication_end"
NAME_VARIANTS_PARSED = "name_variants_parsed"
SUBJECT_AREAS_PARSED = "subject_areas_parsed"
SUBJECT_AREA_LABELS = "subject_area_labels"

# Pair-decision output columns.
LEFT_ROW_ID = "left_row_id"
RIGHT_ROW_ID = "right_row_id"
ROW_ID = "row_id"
LEFT_AUTHOR_ID = "left_author_id"
RIGHT_AUTHOR_ID = "right_author_id"
LEFT_NAME = "left_name"
RIGHT_NAME = "right_name"
LEFT_CANONICAL_NAME = "left_canonical_name"
RIGHT_CANONICAL_NAME = "right_canonical_name"
LEFT_CANONICAL_SURNAME = "left_canonical_surname"
RIGHT_CANONICAL_SURNAME = "right_canonical_surname"
LEFT_CANONICAL_GIVEN_NAME = "left_canonical_given_name"
RIGHT_CANONICAL_GIVEN_NAME = "right_canonical_given_name"
LEFT_CANONICAL_INDEXED_NAME = "left_canonical_indexed_name"
RIGHT_CANONICAL_INDEXED_NAME = "right_canonical_indexed_name"
LEFT_CANONICAL_INITIALS = "left_canonical_initials"
RIGHT_CANONICAL_INITIALS = "right_canonical_initials"
RAW_SCORE = "raw_score"
CONFIDENCE = "confidence"
DECISION_LABEL = "label"
COMPONENTS = "components"

# Cluster, researcher-selection, and score-table columns.
CLUSTER_ID = "cluster_id"
DISAMBIGUATED_CLUSTER_ID = "disambiguated_cluster_id"
CLUSTER_SIZE = "cluster_size"
ORCID = "orcid"
DOCUMENT_COUNT = "document_count"
AUTHOR_HISTORY = "history"
SELECTION_STATUS = "status"
SELECTED_CLUSTER_ID = "selected_cluster_id"
SELECTED_DISAMBIGUATED_CLUSTER_ID = "selected_disambiguated_cluster_id"
SELECTED_AUTHOR_IDS = "selected_author_ids"
POSSIBLE_CLUSTERS = "possible_clusters"
SELECTION_REASON = "reason"
UNIVERSITY_AFFILIATION_IDS = "university_affiliation_ids"
CLUSTER_CURRENT_AFFILIATION_IDS = "cluster_current_affiliation_ids"
CURRENT_AFFILIATION_OVERLAP_IDS = "current_affiliation_overlap_ids"
CURRENT_AFFILIATION_OVERLAP_COUNT = "current_affiliation_overlap_count"
RESEARCHER_GIVEN_NAME_TOKENS = "researcher_given_name_tokens"
CLUSTER_GIVEN_NAME_TOKENS = "cluster_given_name_tokens"
RESEARCHER_GIVEN_NAME_OVERLAP_TOKENS = "researcher_given_name_overlap_tokens"
RESEARCHER_GIVEN_NAME_OVERLAP_COUNT = "researcher_given_name_overlap_count"
RESEARCHER_GIVEN_NAME_COVERAGE = "researcher_given_name_coverage"
ARTICLE_PARENT_SUBJECT_CODES = "article_parent_subject_codes"
ARTICLE_PARENT_SUBJECT_OVERLAP_CODES = "article_parent_subject_overlap_codes"
ARTICLE_PARENT_SUBJECT_OVERLAP_COUNT = "article_parent_subject_overlap_count"
ARTICLE_PARENT_SUBJECT_COVERAGE = "article_parent_subject_coverage"
PROFILE_PARENT_SUBJECT_CODES = "profile_parent_subject_codes"
PROFILE_PARENT_SUBJECT_OVERLAP_CODES = "profile_parent_subject_overlap_codes"
PROFILE_PARENT_SUBJECT_OVERLAP_COUNT = "profile_parent_subject_overlap_count"
PROFILE_PARENT_SUBJECT_COVERAGE = "profile_parent_subject_coverage"
CLUSTER_SUBJECT_CODES = "cluster_subject_codes"
CLUSTER_SUBJECT_COUNT = "cluster_subject_count"
CLUSTER_LEVEL_TWO_SUBJECT_CODES = "cluster_level_two_subject_codes"
EXACT_OVERLAP_CODES = "exact_overlap_codes"
EXACT_OVERLAP_COUNT = "exact_overlap_count"
EXACT_INFERRED_COVERAGE = "exact_inferred_coverage"
EXACT_SPECIFIC_OVERLAP_CODES = "exact_specific_overlap_codes"
EXACT_SPECIFIC_OVERLAP_COUNT = "exact_specific_overlap_count"
EXACT_SPECIFIC_COVERAGE = "exact_specific_coverage"
INFERRED_CODE_OVERLAP_CODES = "inferred_code_overlap_codes"
INFERRED_CODE_OVERLAP_COUNT = "inferred_code_overlap_count"
INFERRED_CODE_COVERAGE = "inferred_code_coverage"
OVERLAP_CODES = "overlap_codes"
OVERLAP_COUNT = "overlap_count"
INFERRED_COVERAGE = "inferred_coverage"
RESEARCHER_GENDER = "researcher_gender"
RESEARCHER_GENDER_SOURCE = "researcher_gender_source"
CLUSTER_GENDER = "cluster_gender"
CLUSTER_GENDER_SOURCE = "cluster_gender_source"
GENDER_COMPATIBLE = "gender_compatible"
LEFT_HISTORY = "left_history"
RIGHT_HISTORY = "right_history"
CURRENT_AFFILIATION_IDS = "current_affiliation_ids"
SELECTED_HISTORY = "selected_history"
POSSIBLE_CLUSTER_HISTORY = "possible_cluster_history"
SCOPUS_ASJC_CODES = "scopus_asjc_codes"
INFERRED_ASJC_LABELS = "inferred_asjc_labels"

# Public evidence labels and hard conflict limits.
INITIAL_ONLY_LATER_GIVEN_INITIAL_LABEL = "initial_only_later_given_initial"
INITIAL_ONLY_GIVEN_NAME_MATCH_LABEL = "initial_only_given_name_match"
GIVEN_CONFLICT_INITIAL_OVERLAP_LABEL = "given_conflict_with_initial_overlap"
TRANSPOSED_NAME_FIELDS_LABEL = "transposed_name_fields"
TEMPORAL_HARD_SPAN_YEARS = 70
TEMPORAL_HARD_GAP_YEARS = 25
TEMPORAL_SPAN_CONFLICT_LABEL = "publication_range_span_conflict"
TEMPORAL_GAP_CONFLICT_LABEL = "publication_range_gap_conflict"
TEMPORAL_HARD_CONFLICT_LABELS = frozenset(
    {TEMPORAL_SPAN_CONFLICT_LABEL, TEMPORAL_GAP_CONFLICT_LABEL}
)
STRONG_INITIAL_ONLY_ANCHOR_LABELS = frozenset(
    {"orcid_match", "grant_number_match", "self_citation"}
)
TEMPORAL_CONFLICT_OVERRIDE_ANCHOR_LABELS = frozenset(
    {"orcid_match", "grant_number_match", "self_citation"}
)
FINAL_SELECTION_STATUS_VALUES = frozenset(
    {
        "assigned_single_candidate",
        "assigned_single_cluster",
        "assigned_by_gender",
        "assigned_by_current_affiliation",
        "assigned_by_researcher_given_name",
        "assigned_by_combined_asjc",
        "assigned_by_article_asjc",
        "assigned_by_profile_asjc",
        "manual_selection_required",
    }
)
