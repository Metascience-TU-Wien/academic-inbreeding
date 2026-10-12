# Copyright © 2026 TU Wien

"""Constants for fields shared across processing stages."""

# Researcher, author, and publication identifiers shared by pipeline stages.
RESEARCHER_ID = "sc_id"
# This is the Scopus author-profile identifier, not the source database row ID.
AUTHOR_ID = "author_id"
EID = "eid"

# Shared author identity and evidence columns.
PERSON_TITLE = "person_title"
TITLE_CLASS = "person_title_class"
TITLE = "title"
UNIVERSITY_NAME = "university_name"
SURNAME = "surname"
GIVEN_NAME = "given_name"
HISTORICAL_IDENTIFIER = "historical_identifier"
SUBJECT_AREAS = "subject_areas"
INFERRED_ASJC_CODES = "inferred_asjc_codes"

# Shared affiliation schema columns.
AFFILIATION_ID = "affiliation_id"
AFFILIATION_ID_PARENT = "affiliation_id_parent"
AFFILIATION_ID_PARENT_CHATGPT = "affiliation_id_parent_chatgpt"
AFFILIATION_ID_PARENT_SPLINK_CHATGPT = "affiliation_id_parent_splink_chatgpt"
AFFILIATION_PARENT_SOURCE = "parent"
AFFILIATION_UMBRELLA_ID = "umbrella_id"
AFFILIATION_NAME = "affiliation_name"
AFFILIATION_IDS_CURRENT = "affiliation_ids_current"
AFFILIATION_ID_CURRENT = "affiliation_id_current"
