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
IS_INBRED = "is_inbred"
N_PUBLIC = "n_public"
HAZEN_PERCENTILE = "hazen_perc_med"
PP_TOP_1 = "pp_top_1"
PP_TOP_5 = "pp_top_5"
PP_TOP_10 = "pp_top_10"
P_TOP_1 = "p_top_1"
P_TOP_5 = "p_top_5"
P_TOP_10 = "p_top_10"
ABSOLUTE_P_TOP_1 = "P_top1"
ABSOLUTE_P_TOP_5 = "P_top5"
ABSOLUTE_P_TOP_10 = "P_top10"
ACADEMIC_AGE = "academic_age"
FIRST_YEAR = "first_year"
PRIMARY_FIELD = "primary_field"
FIELD_SHARE_COLUMNS: tuple[str, ...] = (
    "multi_share",
    "agri_bio_share",
    "arts_hum_share",
    "biochem_share",
    "business_share",
    "chem_eng_share",
    "chem_share",
    "comp_sci_share",
    "decision_share",
    "earth_share",
    "econ_share",
    "energy_share",
    "eng_share",
    "env_sci_share",
    "immuno_share",
    "materials_share",
    "math_share",
    "med_share",
    "neuro_share",
    "nursing_share",
    "pharma_share",
    "physics_share",
    "psych_share",
    "soc_sci_share",
    "vet_share",
    "dentistry_share",
    "health_share",
)
DOMAIN_SHARE_COLUMNS: tuple[str, ...] = (
    "life_sci_share",
    "health_sci_share",
    "phys_sci_share",
    "soc_hum_share",
    "multi_area_share",
)
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
