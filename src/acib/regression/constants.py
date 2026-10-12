# Copyright © 2026 TU Wien

"""Regression-specific outcomes, model terms, and stable data-frame fields."""

from __future__ import annotations

from acib.constants import (
    ABSOLUTE_P_TOP_1,
    ABSOLUTE_P_TOP_5,
    ABSOLUTE_P_TOP_10,
    EID,
    FIRST_YEAR,
    HAZEN_PERCENTILE,
    IS_INBRED,
    N_PUBLIC,
    P_TOP_1,
    P_TOP_5,
    P_TOP_10,
    PERSON_TITLE,
    PP_TOP_1,
    PP_TOP_5,
    PP_TOP_10,
    PRIMARY_FIELD,
    RESEARCHER_ID,
    SUBJECT_AREAS,
    TITLE,
)
from acib.mobility.constants import (
    ACADEMIC_AGE_YEARS,
    ARTICLE_COVER_DATE as COVER_DATE,
    ARTICLE_DOI as DOI,
    IS_INBREEDING_PROXY,
    N_UNIQUE,
    PUBLICATION_GENERAL_ASJC_SUBJECT_AREAS as ARTICLE_ASJC_FIELD_COLUMN,
)
from acib.name_disambiguation.constants import (
    AUTHOR_IDS,
    SELECTED_DISAMBIGUATED_CLUSTER_ID as SELECTED_CLUSTER_ID,
)

PAPER_REGRESSION_DEPENDENT_VARIABLES: tuple[str, ...] = (
    HAZEN_PERCENTILE,
    PP_TOP_10,
    PP_TOP_5,
    PP_TOP_1,
    ABSOLUTE_P_TOP_10,
    ABSOLUTE_P_TOP_5,
    ABSOLUTE_P_TOP_1,
)
DEFAULT_REGRESSION_DEPENDENT_VARIABLES = PAPER_REGRESSION_DEPENDENT_VARIABLES
RELATIVE_PERFORMANCE_OUTCOMES: tuple[str, ...] = (
    HAZEN_PERCENTILE,
    PP_TOP_10,
    PP_TOP_5,
    PP_TOP_1,
)
ABSOLUTE_TOP_PERCENTILE_OUTCOMES: tuple[str, ...] = (
    ABSOLUTE_P_TOP_10,
    ABSOLUTE_P_TOP_5,
    ABSOLUTE_P_TOP_1,
)
ABSOLUTE_PERFORMANCE_OUTCOMES: tuple[str, ...] = (N_PUBLIC, *ABSOLUTE_TOP_PERCENTILE_OUTCOMES)

DEFAULT_REGRESSION_OUTCOME_METADATA: dict[str, tuple[str, str]] = {
    HAZEN_PERCENTILE: ("relative", "mwPR"),
    PP_TOP_10: ("relative", r"PP(top 10\%)"),
    PP_TOP_5: ("relative", r"PP(top 5\%)"),
    PP_TOP_1: ("relative", r"PP(top 1\%)"),
    N_PUBLIC: ("absolute", "Publications"),
    ABSOLUTE_P_TOP_10: ("absolute", r"P(top 10\%)"),
    ABSOLUTE_P_TOP_5: ("absolute", r"P(top 5\%)"),
    ABSOLUTE_P_TOP_1: ("absolute", r"P(top 1\%)"),
}
OUTCOME_DISPLAY_LABELS: dict[str, str] = {
    outcome: alias for outcome, (_, alias) in DEFAULT_REGRESSION_OUTCOME_METADATA.items()
}

FIELD_SHARE_TERMS = "FIELD_SHARE_TERMS"
BROAD_FIELD_SHARE_TERMS = "BROAD_FIELD_SHARE_TERMS"

PAPER_REGRESSION_MODEL_TERMS: dict[str, tuple[str, ...]] = {
    "m1": ("C(is_inbred)", "academic_age", "C(title)"),
    "m2": ("C(is_inbred)", "academic_age", "C(title)", "n_away_years"),
    "m3": (
        "C(is_inbred)",
        "academic_age",
        "C(title)",
        "n_away_years_domestic",
        "n_away_years_abroad",
    ),
    "m4": (
        "C(is_inbred)",
        "academic_age",
        "C(title)",
        "n_away_years",
        "n_away_years_abroad",
    ),
    "m5": (
        "C(is_inbred)",
        "academic_age",
        "C(title)",
        "n_away_years_domestic",
        "n_away_years_abroad",
        "coaffiliation_publication_share",
    ),
    "m6": (
        "C(is_inbred)",
        "academic_age",
        "C(title)",
        "coaffiliation_publication_share",
        "mean_coauthors_per_publication",
        "n_away_years_domestic",
        "n_away_years_abroad",
    ),
    "m7": (
        "C(is_inbred)",
        "academic_age",
        "C(title)",
        "coaffiliation_publication_share",
        "mean_coauthors_per_publication",
        "n_away_years_domestic",
        "n_away_years_abroad",
        "I(academic_age ** 2)",
    ),
    "m8": (
        "C(is_inbred)",
        "academic_age",
        "C(title)",
        "coaffiliation_publication_share",
        "mean_coauthors_per_publication",
        "n_away_years_domestic",
        "n_away_years_abroad",
        FIELD_SHARE_TERMS,
    ),
    "m9": (
        "C(is_inbred)",
        "academic_age",
        "C(title)",
        "coaffiliation_publication_share",
        "mean_coauthors_per_publication",
        "n_away_years_domestic",
        "n_away_years_abroad",
        "first_year",
        FIELD_SHARE_TERMS,
    ),
    "m10": (
        "C(is_inbred)",
        "academic_age",
        "C(title)",
        "coaffiliation_publication_share",
        "mean_coauthors_per_publication",
        "n_away_years_domestic",
        "first_year",
        FIELD_SHARE_TERMS,
    ),
}
REGRESSION_TITLE_EXCLUSIONS: tuple[str, ...] = ()

# Regression-specific fields and preserved input/output projections.
N_PUBLIC_IMPACT = "n_public_impact"
PUBLICATION_RATE = "publication_rate"
LOG_PUBLICATION_RATE = "log_publication_rate"
LOG_N_PUBLIC = "log_n_public"
LOG_N_AFFILIATIONS = "log_n_affiliations"
PRIMARY_FIELD_SHARE = "primary_field_share"
PUBLIC_PROD = "public_prod"
PUBLIC_PROD_IMPACT = "public_prod_impact"
PUBLICATION_MOBILITY = "publication_mobility"
CLASS_NAME = "class_name"
PUBYEAR = "pubyear"
CITED_BY_COUNT = "citedby_count"
IMPACT_ASJC_FIELD_COLUMN = "impact_asjc_subject_area"
LEIDEN_P_TOP_1 = "leiden_p_top_1"
LEIDEN_P_TOP_5 = "leiden_p_top_5"
LEIDEN_P_TOP_10 = "leiden_p_top_10"
PTOP_1_FRACTION = "ptop1_frac"
PTOP_5_FRACTION = "ptop5_frac"
PTOP_10_FRACTION = "ptop10_frac"
DI1 = "di1"
DI5 = "di5"
ARTICLE_IMPACT_ID = "item_id"
FOCAL_EID = "focal_eid"
AUTHOR_ARTICLE_COLUMNS = (
    EID,
    DOI,
    COVER_DATE,
    PUBYEAR,
    CITED_BY_COUNT,
    AUTHOR_IDS,
    HAZEN_PERCENTILE,
    PP_TOP_1,
    PP_TOP_5,
    PP_TOP_10,
    LEIDEN_P_TOP_1,
    LEIDEN_P_TOP_5,
    LEIDEN_P_TOP_10,
    P_TOP_1,
    P_TOP_5,
    P_TOP_10,
    PTOP_1_FRACTION,
    PTOP_5_FRACTION,
    PTOP_10_FRACTION,
    CLASS_NAME,
    IMPACT_ASJC_FIELD_COLUMN,
    SUBJECT_AREAS,
    ARTICLE_ASJC_FIELD_COLUMN,
    DI1,
    DI5,
)
ARTICLE_IMPACT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    HAZEN_PERCENTILE: (HAZEN_PERCENTILE,),
    PP_TOP_1: (PP_TOP_1, LEIDEN_P_TOP_1, P_TOP_1, PTOP_1_FRACTION),
    PP_TOP_5: (PP_TOP_5, LEIDEN_P_TOP_5, P_TOP_5, PTOP_5_FRACTION),
    PP_TOP_10: (PP_TOP_10, LEIDEN_P_TOP_10, P_TOP_10, PTOP_10_FRACTION),
}
FRACTIONAL_TOP_COLUMNS = (PTOP_10_FRACTION, PTOP_5_FRACTION, PTOP_1_FRACTION)
IMPACT_PUBLICATION_YEAR_RANGE = (1996, 2023)
ARTICLE_IMPACT_PASSTHROUGH_COLUMNS = (CLASS_NAME, PUBYEAR, *FRACTIONAL_TOP_COLUMNS)

PROFESSOR_ARTICLE_OUTPUT_COLUMNS = (
    EID,
    DOI,
    COVER_DATE,
    PUBYEAR,
    AUTHOR_IDS,
    SELECTED_CLUSTER_ID,
    RESEARCHER_ID,
    PERSON_TITLE,
    TITLE,
    IS_INBREEDING_PROXY,
    IS_INBRED,
    HAZEN_PERCENTILE,
    PP_TOP_1,
    PP_TOP_5,
    PP_TOP_10,
    *FRACTIONAL_TOP_COLUMNS,
    CLASS_NAME,
    IMPACT_ASJC_FIELD_COLUMN,
    ARTICLE_ASJC_FIELD_COLUMN,
    CITED_BY_COUNT,
    ACADEMIC_AGE_YEARS,
    N_UNIQUE,
    PRIMARY_FIELD,
    FIRST_YEAR,
)

BROAD_ASJC_DESCRIPTIONS: dict[int, str] = {
    10: "Multidisciplinary",
    11: "Agricultural and Biological Sciences",
    12: "Arts and Humanities",
    13: "Biochemistry",
    14: "Business",
    15: "Chemical Engineering",
    16: "Chemistry",
    17: "Computer Science",
    18: "Decision Sciences",
    19: "Earth and Planetary Sciences",
    20: "Economics",
    21: "Energy",
    22: "Engineering",
    23: "Environmental Science",
    24: "Immunology and Microbiology",
    25: "Materials Science",
    26: "Mathematics",
    27: "Medicine",
    28: "Neuroscience",
    29: "Nursing",
    30: "Pharmacology",
    31: "Physics and Astronomy",
    32: "Psychology",
    33: "Social Sciences",
    34: "Veterinary",
    35: "Dentistry",
    36: "Health Professions",
}
