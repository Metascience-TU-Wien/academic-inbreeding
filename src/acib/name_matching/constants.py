# Copyright © 2026 TU Wien

"""DataFrame schema and matching constants for the name-matching stage."""

from __future__ import annotations

from acib.constants import (
    AFFILIATION_ID,
    AFFILIATION_ID_PARENT,
    AFFILIATION_ID_PARENT_CHATGPT,
    AFFILIATION_ID_PARENT_SPLINK_CHATGPT,
    AFFILIATION_IDS_CURRENT,
    AFFILIATION_NAME,
    AFFILIATION_PARENT_SOURCE,
    AFFILIATION_UMBRELLA_ID,
    AUTHOR_ID,
    EID,
    GIVEN_NAME,
    HISTORICAL_IDENTIFIER,
    RESEARCHER_ID,
    SUBJECT_AREAS,
    SURNAME,
)

# Researcher input and output columns.
RESEARCHER_NAME = "researcher_name"
RESEARCHER_SURNAME = "person_surname"
RESEARCHER_GIVEN_NAME = "person_name"
ORGANIZATION_NAME = "organization"
SC_ID_LOGICAL = "sc_id_logical"
PERSON_FULLNAME = "person_fullname"

RESEARCHER_COLUMNS = (
    RESEARCHER_ID,
    RESEARCHER_NAME,
)

# Author candidate input and output columns.
AUTHOR_INTERNAL_ID = "_id"
AUTHOR_NAME = "author_name"
NAME = "name"
NAME_VARIANTS = "name_variants"
INDEXED_NAME = "indexed_name"
INITIALS = "initials"
SC_ID_ALL_PROFS = "sc_id_all_profs"
PUBLICATION_RANGE = "publication_range"
SUBJECT_ABBREVIATION = "subject_abbreviation"
SUBJECT_CODE = "subject_code"

# Affiliation schema shared among matching operations and stage preparation.
CURRENT_AFFILIATION = "current_affiliation"

PARENT_AFFILIATION_COLUMNS = (
    AFFILIATION_ID_PARENT_CHATGPT,
    AFFILIATION_ID_PARENT,
    AFFILIATION_PARENT_SOURCE,
    AFFILIATION_ID_PARENT_SPLINK_CHATGPT,
    AFFILIATION_UMBRELLA_ID,
)

AUTHOR_COLUMNS = (
    EID,
    AUTHOR_INTERNAL_ID,
    AUTHOR_ID,
    AUTHOR_NAME,
    AFFILIATION_ID,
    AFFILIATION_ID_PARENT,
    AFFILIATION_ID_PARENT_CHATGPT,
    AFFILIATION_ID_PARENT_SPLINK_CHATGPT,
    AFFILIATION_NAME,
    PUBLICATION_RANGE,
    SUBJECT_ABBREVIATION,
    SUBJECT_AREAS,
    SUBJECT_CODE,
    INDEXED_NAME,
    INITIALS,
    NAME,
    SURNAME,
    GIVEN_NAME,
    NAME_VARIANTS,
    SC_ID_ALL_PROFS,
    HISTORICAL_IDENTIFIER,
    AFFILIATION_IDS_CURRENT,
)

# Stable result-schema columns.
MATCH_PRIORITY = "priority"
AUTHOR_NAME_PREFIX = "name_prefix"
MAPPING_ID = "mapping_id"

# This split-name pair is consumed by both the helper and containment strategy.
_SPLIT_SURNAME_COLUMN = "_surname"
_SPLIT_GIVEN_NAME_COLUMN = "_given"

VETMED_UNI_NAME = "Vet meduni Wien"
SURNAME_AS_NAME_UNIVERSITIES = (VETMED_UNI_NAME,)

# Source-specific organization-to-affiliation mappings.
BIO_TECH_MED_GRAZ_ID = 60112421
BIO_TECH_MED_GRAZ_ID_OA = "https://openalex.org/I4210117258"
NAWI_GRAZ_ID_OA = "https://openalex.org/I4210108044"

SCOPUS_ID_MAPPING: dict[str, list[int]] = {
    "FP fernfh": [109298005, 120334545],
    "FH Salzburg": [60009709],
    "Kunstuni Graz": [60015267],
    "FH Johanneum": [60023270],
    "Vienna University of Economics and Business": [60000583],
    "FH Gesundheit Tirol": [112336683, 123454241],
    "MEDUNI Graz": [60006224, BIO_TECH_MED_GRAZ_ID],
    "Karl Franzens Uni Graz": [60022457, BIO_TECH_MED_GRAZ_ID],
    "Uni Klagenfurt": [60019660],
    "FH Burgenland": [60022233, 129780914],
    "FH Campus Wien": [60016651],
    "Campus02": [60024197],
    "University of Vienna": [60025988],
    "FH Sankt Poelten": [60025467, 128641636],
    "FH Miles": [113133513],
    "FH Technikum Wien": [60011929],
    "FH Wiener Neustadt": [60030290],
    "TU Graz": [60019663, BIO_TECH_MED_GRAZ_ID],
    "FH Kufstein": [60012136],
    "MEDUNI Innsbruck": [60008222],
    "FH Gesundheitsberufe": [123454241],
    "mozarteum_salzburg": [60029765],
    "Uni Salzburg": [60002963, 100581856],
    "FH BFI Vienna": [60004966],
    "Johannes Kepler Uni Linz": [60021931],
    "Donau Uni Krems": [60004038],
    "MEDUNI Wien": [60011394],
    "Uni Innsbruck": [60009999],
    "FH Wien WKW": [130982383],
    "kunstuni_linz": [112350659, 60009763],
    "TU Wien": [60018163],
    "mci hochschule": [60004100],
    "FH Oberösterreich": [60032193, 119146064, 101928095, 114521147],
    "AKB Wien": [60001045],
    "Uni AK Wien": [60020442],
    "FH Kärnten": [60011747],
    "Montanuni Leoben": [60028349],
    "FH IMC Krems": [60011692],
    "BOKU Wien": [60024895],
    "Lauder_Business": [112709324, 121578158],
    "MUK Wien": [60015267],
    VETMED_UNI_NAME: [60003417],
}

OPENALEX_ID_MAPPING: dict[str, list[str]] = {
    "TU Wien": ["https://openalex.org/I145847075"],
    "TU Graz": ["https://openalex.org/I4092182", NAWI_GRAZ_ID_OA, BIO_TECH_MED_GRAZ_ID_OA],
    "Karl Franzens Uni Graz": [
        "https://openalex.org/I15766117",
        NAWI_GRAZ_ID_OA,
        BIO_TECH_MED_GRAZ_ID_OA,
    ],
    "University of Vienna": ["https://openalex.org/I129774422"],
    "AKB Wien": ["https://openalex.org/I5075986"],
    "FP fernfh": ["https://openalex.org/I4405261147"],
    "FH BFI Vienna": ["https://openalex.org/I4210109805"],
    "FH Burgenland": ["https://openalex.org/I4210094037"],
    "FH Campus Wien": ["https://openalex.org/I4210088715"],
    "FH IMC Krems": ["https://openalex.org/I2800230563"],
    "FH Gesundheitsberufe": ["https://openalex.org/I4210108027"],
    "FH Kärnten": ["https://openalex.org/I4210127953"],
    "FH Kufstein": ["https://openalex.org/I93485673"],
    "FH Miles": [],
    "FH Oberösterreich": ["https://openalex.org/I4210136249"],
    "FH Salzburg": ["https://openalex.org/I4210087982"],
    "FH Sankt Poelten": ["https://openalex.org/I25485817"],
    "FH Technikum Wien": ["https://openalex.org/I121760703"],
    "FH Wiener Neustadt": ["https://openalex.org/I38523324"],
    "FH Gesundheit Tirol": ["https://openalex.org/I4210091108"],
    "Johannes Kepler Uni Linz": ["https://openalex.org/I121883995"],
    "kunstuni_linz": ["https://openalex.org/I4210115111"],
    "Lauder_Business": ["https://openalex.org/I2800706213"],
    "mci hochschule": ["https://openalex.org/I41073452"],
    "MEDUNI Graz": ["https://openalex.org/institutions/i202276237", BIO_TECH_MED_GRAZ_ID_OA],
    "MEDUNI Innsbruck": ["https://openalex.org/I143397708"],
    "MEDUNI Wien": ["https://openalex.org/I76134821"],
    "mozarteum_salzburg": ["https://openalex.org/I318632792"],
    "FH Wien WKW": ["https://openalex.org/I4405252845"],
    "FH Johanneum": ["https://openalex.org/I103365165"],
    "Campus02": ["https://openalex.org/I4210126885"],
    "Uni Salzburg": ["https://openalex.org/I182212641"],
    "Kunstuni Graz": ["https://openalex.org/I317320687"],
    "BOKU Wien": ["https://openalex.org/I92869138"],
    "Donau Uni Krems": ["https://openalex.org/I132976966"],
    "MUK Wien": ["https://openalex.org/I3132041272"],
    "Vienna University of Economics and Business": ["https://openalex.org/I102248843"],
    "Montanuni Leoben": ["https://openalex.org/I182712176"],
    "Uni Innsbruck": ["https://openalex.org/I190249584"],
    "Uni Klagenfurt": ["https://openalex.org/I4210166741"],
    "Uni AK Wien": ["https://openalex.org/I143923934"],
    VETMED_UNI_NAME: ["https://openalex.org/I150540706"],
}
