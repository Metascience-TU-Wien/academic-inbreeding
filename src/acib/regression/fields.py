# Copyright © 2026 TU Wien

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from acib.constants import DOMAIN_SHARE_COLUMNS, FIELD_SHARE_COLUMNS
from acib.mobility.input import general_asjc_subject_area_dict
from acib.name_disambiguation.asjc import ASJC_TAXONOMY

ASJC_MAJOR_FIELDS: dict[int, str] = {
    1000: "Multidisciplinary",
    1100: "Agricultural and Biological Sciences",
    1200: "Arts and Humanities",
    1300: "Biochemistry, Genetics and Molecular Biology",
    1400: "Business, Management and Accounting",
    1500: "Chemical Engineering",
    1600: "Chemistry",
    1700: "Computer Science",
    1800: "Decision Sciences",
    1900: "Earth and Planetary Sciences",
    2000: "Economics, Econometrics and Finance",
    2100: "Energy",
    2200: "Engineering",
    2300: "Environmental Science",
    2400: "Immunology and Microbiology",
    2500: "Materials Science",
    2600: "Mathematics",
    2700: "Medicine",
    2800: "Neuroscience",
    2900: "Nursing",
    3000: "Pharmacology, Toxicology and Pharmaceutics",
    3100: "Physics and Astronomy",
    3200: "Psychology",
    3300: "Social Sciences",
    3400: "Veterinary",
    3500: "Dentistry",
    3600: "Health Professions",
}

ASJC_FIELD_NAMES: dict[int, str] = {
    1000: "multi",
    1100: "agri_bio",
    1200: "arts_hum",
    1300: "biochem",
    1400: "business",
    1500: "chem_eng",
    1600: "chem",
    1700: "comp_sci",
    1800: "decision",
    1900: "earth",
    2000: "econ",
    2100: "energy",
    2200: "eng",
    2300: "env_sci",
    2400: "immuno",
    2500: "materials",
    2600: "math",
    2700: "med",
    2800: "neuro",
    2900: "nursing",
    3000: "pharma",
    3100: "physics",
    3200: "psych",
    3300: "soc_sci",
    3400: "vet",
    3500: "dentistry",
    3600: "health",
}

ASJC_DOMAIN_NAMES: dict[str, str] = {
    "life_sci": "Life Sciences",
    "health_sci": "Health Sciences",
    "phys_sci": "Physical Sciences",
    "soc_hum": "Social Sciences & Humanities",
    "multi_area": "Multidisciplinary",
}
ASJC_DOMAIN_CODES: dict[str, tuple[int, ...]] = {
    "life_sci": (1100, 1300, 2400, 2800, 3000),
    "health_sci": (2700, 2900, 3400, 3500, 3600),
    "phys_sci": (1500, 1600, 1700, 1900, 2100, 2200, 2300, 2500, 2600, 3100),
    "soc_hum": (1200, 1400, 1800, 2000, 3200, 3300),
    "multi_area": (1000,),
}
ASJC_CODE_DOMAIN: dict[int, str] = {
    code: domain for domain, codes in ASJC_DOMAIN_CODES.items() for code in codes
}
IMPACT_CLASS_NAME_ALIASES: dict[str, int] = {
    "Tourism, Leisure and Hospitality Management": 1400,
    "Electronic, Optical and Magnetic Materials": 2500,
    "Surfaces, Coatings and Films": 2500,
    "Atomic and Molecular Physics, and Optics": 3100,
    "Radiology, Nuclear Medicine and Imaging": 2700,
    "Public Health, Environmental and Occupational Health": 2700,
    "Health, Toxicology and Mutagenesis": 2300,
    "Management, Monitoring, Policy and Law": 2300,
}


@dataclass(frozen=True, slots=True)
class FieldShareResult:
    """Hold field-share output tables.

    :param shares: Professor-level field and domain shares.
    :param publication_contributions: Fractional publication-field records.
    :param diagnostics: Summary metrics for the calculated shares.
    """

    shares: pd.DataFrame
    publication_contributions: pd.DataFrame
    diagnostics: pd.DataFrame


def professor_field_shares(
    professor_articles: pd.DataFrame,
    *,
    group_column: str = "sc_id",
    publication_column: str = "eid",
    field_column: str = "article_general_asjc_subject_areas",
) -> FieldShareResult:
    """Calculate professor-level publication field shares and diagnostics.

    :param professor_articles: Article records with professor, publication, and field data.
    :param group_column: Column identifying a professor or other analysis group.
    :param publication_column: Column identifying a publication.
    :param field_column: Column containing publication ASJC subject areas.
    :return: Shares, publication contributions, and diagnostic metrics.
    """
    contributions = fractional_publication_field_contributions(
        professor_articles,
        group_column=group_column,
        publication_column=publication_column,
        field_column=field_column,
    )
    shares = aggregate_field_shares(contributions, group_column=group_column)
    diagnostics = field_share_diagnostics(
        shares=shares,
        contributions=contributions,
        group_column=group_column,
        publication_column=publication_column,
    )
    return FieldShareResult(
        shares=shares,
        publication_contributions=contributions,
        diagnostics=diagnostics,
    )


def fractional_publication_field_contributions(
    professor_articles: pd.DataFrame,
    *,
    group_column: str,
    publication_column: str,
    field_column: str,
) -> pd.DataFrame:
    """Assign fractional major-field weights to each unique publication.

    :param professor_articles: Article records to classify.
    :param group_column: Column identifying the analysis group.
    :param publication_column: Column identifying a publication.
    :param field_column: Column containing ASJC subject areas.
    :return: One row per publication and recognized major field.
    :raises KeyError: If a group or publication identifier is absent.
    """
    required = {group_column, publication_column}
    missing = required - set(professor_articles.columns)
    if missing:
        raise KeyError(f"Field-share input is missing required columns: {sorted(missing)}")

    columns = [group_column, publication_column]
    if field_column in professor_articles.columns:
        columns.append(field_column)
    publications = professor_articles[columns].drop_duplicates([group_column, publication_column])
    rows: list[dict[str, Any]] = []
    for row in publications.to_dict(orient="records"):
        raw_fields = row.get(field_column)
        field_codes = publication_major_codes(raw_fields)
        classified_codes = tuple(code for code in field_codes if code in ASJC_FIELD_NAMES)
        if not classified_codes:
            rows.append(
                {
                    group_column: row[group_column],
                    publication_column: row[publication_column],
                    "field_code": None,
                    "field": None,
                    "field_share_column": None,
                    "domain": None,
                    "domain_share_column": None,
                    "field_weight": 0.0,
                    "n_publication_fields": 0,
                    "is_unclassified": True,
                }
            )
            continue
        weight = 1.0 / len(classified_codes)
        for field_code in classified_codes:
            field = ASJC_FIELD_NAMES[field_code]
            domain = ASJC_CODE_DOMAIN.get(field_code) if field_code is not None else None
            rows.append(
                {
                    group_column: row[group_column],
                    publication_column: row[publication_column],
                    "field_code": field_code,
                    "field": field,
                    "field_share_column": f"{field}_share",
                    "domain": domain,
                    "domain_share_column": f"{domain}_share" if domain else None,
                    "field_weight": weight,
                    "n_publication_fields": len(classified_codes),
                    "is_unclassified": not publication_has_classified_field(raw_fields),
                }
            )
    return pd.DataFrame(rows)


def publication_field_names(value: Any) -> tuple[str, ...]:
    """Return sorted unique recognized major-field names for a publication.

    :param value: Parsed or serialized ASJC subject areas.
    :return: Recognized field abbreviations in sorted order.
    """
    fields = []
    for code in publication_major_codes(value):
        if code in ASJC_FIELD_NAMES:
            fields.append(ASJC_FIELD_NAMES[code])
    return tuple(sorted(set(fields)))


@lru_cache(maxsize=1)
def impact_class_name_asjc_map() -> dict[str, int]:
    """Map normalized impact ``class_name`` labels to ASJC major-field codes.

    :return: Normalized labels mapped to major-field codes.
    """
    mapping: dict[str, int] = {}
    for code, entry in ASJC_TAXONOMY.items():
        major_code = major_asjc_code(code)
        if major_code not in ASJC_FIELD_NAMES:
            continue
        add_impact_class_name(mapping, entry.description, major_code)
        if code == major_code:
            add_impact_class_name(mapping, entry.description.removeprefix("General "), major_code)
    for code, description in ASJC_MAJOR_FIELDS.items():
        add_impact_class_name(mapping, description, code)
        add_impact_class_name(mapping, f"{description} (all)", code)
    for label, code in IMPACT_CLASS_NAME_ALIASES.items():
        add_impact_class_name(mapping, label, code)
    return mapping


def add_impact_class_name(mapping: dict[str, int], label: str, major_code: int) -> None:
    """Add a normalized class-name label to an ASJC lookup.

    :param mapping: Lookup to update.
    :param label: Impact-file class label.
    :param major_code: Corresponding ASJC major-field code.
    """
    normalized = normalize_impact_class_name(label)
    if normalized:
        mapping[normalized] = major_code


def impact_asjc_subject_area(value: Any) -> dict[int, str]:
    """Estimate ASJC major fields from an impact-file ``class_name`` value.

    :param value: Impact-file class-name value.
    :return: Recognized major codes mapped to their descriptions.
    """
    codes = impact_asjc_major_codes(value)
    return {code: ASJC_MAJOR_FIELDS[code] for code in codes if code in ASJC_MAJOR_FIELDS}


def impact_asjc_major_codes(value: Any) -> tuple[int, ...]:
    """Infer recognized ASJC major codes from an impact class-name value.

    :param value: Impact-file class-name value.
    :return: Recognized major codes in ascending order.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return ()
    if isinstance(value, float) and np.isnan(value):
        return ()
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null", "{}"}:
        return ()

    mapping = impact_class_name_asjc_map()
    codes = {
        code
        for label in matched_impact_class_name_labels(text)
        if (code := mapping.get(normalize_impact_class_name(label)))
    }
    return tuple(sorted(codes))


def impact_class_name_tokens(value: str) -> tuple[str, ...]:
    """Split a serialized impact class-name value into nonempty tokens.

    :param value: Serialized class-name value.
    :return: Trimmed tokens in source order.
    """
    text = value.strip().strip("{}").strip()
    return tuple(token.strip() for token in text.split(",") if token.strip())


def matched_impact_class_name_labels(value: str) -> tuple[str, ...]:
    """Return full ASJC labels matched from a possibly comma-containing class string.

    :param value: Serialized impact class-name value.
    :return: Matched labels in source order.
    """
    matched, _ = parse_impact_class_name_labels(value)
    return matched


def unmatched_impact_class_name_labels(value: str) -> tuple[str, ...]:
    """Return fragments that cannot be covered by known ASJC labels.

    :param value: Serialized impact class-name value.
    :return: Unmatched fragments in source order.
    """
    _, unmatched = parse_impact_class_name_labels(value)
    return unmatched


def parse_impact_class_name_labels(value: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Separate recognized labels from unmatched class-name fragments.

    :param value: Serialized impact class-name value.
    :return: Matched labels and unmatched fragments, in source order.
    """
    parts = impact_class_name_tokens(value)
    if not parts:
        return (), ()
    known_labels = impact_class_name_asjc_map()
    _, segments = best_impact_class_name_segments(parts, known_labels=set(known_labels))
    matched: list[str] = []
    unmatched: list[str] = []
    current_unmatched: list[str] = []
    for is_known, start, end in segments:
        label = ", ".join(parts[start:end])
        if is_known:
            if current_unmatched:
                unmatched.append(", ".join(current_unmatched))
                current_unmatched = []
            matched.append(label)
        else:
            current_unmatched.extend(parts[start:end])
    if current_unmatched:
        unmatched.append(", ".join(current_unmatched))
    return tuple(matched), tuple(unmatched)


def best_impact_class_name_segments(
    parts: tuple[str, ...], *, known_labels: set[str]
) -> tuple[int, list[tuple[bool, int, int]]]:
    """Find a maximum-coverage segmentation into known class labels.

    :param parts: Ordered fragments from a class-name value.
    :param known_labels: Normalized labels recognized by the ASJC lookup.
    :return: Coverage score and ordered known/unknown segment boundaries.
    """
    n_parts = len(parts)
    best: list[tuple[int, list[tuple[bool, int, int]]]] = [(-1, []) for _ in range(n_parts + 1)]
    best[n_parts] = (0, [])
    for index in range(n_parts - 1, -1, -1):
        suffix_score, suffix_segments = best[index + 1]
        best_score = suffix_score
        best_segments = [(False, index, index + 1), *suffix_segments]
        for end in range(n_parts, index, -1):
            label = ", ".join(parts[index:end])
            if normalize_impact_class_name(label) not in known_labels:
                continue
            candidate_score, candidate_segments = best[end]
            candidate_score += end - index
            if candidate_score > best_score:
                best_score = candidate_score
                best_segments = [(True, index, end), *candidate_segments]
        best[index] = (best_score, best_segments)
    return best[0]


def normalize_impact_class_name(value: str) -> str:
    """Normalize a class-name label for case- and punctuation-insensitive matching.

    :param value: Class-name label.
    :return: Normalized label.
    """
    normalized = value.casefold().replace("&", " and ")
    normalized = re.sub(r"\(all\)", " ", normalized)
    normalized = re.sub(r"^\s*general\s+", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def publication_has_classified_field(value: Any) -> bool:
    """Return whether a publication contains a recognized major field.

    :param value: Parsed or serialized ASJC subject areas.
    :return: Whether at least one recognized major field is present.
    """
    return any(code in ASJC_FIELD_NAMES for code in publication_major_codes(value))


def publication_major_codes(value: Any) -> tuple[int, ...]:
    """Return sorted major ASJC codes parsed from publication subject areas.

    :param value: Parsed or serialized ASJC subject areas.
    :return: Unique major codes in ascending order.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return ()
    if isinstance(value, float) and np.isnan(value):
        return ()

    if isinstance(value, dict):
        codes = value.keys()
    else:
        codes = general_asjc_subject_area_dict(value).keys()
    return tuple(sorted({code for raw_code in codes if (code := major_asjc_code(raw_code))}))


def major_asjc_code(value: Any) -> int | None:
    """Normalize an ASJC code to its hundred-level major-field code.

    :param value: Numeric or string ASJC code.
    :return: Major-field code, or ``None`` when the value is invalid.
    """
    try:
        code = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    if 10 <= code <= 36:
        return code * 100
    if code >= 1000:
        return (code // 100) * 100
    return None


def aggregate_field_shares(contributions: pd.DataFrame, *, group_column: str) -> pd.DataFrame:
    """Aggregate publication contributions into group-level field shares.

    :param contributions: Fractional publication-field contributions.
    :param group_column: Column identifying an analysis group.
    :return: Field/domain share table with effective field counts.
    """
    if contributions.empty:
        return pd.DataFrame(
            columns=[
                group_column,
                *FIELD_SHARE_COLUMNS,
                *DOMAIN_SHARE_COLUMNS,
                "effective_n_fields",
            ]
        )
    group_index = pd.Index(contributions[group_column].drop_duplicates())
    field_rows = contributions.dropna(subset=["field_share_column"])
    if field_rows.empty:
        counts = pd.DataFrame(index=group_index)
    else:
        field_group = [group_column, "field_share_column"]
        counts = (
            field_rows.groupby(field_group, dropna=False)["field_weight"]
            .sum()
            .unstack(fill_value=0.0)
        )
        counts = counts.reindex(group_index, fill_value=0.0)
    for column in FIELD_SHARE_COLUMNS:
        if column not in counts.columns:
            counts[column] = 0.0
    counts = counts.loc[:, list(FIELD_SHARE_COLUMNS)]
    totals = counts.sum(axis=1).replace(0, np.nan)
    domain_counts = aggregate_domain_counts(contributions, group_column=group_column)
    shares = counts.div(totals, axis=0).fillna(0.0)
    domain_shares = domain_counts.div(totals, axis=0).fillna(0.0)
    shares = shares.join(domain_shares)
    shares["effective_n_fields"] = 1.0 / shares.loc[:, list(FIELD_SHARE_COLUMNS)].pow(2).sum(
        axis=1
    ).replace(0, np.nan)
    return shares.reset_index()


def aggregate_domain_counts(contributions: pd.DataFrame, *, group_column: str) -> pd.DataFrame:
    """Aggregate fractional field contributions by group and domain.

    :param contributions: Fractional publication-field contributions.
    :param group_column: Column identifying an analysis group.
    :return: Group-indexed domain contribution table.
    """
    group_index = pd.Index(contributions[group_column].drop_duplicates())
    domain_rows = contributions.dropna(subset=["domain_share_column"])
    if domain_rows.empty:
        result = pd.DataFrame(index=group_index)
    else:
        result = (
            domain_rows.groupby([group_column, "domain_share_column"], dropna=False)["field_weight"]
            .sum()
            .unstack(fill_value=0.0)
        )
        result = result.reindex(group_index, fill_value=0.0)
    for column in DOMAIN_SHARE_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0
    return result.loc[:, list(DOMAIN_SHARE_COLUMNS)]


def field_share_diagnostics(
    *,
    shares: pd.DataFrame,
    contributions: pd.DataFrame,
    group_column: str,
    publication_column: str,
) -> pd.DataFrame:
    """Build diagnostics for publication weights and group-level share totals.

    :param shares: Aggregated group-level shares.
    :param contributions: Publication-level field contributions.
    :param group_column: Column identifying an analysis group.
    :param publication_column: Column identifying a publication.
    :return: Two-column diagnostic metric table.
    """
    rows = []
    publication_weights = contributions.groupby([group_column, publication_column])[
        "field_weight"
    ].sum()
    share_sums = (
        shares.loc[:, list(FIELD_SHARE_COLUMNS)].sum(axis=1)
        if not shares.empty
        else pd.Series(dtype=float)
    )
    rows.append(metric_row("professors", len(shares)))
    rows.append(metric_row("professors_unique", shares[group_column].nunique(dropna=True)))
    rows.append(metric_row("publications", publication_weights.size))
    rows.append(metric_row("publication_weight_sum_min", publication_weights.min()))
    rows.append(metric_row("publication_weight_sum_max", publication_weights.max()))
    rows.append(
        metric_row("publication_weight_sum_off_one", (~np.isclose(publication_weights, 1.0)).sum())
    )
    rows.append(metric_row("professor_share_sum_min", share_sums.min()))
    rows.append(metric_row("professor_share_sum_max", share_sums.max()))
    rows.append(metric_row("professor_share_sum_off_one", (~np.isclose(share_sums, 1.0)).sum()))
    rows.append(
        metric_row(
            "unclassified_publications",
            unclassified_publication_count(
                contributions,
                group_column=group_column,
                publication_column=publication_column,
            ),
        )
    )
    rows.append(metric_row("unclassified_rule", "No detailed share bucket is created."))
    return pd.DataFrame(rows)


def metric_row(metric: str, value: Any) -> dict[str, Any]:
    """Create a diagnostic metric/value record.

    :param metric: Diagnostic metric name.
    :param value: Metric value.
    :return: Metric record.
    """
    return {"metric": metric, "value": value}


def unclassified_publication_count(
    contributions: pd.DataFrame,
    *,
    group_column: str,
    publication_column: str,
) -> int:
    """Count unique group-publication records without a classified field.

    :param contributions: Publication-level field contributions.
    :param group_column: Column identifying an analysis group.
    :param publication_column: Column identifying a publication.
    :return: Number of unclassified publications.
    """
    if contributions.empty:
        return 0
    publications = contributions.drop_duplicates([group_column, publication_column])
    return int(publications["is_unclassified"].sum())


def field_share_columns(data: pd.DataFrame) -> list[str]:
    """Return field-share columns present in a data frame, in canonical order.

    :param data: Data frame to inspect.
    :return: Present field-share columns.
    """
    return [column for column in FIELD_SHARE_COLUMNS if column in data.columns]


def domain_share_columns(data: pd.DataFrame) -> list[str]:
    """Return domain-share columns present in a data frame, in canonical order.

    :param data: Data frame to inspect.
    :return: Present domain-share columns.
    """
    return [column for column in DOMAIN_SHARE_COLUMNS if column in data.columns]


def omitted_reference_field(data: pd.DataFrame) -> str:
    """Choose the field with the largest total share as regression reference.

    :param data: Data frame containing field shares.
    :return: Omitted reference column, or an empty string if none are present.
    """
    columns = field_share_columns(data)
    if not columns:
        return ""
    totals = data[columns].sum(axis=0).sort_values(ascending=False)
    return str(totals.index[0])


def omitted_reference_domain(data: pd.DataFrame) -> str:
    """Choose the domain with the largest total share as regression reference.

    :param data: Data frame containing domain shares.
    :return: Omitted reference column, or an empty string if none are present.
    """
    columns = domain_share_columns(data)
    if not columns:
        return ""
    totals = data[columns].sum(axis=0).sort_values(ascending=False)
    return str(totals.index[0])


def included_field_share_terms(
    data: pd.DataFrame, *, reference_field: str | None = None
) -> tuple[str, ...]:
    """Return field-share terms excluding the reference field.

    :param data: Data frame containing field shares.
    :param reference_field: Optional reference column; inferred when omitted.
    :return: Included field-share column names.
    """
    reference = reference_field or omitted_reference_field(data)
    return tuple(column for column in field_share_columns(data) if column != reference)


def included_domain_share_terms(
    data: pd.DataFrame, *, reference_domain: str | None = None
) -> tuple[str, ...]:
    """Return domain-share terms excluding the reference domain.

    :param data: Data frame containing domain shares.
    :param reference_domain: Optional reference column; inferred when omitted.
    :return: Included domain-share column names.
    """
    reference = reference_domain or omitted_reference_domain(data)
    return tuple(column for column in domain_share_columns(data) if column != reference)
