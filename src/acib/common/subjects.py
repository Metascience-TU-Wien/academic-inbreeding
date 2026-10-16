# Copyright © 2026 TU Wien

"""Infer ASJC subject evidence from organization and faculty names."""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable

ASJC_PARENT_LABELS = {
    1100: "Agricultural and Biological Sciences",
    1200: "Arts and Humanities",
    1300: "Biochemistry, Genetics and Molecular Biology",
    1400: "Business, Management and Accounting",
    1600: "Chemistry",
    1700: "Computer Science",
    1800: "Decision Sciences",
    2000: "Economics, Econometrics and Finance",
    2100: "Energy",
    2200: "Engineering",
    2300: "Environmental Science",
    2500: "Materials Science",
    2600: "Mathematics",
    2700: "Medicine",
    2800: "Neuroscience",
    2900: "Nursing",
    3000: "Pharmacology, Toxicology and Pharmaceutics",
    3100: "Physics and Astronomy",
    3200: "Psychology",
    3300: "Social Sciences",
}


@dataclass(frozen=True, slots=True)
class OrganizationRule:
    keywords: tuple[str, ...]
    codes: tuple[int, ...]
    require_all: bool = False


ORGANIZATION_RULES: tuple[OrganizationRule, ...] = (
    # FH Salzburg, Kuchl Campus: sustainability-focused design/engineering
    # department covering renewable materials, construction/renovation,
    # circular design, and circular economy. This is not generic visual design.
    OrganizationRule(
        ("design", "green", "engineering"),
        (1107, 2105, 2205, 2215, 2305, 2311, 2500),
        require_all=True,
    ),
    OrganizationRule(
        ("kuchl", "sustainability"),
        (1107, 2105, 2205, 2215, 2305, 2311, 2500),
        require_all=True,
    ),
    # Broad business faculty context. Do not infer economics/finance from the
    # faculty label alone, because that weakens downstream disambiguation by
    # making too many candidate clusters ASJC-compatible.
    OrganizationRule(("fakultat", "wirtschaft"), (1400,), require_all=True),
    OrganizationRule(("faculty", "business"), (1400, 2000), require_all=True),
    OrganizationRule(("school", "business"), (1400, 2000), require_all=True),
    OrganizationRule(("wirtschaft",), (1400,)),
    OrganizationRule(("business",), (1400,)),
    OrganizationRule(("betriebswirtschaft",), (1400,)),
    OrganizationRule(("volkswirtschaft",), (2000,)),
    OrganizationRule(("economics",), (2000,)),
    OrganizationRule(("econometrics",), (2002,)),
    OrganizationRule(("finance",), (2000, 2003)),
    OrganizationRule(("finanz",), (2000, 2003)),
    OrganizationRule(("bank",), (2000, 2003)),
    OrganizationRule(("accounting",), (1402,)),
    OrganizationRule(("rechnungswesen",), (1402,)),
    OrganizationRule(("unternehmensrechnung",), (1402,)),
    OrganizationRule(("controlling",), (1402,)),
    OrganizationRule(("management",), (1400, 1408)),
    OrganizationRule(("leadership",), (1400, 1407, 1408)),
    OrganizationRule(("human", "resource"), (1407,), require_all=True),
    OrganizationRule(("marketing",), (1406,)),
    OrganizationRule(("marktforschung",), (1406,)),
    OrganizationRule(("e commerce",), (1400, 1406)),
    OrganizationRule(("entrepreneurship",), (1400,)),
    OrganizationRule(("innovation",), (1400, 1405)),
    OrganizationRule(("recht",), (3308,)),
    OrganizationRule(("law",), (3308,)),
    OrganizationRule(("steuerrecht",), (3308,)),
    OrganizationRule(("tax",), (2000, 3308)),
    OrganizationRule(("nachhaltigkeit",), (2300,)),
    OrganizationRule(("sustainability",), (2300,)),
    OrganizationRule(("lebensmittelwissenschaft",), (1100,)),
    OrganizationRule(("food", "science"), (1100,), require_all=True),
    OrganizationRule(("personliche", "kompetenzentwicklung"), (1400, 1407), require_all=True),
    OrganizationRule(("personal", "development"), (1400, 1407), require_all=True),
    # Existing broad rules retained from the original mapping.
    OrganizationRule(("theologie",), (1200,)),
    OrganizationRule(("religion",), (1200,)),
    OrganizationRule(("philosophie",), (1200,)),
    OrganizationRule(("geschichte",), (1200,)),
    OrganizationRule(("germanistik",), (1200,)),
    OrganizationRule(("anglistik",), (1200,)),
    OrganizationRule(("musik",), (1200,)),
    OrganizationRule(("kunst",), (1200,)),
    OrganizationRule(("architektur",), (1200, 2200)),
    OrganizationRule(("soziologie",), (3300,)),
    OrganizationRule(("politikwissenschaft",), (3300,)),
    OrganizationRule(("kommunikationswissenschaft",), (3300,)),
    OrganizationRule(("padagog",), (3300,)),
    OrganizationRule(("psychologie",), (3200,)),
    OrganizationRule(("medizin",), (2700,)),
    OrganizationRule(("klinik",), (2700,)),
    OrganizationRule(("chirurgie",), (2700,)),
    OrganizationRule(("radiologie",), (2700,)),
    OrganizationRule(("pathologie",), (2700,)),
    OrganizationRule(("biologie",), (1100,)),
    OrganizationRule(("biotechnologie",), (1300,)),
    OrganizationRule(("mikrobiologie",), (1100,)),
    OrganizationRule(("chemie",), (1600,)),
    OrganizationRule(("physik",), (3100,)),
    OrganizationRule(("mathematik",), (2600,)),
    OrganizationRule(("statistik",), (2600,)),
    OrganizationRule(("informatik",), (1700,)),
    OrganizationRule(("software",), (1700,)),
    OrganizationRule(("artificial", "intelligence"), (1700,), require_all=True),
    OrganizationRule(("maschinenbau",), (2200,)),
    OrganizationRule(("mechatronik",), (2200,)),
    OrganizationRule(("elektrotechnik",), (2200,)),
    OrganizationRule(("werkstoff",), (2500,)),
    OrganizationRule(("verfahrenstechnik",), (2200,)),
    OrganizationRule(("erdwissenschaft",), (1900,)),
    OrganizationRule(("geographie",), (1900,)),
    OrganizationRule(("klima",), (1900,)),
    OrganizationRule(("umwelt",), (2300,)),
    OrganizationRule(("waldbau",), (1100,)),
    OrganizationRule(("pflanzenbau",), (1100,)),
    OrganizationRule(("landtechnik",), (1100, 2200)),
    OrganizationRule(("nutztier",), (1100,)),
)


def map_institute_to_subjects(inst_name: Any) -> list[str]:
    """Return broad Scopus subject labels for an organization value.

    Kept for compatibility with older exploratory code. Newer code should use
    `map_organization_to_asjc_codes` when numeric ASJC evidence is needed.
    """
    codes = map_organization_to_asjc_codes(inst_name)
    labels = {ASJC_PARENT_LABELS.get(parent_code(code)) for code in codes}
    return sorted(label for label in labels if label)


def map_organization_to_asjc_codes(value: Any) -> list[int]:
    """Map robustly parsed organization/faculty information to ASJC codes."""
    names = organization_names(value)
    codes: set[int] = set()
    normalized_names = [normalize_organization_name(name) for name in names]
    combined = " | ".join(normalized_names)

    for rule in ORGANIZATION_RULES:
        if rule_matches(rule, combined):
            codes.update(rule.codes)

    # Use ordered hierarchy context: if a row has a business faculty plus a
    # narrower institute, keep the broad business faculty evidence as well.
    if has_wirtschaft_faculty(normalized_names):
        codes.add(1400)

    return sorted(codes)


def organization_names(value: Any) -> list[str]:
    """Extract organization names from JSON, Python literals, lists, or strings."""
    parsed = parse_literal(value)
    names = list(extract_names(parsed))
    if names:
        return deduplicate(names)
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text and text.lower() not in {"nan", "none", "null"} else []


def parse_literal(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return text


def extract_names(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        if "value" in value:
            yield from extract_names(value["value"])
        elif "name" in value:
            yield str(value["name"])
        else:
            for item in value.values():
                yield from extract_names(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from extract_names(item)
    elif isinstance(value, str):
        text = value.strip()
        if text:
            yield text


def normalize_organization_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower().replace("&", " and ")
    normalized = normalized.replace("ß", "ss")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def rule_matches(rule: OrganizationRule, normalized_text: str) -> bool:
    if rule.require_all:
        return all(keyword_matches(keyword, normalized_text) for keyword in rule.keywords)
    return any(keyword_matches(keyword, normalized_text) for keyword in rule.keywords)


def keyword_matches(keyword: str, normalized_text: str) -> bool:
    normalized_keyword = normalize_organization_name(keyword)
    if not normalized_keyword:
        return False
    if normalized_keyword in normalized_text:
        return True
    keyword_tokens = normalized_keyword.split()
    text_tokens = normalized_text.split()
    if len(keyword_tokens) != 1:
        return False
    return any(similar_token(keyword_tokens[0], token) for token in text_tokens)


def similar_token(expected: str, actual: str) -> bool:
    if abs(len(expected) - len(actual)) > 2:
        return False
    return SequenceMatcher(a=expected, b=actual).ratio() >= 0.86


def has_wirtschaft_faculty(normalized_names: list[str]) -> bool:
    return any(
        keyword_matches("fakultat", name) and keyword_matches("wirtschaft", name)
        for name in normalized_names
    )


def parent_code(code: int) -> int:
    return (code // 100) * 100 if code >= 1000 else code


def deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result
