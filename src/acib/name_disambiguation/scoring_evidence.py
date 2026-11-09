# Copyright © 2026 TU Wien

"""Table 2 evidence rules derived from author and publication metadata."""

from __future__ import annotations

import ast
import math
import re
from typing import Any

from acib.constants import AFFILIATION_ID_CURRENT, AFFILIATION_IDS_CURRENT

from .asjc import describe_codes, level_two_asjc_codes
from .constants import (
    ARTICLE_AFFILIATION_CITY as ARTICLE_AFFILIATION_CITY_COLUMN,
    ARTICLE_AFFILIATION_COUNTRY as ARTICLE_AFFILIATION_COUNTRY_COLUMN,
    ARTICLE_CITY as CITY_COLUMN,
    ARTICLE_COUNTRY as COUNTRY_COLUMN,
    ARTICLE_COUNTRY_CODE,
    ARTICLE_GRANT_NUMBER,
    ARTICLE_GRANT_NUMBERS,
    ARTICLE_JOURNAL,
    ARTICLE_JOURNALS,
    ARTICLE_PUBLICATION_CITY,
    ARTICLE_PUBLICATION_COUNTRY,
    ARTICLE_SOURCE_TITLE,
    ARTICLE_SOURCE_TITLES,
    ISO3_COUNTRY_CODE as ISO3_COUNTRY_CODE_COLUMN,
    TEMPORAL_GAP_CONFLICT_LABEL,
    TEMPORAL_HARD_GAP_YEARS,
    TEMPORAL_HARD_SPAN_YEARS,
    TEMPORAL_SPAN_CONFLICT_LABEL,
)
from .models import AuthorRecord, ScoreComponent
from .scoring_common import (
    normalize_country,
    normalize_identifier,
    normalize_text,
    parse_collection_field,
    preview_values,
    score_from_bins,
)
from .scoring_context import PublicationEvidenceIndex, ScoringContext


def score_email_evidence(left: AuthorRecord, right: AuthorRecord) -> list[ScoreComponent]:
    """Score an exact email match if one is available for both profiles.

    :param left: First author profile.
    :param right: Second author profile.
    :return: Email-match evidence, or an empty list.
    """
    shared_emails = sorted(extract_emails(left) & extract_emails(right))
    if not shared_emails:
        return []
    return [
        ScoreComponent(
            "email_exact",
            100.0,
            f"matching email address: {preview_values(shared_emails)}",
        )
    ]


def score_author_linked_address(left: AuthorRecord, right: AuthorRecord) -> list[ScoreComponent]:
    """Score a linked author-address match on both city and country.

    :param left: First author profile.
    :param right: Second author profile.
    :return: Address evidence, or an empty list.
    """
    shared_countries = extract_author_countries(left) & extract_author_countries(right)
    shared_cities = extract_author_cities(left) & extract_author_cities(right)
    if not shared_countries or not shared_cities:
        return []
    return [
        ScoreComponent(
            "author_address_match",
            4.0,
            f"matching linked author address city/country: "
            f"{preview_values(shared_cities)} / {preview_values(shared_countries)}",
        )
    ]


def score_current_affiliation_id(left: AuthorRecord, right: AuthorRecord) -> list[ScoreComponent]:
    """Score matching current affiliation IDs, ignoring missing sentinels.

    :param left: First author profile.
    :param right: Second author profile.
    :return: Current-affiliation evidence, or an empty list.
    """
    shared_affiliation_ids = sorted(
        extract_current_affiliation_ids(left) & extract_current_affiliation_ids(right)
    )
    if not shared_affiliation_ids:
        return []
    return [
        ScoreComponent(
            "current_affiliation_id_match",
            4.0,
            f"matching current affiliation ID: {preview_values(shared_affiliation_ids)}",
        )
    ]


def score_publication_range_consistency(
    left: AuthorRecord, right: AuthorRecord
) -> list[ScoreComponent]:
    """Flag implausible temporal combinations before merging profiles.

    :param left: First author profile.
    :param right: Second author profile.
    :return: Hard temporal conflict components, if present.
    """
    left_range = normalized_publication_range(left)
    right_range = normalized_publication_range(right)
    if left_range is None or right_range is None:
        return []

    left_start, left_end = left_range
    right_start, right_end = right_range
    combined_start = min(left_start, right_start)
    combined_end = max(left_end, right_end)
    combined_span = combined_end - combined_start
    gap = publication_range_gap(left_start, left_end, right_start, right_end)

    components: list[ScoreComponent] = []
    range_description = (
        f"{left_start}-{left_end} vs {right_start}-{right_end}; "
        f"combined span {combined_span} years"
    )
    if gap > TEMPORAL_HARD_GAP_YEARS:
        components.append(
            ScoreComponent(
                TEMPORAL_GAP_CONFLICT_LABEL,
                0.0,
                f"non-overlapping publication ranges have a {gap}-year gap "
                f"({range_description})",
            )
        )
    if combined_span > TEMPORAL_HARD_SPAN_YEARS:
        components.append(
            ScoreComponent(
                TEMPORAL_SPAN_CONFLICT_LABEL,
                0.0,
                f"combined publication span exceeds {TEMPORAL_HARD_SPAN_YEARS} years "
                f"({range_description})",
            )
        )
    return components


def normalized_publication_range(author: AuthorRecord) -> tuple[int, int] | None:
    """Return normalized year bounds, filling a missing bound from the other.

    :param author: Author profile with parsed publication range.
    :return: Ordered year bounds, or ``None`` when both are absent.
    """
    start = normalized_publication_year(author.publication_range.start)
    end = normalized_publication_year(author.publication_range.end)
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return None
    return (min(start, end), max(start, end))


def normalized_publication_year(value: Any) -> int | None:
    """Normalize a finite, plausible publication year.

    :param value: Candidate year value.
    :return: Rounded year in the supported range, or ``None``.
    """
    if value is None:
        return None
    try:
        year = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(year):
        return None
    rounded = int(round(year))  # noqa: RUF046 - preserve legacy conversion
    if rounded < 1600 or rounded > 2200:
        return None
    return rounded


def publication_range_gap(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    """Calculate the gap between disjoint publication ranges.

    :param left_start: First range's start year.
    :param left_end: First range's end year.
    :param right_start: Second range's start year.
    :param right_end: Second range's end year.
    :return: Gap in years, or zero for overlapping ranges.
    """
    if left_end < right_start:
        return right_start - left_end
    if right_end < left_start:
        return left_start - right_end
    return 0


def score_shared_coauthors(
    left: AuthorRecord,
    right: AuthorRecord,
    *,
    context: ScoringContext | None = None,
) -> list[ScoreComponent]:
    """Score the number of shared coauthors between two profiles.

    :param left: First author profile.
    :param right: Second author profile.
    :param context: Optional publication index for shared coauthors.
    :return: Shared-coauthor score component, if any.
    """
    shared_coauthors = shared_coauthor_values(left, right, context)
    shared_count = len(shared_coauthors)
    if shared_count == 0:
        return []

    if shared_count == 1:
        weight = 4.0
    elif shared_count == 2:
        weight = 7.0
    else:
        weight = 10.0

    return [
        ScoreComponent(
            "shared_coauthors",
            weight,
            f"shared coauthors ({shared_count}): {preview_values(sorted(shared_coauthors))}",
        )
    ]


def score_grant_numbers(
    left: AuthorRecord,
    right: AuthorRecord,
    *,
    context: ScoringContext | None = None,
) -> list[ScoreComponent]:
    """Score grant identifiers shared between two profiles.

    :param left: First author profile.
    :param right: Second author profile.
    :param context: Optional publication evidence context.
    :return: Grant-match component, if any.
    """
    shared_grants = sorted(
        combined_grant_numbers(left, context) & combined_grant_numbers(right, context)
    )
    if not shared_grants:
        return []
    return [
        ScoreComponent(
            "grant_number_match",
            10.0,
            f"shared grant number(s): {preview_values(shared_grants)}",
        )
    ]


def score_publication_address(
    left: AuthorRecord,
    right: AuthorRecord,
    *,
    context: ScoringContext | None = None,
) -> list[ScoreComponent]:
    """Score a matching city and country found in publication metadata.

    :param left: First author profile.
    :param right: Second author profile.
    :param context: Optional publication evidence context.
    :return: Publication-address component, if any.
    """
    shared_countries = combined_publication_countries(
        left, context
    ) & combined_publication_countries(right, context)
    shared_cities = combined_publication_cities(left, context) & combined_publication_cities(
        right, context
    )
    if not shared_countries or not shared_cities:
        return []
    return [
        ScoreComponent(
            "publication_address_match",
            2.0,
            f"matching publication address city/country: "
            f"{preview_values(shared_cities)} / {preview_values(shared_countries)}",
        )
    ]


def score_subject_areas(
    left: AuthorRecord,
    right: AuthorRecord,
    context: ScoringContext | None = None,
) -> list[ScoreComponent]:
    """Score overlap in article and profile subject-area parent codes.

    :param left: First author profile.
    :param right: Second author profile.
    :param context: Optional article-level subject evidence.
    :return: One component for each supported subject overlap.
    """
    components: list[ScoreComponent] = []

    article_overlap = article_subject_codes(left, context) & article_subject_codes(right, context)
    if article_overlap:
        components.append(
            ScoreComponent(
                "article_subject_overlap",
                3.0,
                f"shared article parent subject area code(s): {describe_codes(article_overlap)}",
            )
        )

    profile_overlap = profile_parent_subject_codes(left) & profile_parent_subject_codes(right)
    if profile_overlap:
        components.append(
            ScoreComponent(
                "profile_subject_overlap",
                1.5,
                f"shared profile parent subject area code(s): {describe_codes(profile_overlap)}",
            )
        )

    return components


def article_subject_codes(author: AuthorRecord, context: ScoringContext | None) -> set[int]:
    """Return article-derived parent subject codes for an author.

    :param author: Author profile.
    :param context: Optional publication evidence index.
    :return: Article subject-area parent codes.
    """
    if context is None or context.publication_index is None:
        return set()
    return set(context.publication_index.author_article_subject_codes.get(author.author_id, set()))


def profile_parent_subject_codes(author: AuthorRecord) -> set[int]:
    """Collapse the author's Scopus subject codes to parent categories.

    :param author: Author profile.
    :return: Valid level-two parent codes.
    """
    return level_two_asjc_codes(codes=valid_subject_codes(author))


def score_journal_overlap(
    left: AuthorRecord,
    right: AuthorRecord,
    *,
    context: ScoringContext | None = None,
) -> list[ScoreComponent]:
    """Score overlap in normalized journal and source-title values.

    :param left: First author profile.
    :param right: Second author profile.
    :param context: Optional publication evidence context.
    :return: Journal-match component, if any.
    """
    shared_journals = sorted(combined_journals(left, context) & combined_journals(right, context))
    if not shared_journals:
        return []
    return [
        ScoreComponent(
            "journal_match",
            6.0,
            f"shared journal(s): {preview_values(shared_journals)}",
        )
    ]


def score_citation_relationships(
    left: AuthorRecord,
    right: AuthorRecord,
    publication_index: PublicationEvidenceIndex,
) -> list[ScoreComponent]:
    """Score self-citation, bibliographic coupling, and co-citation evidence.

    :param left: First author profile.
    :param right: Second author profile.
    :param publication_index: Precomputed author-publication evidence.
    :return: Citation-related score components.
    """
    components: list[ScoreComponent] = []

    left_articles = publication_index.author_articles.get(left.author_id, set())
    right_articles = publication_index.author_articles.get(right.author_id, set())
    left_references = publication_index.author_cited_references.get(left.author_id, set())
    right_references = publication_index.author_cited_references.get(right.author_id, set())

    if (left_references & right_articles) or (right_references & left_articles):
        components.append(
            ScoreComponent(
                "self_citation",
                10.0,
                "at least one publication of one profile cites a publication of the other",
            )
        )

    shared_references = left_references & right_references
    coupling_score = score_from_bins(len(shared_references), {1: 2.0, 2: 4.0, 3: 6.0, 4: 8.0}, 10.0)
    if coupling_score > 0:
        components.append(
            ScoreComponent(
                "bibliographic_coupling",
                coupling_score,
                f"shared cited references: {len(shared_references)}",
            )
        )

    left_citers = citing_papers_for_articles(
        left_articles, publication_index.citing_papers_by_cited_article
    )
    right_citers = citing_papers_for_articles(
        right_articles, publication_index.citing_papers_by_cited_article
    )
    shared_citers = third_party_shared_citers(
        left_articles=left_articles,
        right_articles=right_articles,
        left_citers=left_citers,
        right_citers=right_citers,
    )
    cocitation_score = score_from_bins(len(shared_citers), {1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}, 6.0)
    if cocitation_score > 0:
        components.append(
            ScoreComponent(
                "co_citation",
                cocitation_score,
                f"shared citing papers: {len(shared_citers)}",
            )
        )

    return components


def third_party_shared_citers(
    *,
    left_articles: set[str],
    right_articles: set[str],
    left_citers: set[str],
    right_citers: set[str],
) -> set[str]:
    """Return papers that cite both profiles, excluding their own papers.

    Co-citation is third-party evidence: paper X counts only if X cites at
    least one paper from the left profile and at least one paper from the right
    profile. Papers authored by either compared profile are handled separately
    as self-citation evidence and must not inflate co-citation.

    :param left_articles: Article IDs for the first profile.
    :param right_articles: Article IDs for the second profile.
    :param left_citers: Papers citing the first profile's articles.
    :param right_citers: Papers citing the second profile's articles.
    :return: Shared third-party citing paper IDs.
    """
    return (left_citers & right_citers) - left_articles - right_articles


def citing_papers_for_articles(
    article_ids: set[str],
    citing_papers_by_cited_article: dict[str, set[str]],
) -> set[str]:
    """Collect citing papers for a set of article IDs.

    :param article_ids: Article IDs whose citations are being queried.
    :param citing_papers_by_cited_article: Reverse citation index.
    :return: Unique citing paper IDs.
    """
    citing_papers: set[str] = set()
    for article_id in article_ids:
        citing_papers.update(citing_papers_by_cited_article.get(article_id, set()))
    return citing_papers


def valid_subject_codes(author: AuthorRecord) -> set[int]:
    """Return nonzero numeric subject codes for one author.

    :param author: Parsed author profile.
    :return: Valid Scopus subject-area codes.
    """
    return {
        int(area.code) for area in author.subject_areas if area.code is not None and area.code != 0
    }


def extract_emails(author: AuthorRecord) -> set[str]:
    """Extract normalized email values from raw profile fields.

    :param author: Parsed author profile.
    :return: Normalized email values.
    """
    return matching_raw_values(author, contains=("email",), normalizer=normalize_text)


def extract_author_countries(author: AuthorRecord) -> set[str]:
    """Extract country values linked to the author profile.

    :param author: Parsed author profile.
    :return: Normalized country codes or names.
    """
    return matching_raw_values(
        author,
        exact_keys=(
            ISO3_COUNTRY_CODE_COLUMN,
            COUNTRY_COLUMN,
            ARTICLE_COUNTRY_CODE,
            ARTICLE_AFFILIATION_COUNTRY_COLUMN,
        ),
        contains=(COUNTRY_COLUMN, "iso3"),
        exclude_contains=("publication",),
        normalizer=normalize_country,
    )


def extract_author_cities(author: AuthorRecord) -> set[str]:
    """Extract cities linked to the author profile.

    :param author: Parsed author profile.
    :return: Normalized city values.
    """
    return matching_raw_values(
        author,
        exact_keys=(CITY_COLUMN, ARTICLE_AFFILIATION_CITY_COLUMN),
        contains=(CITY_COLUMN,),
        exclude_contains=("publication",),
        normalizer=normalize_text,
    )


def extract_current_affiliation_ids(author: AuthorRecord) -> set[str]:
    """Extract normalized current affiliation IDs from a profile.

    :param author: Parsed author profile.
    :return: Valid current affiliation ID strings.
    """
    values = parse_current_affiliation_id_values(author.raw.get(AFFILIATION_IDS_CURRENT))
    if not values:
        values = parse_current_affiliation_id_values(author.raw.get(AFFILIATION_ID_CURRENT))
    return {
        normalized for value in values if (normalized := normalize_current_affiliation_id(value))
    }


def parse_current_affiliation_id_values(value: Any) -> list[Any]:
    """Parse one or many current-affiliation IDs from CSV-safe representations.

    :param value: Scalar, collection, or serialized collection.
    :return: Flattened affiliation identifier values.
    """
    if value is None:
        return []
    try:
        if bool(value != value):  # noqa: PLR0124 - legacy scalar missing-value test
            return []
    except (TypeError, ValueError):
        return []

    if isinstance(value, (list, tuple, set)):
        values: list[Any] = []
        for item in value:
            values.extend(parse_current_affiliation_id_values(item))
        return values

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>", "[]"}:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return parse_current_affiliation_id_values(parsed)
    if parsed is not None and not isinstance(parsed, (str, bytes)):
        return [parsed]

    stripped = text.strip("[]{}()")
    if any(separator in stripped for separator in [",", ";", "|", "\n"]):
        values = []
        for part in re.split(r"[,;|\n]", stripped):
            values.extend(parse_current_affiliation_id_values(part))
        return values
    return parse_collection_field(stripped)


def normalize_current_affiliation_id(value: Any) -> str:
    """Normalize one affiliation ID and discard missing sentinel values.

    :param value: Source affiliation identifier.
    :return: Normalized identifier, or empty text for missing values.
    """
    try:
        if value is None or bool(value is not None and value != value):  # noqa: PLR0124
            return ""
    except TypeError:
        return ""

    normalized = normalize_identifier(value)
    if not normalized:
        return ""
    if normalized.lower() in {"<na>", "-9999"}:
        return ""

    try:
        numeric = float(normalized)
    except ValueError:
        pass
    else:
        if not math.isfinite(numeric) or numeric == -9999:
            return ""
        if numeric.is_integer():
            return str(int(numeric))

    return normalized


def extract_publication_countries(author: AuthorRecord) -> set[str]:
    """Extract publication-linked country values from raw profile metadata.

    :param author: Parsed author profile.
    :return: Normalized country values.
    """
    return matching_raw_values(
        author,
        exact_keys=(ARTICLE_PUBLICATION_COUNTRY, "pub_country"),
        contains=(ARTICLE_PUBLICATION_COUNTRY, "pub_country"),
        normalizer=normalize_country,
    )


def extract_publication_cities(author: AuthorRecord) -> set[str]:
    """Extract publication-linked city values from raw profile metadata.

    :param author: Parsed author profile.
    :return: Normalized city values.
    """
    return matching_raw_values(
        author,
        exact_keys=(ARTICLE_PUBLICATION_CITY, "pub_city"),
        contains=(ARTICLE_PUBLICATION_CITY, "pub_city"),
        normalizer=normalize_text,
    )


def extract_coauthor_names(author: AuthorRecord) -> set[str]:
    """Extract normalized coauthor-name values from profile metadata.

    :param author: Parsed author profile.
    :return: Unique coauthor names.
    """
    return matching_raw_values(
        author,
        exact_keys=("coauthors", "coauthor_names", "coauthor_list"),
        contains=("coauthor",),
        exclude_contains=("count", "link", "id"),
        normalizer=normalize_text,
    )


def extract_coauthor_ids(author: AuthorRecord) -> set[str]:
    """Extract normalized coauthor IDs from profile metadata.

    :param author: Parsed author profile.
    :return: Unique coauthor identifiers.
    """
    return matching_raw_values(
        author,
        exact_keys=("coauthor_ids",),
        contains=("coauthor_id",),
        exclude_contains=("count", "link"),
        normalizer=normalize_identifier,
    )


def extract_grant_numbers(author: AuthorRecord) -> set[str]:
    """Extract normalized grant identifiers from profile metadata.

    :param author: Parsed author profile.
    :return: Unique grant identifiers.
    """
    return matching_raw_values(
        author,
        exact_keys=(ARTICLE_GRANT_NUMBER, ARTICLE_GRANT_NUMBERS, "grants"),
        contains=("grant",),
        normalizer=normalize_identifier,
    )


def extract_journals(author: AuthorRecord) -> set[str]:
    """Extract normalized journal and source-title values from a profile.

    :param author: Parsed author profile.
    :return: Unique journal names.
    """
    return matching_raw_values(
        author,
        exact_keys=(
            ARTICLE_JOURNAL,
            ARTICLE_JOURNALS,
            ARTICLE_SOURCE_TITLE,
            ARTICLE_SOURCE_TITLES,
        ),
        contains=("journal", "source_title"),
        normalizer=normalize_text,
    )


def combined_coauthors(author: AuthorRecord, context: ScoringContext | None) -> set[str]:
    """Combine profile-derived coauthor IDs and names.

    :param author: Parsed author profile.
    :param context: Optional corpus evidence index.
    :return: Combined coauthor values.
    """
    return combined_coauthor_ids(author, context) | combined_coauthor_names(author, context)


def shared_coauthor_values(
    left: AuthorRecord,
    right: AuthorRecord,
    context: ScoringContext | None,
) -> set[str]:
    """Return shared coauthor IDs, falling back to shared names.

    :param left: First author profile.
    :param right: Second author profile.
    :param context: Optional publication evidence index.
    :return: Shared coauthor identifiers or names.
    """
    shared_ids = combined_coauthor_ids(left, context) & combined_coauthor_ids(right, context)
    if shared_ids:
        return shared_ids
    return combined_coauthor_names(left, context) & combined_coauthor_names(right, context)


def combined_coauthor_ids(author: AuthorRecord, context: ScoringContext | None) -> set[str]:
    """Combine profile and publication-index coauthor identifiers.

    :param author: Parsed author profile.
    :param context: Optional publication evidence index.
    :return: Coauthor IDs excluding the author themself.
    """
    values = extract_coauthor_ids(author)
    if context and context.publication_index is not None:
        values |= context.publication_index.author_coauthor_ids.get(author.author_id, set())
    values.discard(normalize_identifier(author.author_id))
    return values


def combined_coauthor_names(author: AuthorRecord, context: ScoringContext | None) -> set[str]:
    """Combine profile and publication-index coauthor names.

    :param author: Parsed author profile.
    :param context: Optional publication evidence index.
    :return: Coauthor names excluding the profile's own identity tokens.
    """
    values = extract_coauthor_names(author)
    if context and context.publication_index is not None:
        values |= context.publication_index.author_coauthor_names.get(author.author_id, set())
    values -= self_identity_tokens(author)
    own_name_signatures = _self_identity_name_signatures(author)
    return {
        value for value in values if _normalized_name_signature(value) not in own_name_signatures
    }


def combined_grant_numbers(author: AuthorRecord, context: ScoringContext | None) -> set[str]:
    """Combine profile and publication-index grant identifiers.

    :param author: Parsed author profile.
    :param context: Optional publication evidence index.
    :return: Unique grant identifiers.
    """
    values = extract_grant_numbers(author)
    if context and context.publication_index is not None:
        values |= context.publication_index.author_grant_numbers.get(author.author_id, set())
    return values


def combined_publication_cities(author: AuthorRecord, context: ScoringContext | None) -> set[str]:
    """Combine profile and publication-index city values.

    :param author: Parsed author profile.
    :param context: Optional publication evidence index.
    :return: Unique normalized publication cities.
    """
    values = extract_publication_cities(author)
    if context and context.publication_index is not None:
        values |= context.publication_index.author_publication_cities.get(author.author_id, set())
    return values


def combined_publication_countries(
    author: AuthorRecord, context: ScoringContext | None
) -> set[str]:
    """Combine profile and publication-index country values.

    :param author: Parsed author profile.
    :param context: Optional publication evidence index.
    :return: Unique normalized publication countries.
    """
    values = extract_publication_countries(author)
    if context and context.publication_index is not None:
        values |= context.publication_index.author_publication_countries.get(
            author.author_id, set()
        )
    return values


def combined_journals(author: AuthorRecord, context: ScoringContext | None) -> set[str]:
    """Combine profile and publication-index journal values.

    :param author: Parsed author profile.
    :param context: Optional publication evidence index.
    :return: Unique normalized journal names.
    """
    values = extract_journals(author)
    if context and context.publication_index is not None:
        values |= context.publication_index.author_journals.get(author.author_id, set())
    return values


def matching_raw_values(
    author: AuthorRecord,
    *,
    exact_keys: tuple[str, ...] = (),
    contains: tuple[str, ...] = (),
    exclude_contains: tuple[str, ...] = (),
    normalizer: Any = None,
) -> set[str]:
    """Extract normalized values from selected raw author-profile fields.

    :param author: Parsed author profile with the raw source row.
    :param exact_keys: Exact source fields to include.
    :param contains: Source-key fragments to include.
    :param exclude_contains: Source-key fragments that disqualify a field.
    :param normalizer: Optional value normalizer; text normalization is the default.
    :return: Unique normalized values from matching fields.
    """
    values: set[str] = set()
    for key, value in author.raw.items():
        lowered_key = key.lower()
        if lowered_key in exact_keys or any(token in lowered_key for token in contains):
            if any(token in lowered_key for token in exclude_contains):
                continue
            for item in parse_collection_field(value):
                normalized = normalizer(item) if normalizer is not None else normalize_text(item)
                if normalized:
                    values.add(normalized)
    return values


def self_identity_tokens(author: AuthorRecord) -> set[str]:
    """Build normalized names and IDs used to exclude self-coauthor matches.

    :param author: Parsed author profile.
    :return: Nonempty identity tokens for the author and variants.
    """
    tokens = {
        normalize_text(author.author_id),
        normalize_text(author.name),
        normalize_text(author.canonical_name),
        normalize_text(author.indexed_name),
    }
    for variant in author.name_variants:
        tokens.update(
            {
                normalize_text(f"{variant.surname}, {variant.given_name}"),
                normalize_text(variant.canonical_full_name),
                normalize_text(variant.indexed_name),
            }
        )
    return {token for token in tokens if token}


def _self_identity_name_signatures(author: AuthorRecord) -> set[frozenset[str]]:
    """Build order-independent signatures for the author's name forms.

    Publication author names may use given-name-first order while profile names
    use surname-first order. Comparing normalized token sets excludes the same
    person in either order without treating a partial name overlap as identity.

    :param author: Parsed author profile.
    :return: Nonempty name-token signatures for the author and variants.
    """
    names = {author.name, author.canonical_name, author.indexed_name}
    for variant in author.name_variants:
        names.update(
            {
                f"{variant.surname}, {variant.given_name}",
                variant.canonical_full_name,
                variant.indexed_name,
            }
        )
    return {signature for name in names if (signature := _normalized_name_signature(name))}


def _normalized_name_signature(value: str) -> frozenset[str]:
    """Return normalized name tokens without depending on their order.

    :param value: Name in any supported word order.
    :return: Normalized tokens with word order ignored.
    """
    return frozenset(normalize_text(value).split())
