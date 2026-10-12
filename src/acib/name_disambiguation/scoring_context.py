# Copyright © 2026 TU Wien

"""Context builders for publication evidence and first-name frequencies."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from acib.constants import AUTHOR_ID as AUTHOR_SCOPUS_ID_COLUMN
from acib.constants import EID as EID_COLUMN
from acib.constants import GIVEN_NAME
from acib.constants import SUBJECT_AREAS as SUBJECT_AREA_COLUMN
from acib.name_disambiguation.constants import (
    ARTICLE_AFFILIATION_CITY as ARTICLE_AFFILIATION_CITY_COLUMN,
)
from acib.name_disambiguation.constants import (
    ARTICLE_AUTHOR_NAMES,
)
from acib.name_disambiguation.constants import ARTICLE_CITY as CITY_COLUMN
from acib.name_disambiguation.constants import ARTICLE_COUNTRY as COUNTRY_COLUMN
from acib.name_disambiguation.constants import (
    ARTICLE_COUNTRY_CODE,
)
from acib.name_disambiguation.constants import (
    ARTICLE_FUND_NUMBER as ARTICLE_FUND_NO_COLUMN,
)
from acib.name_disambiguation.constants import (
    ARTICLE_GRANT_ID,
    ARTICLE_GRANT_NUMBER,
    ARTICLE_GRANT_NUMBERS,
    ARTICLE_JOURNAL,
    ARTICLE_PUBLICATION_CITY,
    ARTICLE_PUBLICATION_COUNTRY,
    ARTICLE_PUBLICATION_NAME,
    ARTICLE_SOURCE_TITLE,
    ARTICLE_SOURCE_TITLE_SHORT,
)
from acib.name_disambiguation.constants import AUTHOR_IDS as ARTICLE_AUTHOR_ID_COLUMN
from acib.name_disambiguation.constants import (
    CANONICAL_GIVEN_NAME,
    CITED_EID,
    CITING_EID,
)
from acib.name_disambiguation.constants import (
    ISO3_COUNTRY_CODE as ISO3_COUNTRY_CODE_COLUMN,
)

from .asjc import asjc_entry, level_two_asjc_code
from .models import AuthorRecord
from .parsing import author_from_row, parse_subject_areas
from .scoring_common import (
    canonical_first_name,
    extract_publication_field_values,
    normalize_country,
    normalize_identifier,
    normalize_text,
    parse_identifier_collection,
)


@dataclass(slots=True)
class PublicationEvidenceIndex:
    """Per-author and citation-edge sets used by publication scoring rules.

    :param author_articles: Article IDs keyed by author ID.
    :param author_cited_references: Cited article IDs keyed by author ID.
    :param citing_papers_by_cited_article: Reverse citation-edge index.
    :param author_journals: Journal names keyed by author ID.
    :param author_coauthor_ids: Coauthor IDs keyed by author ID.
    :param author_coauthor_names: Coauthor names keyed by author ID.
    :param author_publication_cities: Publication cities keyed by author ID.
    :param author_publication_countries: Publication countries keyed by author ID.
    :param author_grant_numbers: Grant identifiers keyed by author ID.
    :param author_article_subject_codes: Subject codes keyed by author ID.
    :param author_article_subject_labels: Subject descriptions keyed by author ID.
    """

    author_articles: dict[str, set[str]]
    author_cited_references: dict[str, set[str]]
    citing_papers_by_cited_article: dict[str, set[str]]
    author_journals: dict[str, set[str]]
    author_coauthor_ids: dict[str, set[str]]
    author_coauthor_names: dict[str, set[str]]
    author_publication_cities: dict[str, set[str]]
    author_publication_countries: dict[str, set[str]]
    author_grant_numbers: dict[str, set[str]]
    author_article_subject_codes: dict[str, set[int]]
    author_article_subject_labels: dict[str, dict[int, str]]


@dataclass(slots=True)
class FirstNameFrequencyIndex:
    """Corpus-derived frequency cutoff and common given-name set.

    :param counts: Frequency counts by normalized first name.
    :param general_names: Names considered common for scoring.
    :param threshold_count: Count cutoff used to create ``general_names``.
    """

    counts: dict[str, int]
    general_names: set[str]
    threshold_count: int


@dataclass(slots=True)
class ScoringContext:
    """Reusable publication and first-name indices for one scoring run.

    :param publication_index: Optional publication evidence lookup.
    :param first_name_index: Optional corpus frequency lookup.
    """

    publication_index: PublicationEvidenceIndex | None = None
    first_name_index: FirstNameFrequencyIndex | None = None


def build_scoring_context(
    authors: list[AuthorRecord] | None = None,
    article_author_df: pd.DataFrame | None = None,
    name_frequency_df: pd.DataFrame | None = None,
    *,
    article_reference_df: pd.DataFrame | None = None,
) -> ScoringContext:
    """Assemble reusable scoring indices from author and article corpora.

    :param authors: Candidate author records or a prepared author frame.
    :param article_author_df: Optional article-author evidence table.
    :param name_frequency_df: Optional broad author corpus for name frequency.
    :param article_reference_df: Optional citation-edge table.
    :return: Context shared by pair and cluster scoring.
    """
    if isinstance(authors, pd.DataFrame) and article_author_df is None:
        article_author_df = authors
        authors = None

    candidate_author_ids = candidate_author_ids_from_authors(authors)

    return ScoringContext(
        publication_index=build_publication_evidence_index(
            article_author_df,
            article_reference_df=article_reference_df,
            candidate_author_ids=candidate_author_ids,
        ),
        first_name_index=build_first_name_frequency_index(authors, name_frequency_df),
    )


def candidate_author_ids_from_authors(
    authors: pd.DataFrame | list[AuthorRecord] | None,
) -> set[str] | None:
    """Collect normalized Scopus IDs to restrict publication indexing.

    :param authors: Candidate author rows, parsed records, or ``None``.
    :return: Candidate ID set, or ``None`` when IDs are unavailable.
    """
    if authors is None:
        return None
    if isinstance(authors, pd.DataFrame):
        if AUTHOR_SCOPUS_ID_COLUMN not in authors.columns:
            return None
        return {
            normalize_identifier(value)
            for value in authors[AUTHOR_SCOPUS_ID_COLUMN].dropna().astype(str)
            if normalize_identifier(value)
        }
    return {author.author_id for author in authors if author.author_id}


def build_first_name_frequency_index(
    authors: pd.DataFrame | list[AuthorRecord] | None,
    name_frequency_df: pd.DataFrame | None,
) -> FirstNameFrequencyIndex | None:
    """Estimate which first names are general/common from a large corpus.

    :param authors: Candidate frame or parsed author records.
    :param name_frequency_df: Optional corpus with a given-name column.
    :return: Frequency index, or ``None`` when no names are available.
    """
    if name_frequency_df is not None and GIVEN_NAME in name_frequency_df.columns:
        first_names = name_frequency_df[GIVEN_NAME].dropna().map(canonical_first_name)
        first_names = first_names[first_names != ""]
        counts = first_names.value_counts().to_dict()
    elif isinstance(authors, pd.DataFrame):
        first_names = authors.get(CANONICAL_GIVEN_NAME, pd.Series(dtype=str)).map(
            canonical_first_name
        )
        first_names = first_names[first_names != ""]
        counts = first_names.value_counts().to_dict()
    else:
        counts = Counter[str]()

    if not counts and authors and not isinstance(authors, pd.DataFrame):
        for author in authors:
            if not isinstance(author, AuthorRecord):
                author = author_from_row(author)
            given_names = [
                author.canonical_given_name,
                *(variant.canonical_given_name for variant in author.name_variants),
            ]
            for given_name in given_names:
                first_name = canonical_first_name(given_name)
                if first_name:
                    counts[first_name] += 1

    if not counts:
        return None

    ranked_counts = sorted(counts.values(), reverse=True)
    percentile_index = max(0, math.ceil(len(ranked_counts) * 0.05) - 1)
    threshold_count = max(10, ranked_counts[percentile_index]) if ranked_counts else 10
    general_names = {name for name, count in counts.items() if count >= threshold_count}

    return FirstNameFrequencyIndex(
        counts=dict(counts),
        general_names=general_names,
        threshold_count=threshold_count,
    )


def build_publication_evidence_index(
    article_author_df: pd.DataFrame | None,
    *,
    article_reference_df: pd.DataFrame | None = None,
    candidate_author_ids: set[str] | None = None,
) -> PublicationEvidenceIndex | None:
    """Index article-level evidence used by Table 2 publication rules.

    :param article_author_df: Publication rows linking articles and authors.
    :param article_reference_df: Optional citing-to-cited article edges.
    :param candidate_author_ids: Optional IDs used to filter source rows.
    :return: Publication evidence index, or ``None`` when input evidence is absent.
    """
    if (
        article_author_df is None
        or article_author_df.empty
        or EID_COLUMN not in article_author_df.columns
    ):
        return None

    author_articles: dict[str, set[str]] = defaultdict(set)
    author_cited_references: dict[str, set[str]] = defaultdict(set)
    citing_papers_by_cited_article: dict[str, set[str]] = defaultdict(set)
    author_journals: dict[str, set[str]] = defaultdict(set)
    author_coauthor_ids: dict[str, set[str]] = defaultdict(set)
    author_coauthor_names: dict[str, set[str]] = defaultdict(set)
    author_publication_cities: dict[str, set[str]] = defaultdict(set)
    author_publication_countries: dict[str, set[str]] = defaultdict(set)
    author_grant_numbers: dict[str, set[str]] = defaultdict(set)
    author_article_subject_codes: dict[str, set[int]] = defaultdict(set)
    author_article_subject_labels: dict[str, dict[int, str]] = defaultdict(dict)

    author_id_column = next(
        (
            column
            for column in (ARTICLE_AUTHOR_ID_COLUMN, AUTHOR_SCOPUS_ID_COLUMN)
            if column in article_author_df.columns
        ),
        None,
    )
    if author_id_column is None:
        return None

    if candidate_author_ids:
        article_author_df = filter_article_rows_to_candidate_authors(
            article_author_df=article_author_df,
            author_id_column=author_id_column,
            candidate_author_ids=candidate_author_ids,
        )
        if article_author_df.empty:
            return None

    article_author_df = attach_article_references(
        article_author_df=article_author_df,
        article_reference_df=article_reference_df,
        author_id_column=author_id_column,
    )

    grouped_articles = article_author_df.groupby(EID_COLUMN, dropna=True, sort=False)
    for _, article_rows in grouped_articles:
        row_dict = article_rows.iloc[0].to_dict()
        source_eid = normalize_identifier(row_dict.get(EID_COLUMN))
        cited_eids: set[str] = set()
        if CITED_EID in article_rows.columns:
            for value in article_rows[CITED_EID].tolist():
                cited_eids.update(parse_identifier_collection(value))
        author_ids: set[str] = set()
        for value in article_rows[author_id_column].tolist():
            author_ids.update(parse_identifier_collection(value))
        target_author_ids = author_ids
        if candidate_author_ids:
            target_author_ids = author_ids & candidate_author_ids
        journals = extract_publication_field_values(
            row_dict,
            exact_keys=(
                ARTICLE_JOURNAL,
                ARTICLE_SOURCE_TITLE,
                ARTICLE_SOURCE_TITLE_SHORT,
                ARTICLE_PUBLICATION_NAME,
            ),
            contains=(ARTICLE_JOURNAL, ARTICLE_SOURCE_TITLE, ARTICLE_SOURCE_TITLE_SHORT),
            normalizer=normalize_text,
        )
        publication_cities = extract_publication_field_values(
            row_dict,
            exact_keys=(CITY_COLUMN, ARTICLE_PUBLICATION_CITY, ARTICLE_AFFILIATION_CITY_COLUMN),
            contains=(CITY_COLUMN,),
            normalizer=normalize_text,
        )
        publication_countries = extract_publication_field_values(
            row_dict,
            exact_keys=(
                COUNTRY_COLUMN,
                ARTICLE_COUNTRY_CODE,
                ARTICLE_PUBLICATION_COUNTRY,
                ISO3_COUNTRY_CODE_COLUMN,
            ),
            contains=(COUNTRY_COLUMN, "iso3"),
            normalizer=normalize_country,
        )
        grant_numbers = extract_publication_field_values(
            row_dict,
            exact_keys=(
                ARTICLE_GRANT_NUMBER,
                ARTICLE_GRANT_NUMBERS,
                ARTICLE_GRANT_ID,
                ARTICLE_FUND_NO_COLUMN,
            ),
            contains=("grant", "fund"),
            normalizer=normalize_identifier,
        )
        coauthor_names = extract_publication_field_values(
            row_dict,
            exact_keys=(ARTICLE_AUTHOR_NAMES,),
            contains=(ARTICLE_AUTHOR_NAMES,),
            normalizer=normalize_text,
        )
        article_subject_labels = general_article_subject_areas(row_dict.get(SUBJECT_AREA_COLUMN))
        article_subject_codes = set(article_subject_labels)

        for cited_eid in cited_eids:
            if source_eid and cited_eid:
                citing_papers_by_cited_article[cited_eid].add(source_eid)

        if not target_author_ids:
            continue

        for author_id in target_author_ids:
            if source_eid:
                author_articles[author_id].add(source_eid)
            author_cited_references[author_id].update(cited_eids)
            author_journals[author_id].update(journals)
            author_coauthor_ids[author_id].update(author_ids - {author_id})
            author_coauthor_names[author_id].update(coauthor_names)
            author_publication_cities[author_id].update(publication_cities)
            author_publication_countries[author_id].update(publication_countries)
            author_grant_numbers[author_id].update(grant_numbers)
            author_article_subject_codes[author_id].update(article_subject_codes)
            author_article_subject_labels[author_id].update(article_subject_labels)

    add_cited_side_reference_edges(
        article_reference_df=article_reference_df,
        cited_article_ids=set().union(*author_articles.values()) if author_articles else set(),
        citing_papers_by_cited_article=citing_papers_by_cited_article,
    )

    if not any(
        [
            author_articles,
            author_cited_references,
            citing_papers_by_cited_article,
            author_journals,
            author_coauthor_ids,
            author_coauthor_names,
            author_publication_cities,
            author_publication_countries,
            author_grant_numbers,
            author_article_subject_codes,
        ]
    ):
        return None

    return PublicationEvidenceIndex(
        author_articles=dict(author_articles),
        author_cited_references=dict(author_cited_references),
        citing_papers_by_cited_article=dict(citing_papers_by_cited_article),
        author_journals=dict(author_journals),
        author_coauthor_ids=dict(author_coauthor_ids),
        author_coauthor_names=dict(author_coauthor_names),
        author_publication_cities=dict(author_publication_cities),
        author_publication_countries=dict(author_publication_countries),
        author_grant_numbers=dict(author_grant_numbers),
        author_article_subject_codes=dict(author_article_subject_codes),
        author_article_subject_labels=dict(author_article_subject_labels),
    )


def filter_article_rows_to_candidate_authors(
    *,
    article_author_df: pd.DataFrame,
    author_id_column: str,
    candidate_author_ids: set[str],
) -> pd.DataFrame:
    """Keep publication rows containing at least one candidate author.

    :param article_author_df: Article-author table to filter.
    :param author_id_column: Column containing one or more author IDs.
    :param candidate_author_ids: IDs retained in the evidence index.
    :return: Filtered rows in their original order.
    """
    if not candidate_author_ids:
        return article_author_df
    mask = article_author_df[author_id_column].map(
        lambda value: bool(parse_identifier_collection(value) & candidate_author_ids)
    )
    return article_author_df.loc[mask].reset_index(drop=True)


def add_cited_side_reference_edges(
    *,
    article_reference_df: pd.DataFrame | None,
    cited_article_ids: set[str],
    citing_papers_by_cited_article: dict[str, set[str]],
) -> None:
    """Index external papers citing candidate articles for co-citation evidence.

    :param article_reference_df: Citation-edge table.
    :param cited_article_ids: Candidate-owned articles to search for.
    :param citing_papers_by_cited_article: Mutable citation index to update.
    """
    if (
        article_reference_df is None
        or article_reference_df.empty
        or not cited_article_ids
        or not {CITING_EID, CITED_EID}.issubset(article_reference_df.columns)
    ):
        return
    references = article_reference_df.loc[:, [CITING_EID, CITED_EID]].copy()
    references[CITING_EID] = normalized_identifier_series(references[CITING_EID])
    references[CITED_EID] = normalized_identifier_series(references[CITED_EID])
    references = references[
        references[CITING_EID].ne("") & references[CITED_EID].isin(cited_article_ids)
    ]
    for cited_eid, citing_eids in references.groupby(CITED_EID)[CITING_EID]:
        citing_papers_by_cited_article[cited_eid].update(citing_eids)


def general_article_subject_areas(value: object) -> dict[int, str]:
    """Parse article subject areas and collapse codes to parent categories.

    :param value: Serialized or structured subject-area value.
    :return: Parent-code to description mapping.
    """
    if value is None:
        return {}
    if isinstance(value, list):
        return general_article_subject_areas_from_entries(value)
    if isinstance(value, dict):
        return general_article_subject_areas_from_entries([value])
    try:
        if pd.isna(value):
            return {}
    except (TypeError, ValueError):
        pass

    result: dict[int, str] = {}
    for area in parse_subject_areas(str(value)):
        if area.code is None:
            continue
        parent_code = level_two_asjc_code(int(area.code))
        if parent_code is None:
            continue
        entry = asjc_entry(parent_code)
        if entry is not None:
            result[parent_code] = entry.description
    return dict(sorted(result.items()))


def general_article_subject_areas_from_entries(entries: list[object]) -> dict[int, str]:
    """Collapse subject-area entry objects to known parent categories.

    :param entries: Parsed article subject-area objects.
    :return: Parent-code to description mapping.
    """
    result: dict[int, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        code = parse_subject_area_code(entry.get("code"))
        if code is None:
            continue
        parent_code = level_two_asjc_code(code)
        if parent_code is None:
            continue
        taxonomy_entry = asjc_entry(parent_code)
        if taxonomy_entry is not None:
            result[parent_code] = taxonomy_entry.description
    return dict(sorted(result.items()))


def parse_subject_area_code(value: object) -> int | None:
    """Parse a finite integral ASJC code while excluding zero.

    :param value: Candidate code value.
    :return: Integral nonzero code, or ``None`` for invalid values.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        numeric_code = float(str(value).strip())
    except ValueError:
        return None
    if not numeric_code.is_integer():
        return None
    code = int(numeric_code)
    return code if code != 0 else None


def attach_article_references(
    *,
    article_author_df: pd.DataFrame,
    article_reference_df: pd.DataFrame | None,
    author_id_column: str,
) -> pd.DataFrame:
    """Attach cited references to author-article rows via `eid -> citing_eid`.

    The article-author table remains the source of author/article ownership. The
    reference table is only an edge list from citing article to cited article.

    :param article_author_df: Article ownership rows.
    :param article_reference_df: Optional citation edges.
    :param author_id_column: Author identifier column in the ownership table.
    :return: Ownership rows with cited-reference values attached.
    """
    article_author_df = article_author_df.copy()
    article_author_df[EID_COLUMN] = normalized_identifier_series(
        series=article_author_df[EID_COLUMN]
    )

    if article_reference_df is None or article_reference_df.empty:
        return article_author_df
    if not {CITING_EID, CITED_EID}.issubset(article_reference_df.columns):
        return article_author_df

    article_keys = article_author_df.loc[:, [EID_COLUMN, author_id_column]].dropna(
        subset=[EID_COLUMN]
    )
    if article_keys.empty:
        return article_author_df

    references = article_reference_df.loc[:, [CITING_EID, CITED_EID]].copy()
    references[CITING_EID] = normalized_identifier_series(references[CITING_EID])
    references[CITED_EID] = normalized_identifier_series(references[CITED_EID])
    references = references[(references[CITING_EID] != "") & (references[CITED_EID] != "")]
    references = references.drop_duplicates()
    if references.empty:
        return article_author_df

    reference_rows = article_keys.merge(
        references,
        left_on=EID_COLUMN,
        right_on=CITING_EID,
        how="inner",
        copy=False,
    )
    if reference_rows.empty:
        return article_author_df

    reference_rows = reference_rows.loc[:, [EID_COLUMN, author_id_column, CITED_EID]]
    reference_rows = reference_rows.drop_duplicates()

    return pd.concat([article_author_df, reference_rows], ignore_index=True, sort=False)


def normalized_identifier_series(series: pd.Series) -> pd.Series:
    """Normalize identifier values in a Series while preserving its index.

    :param series: Source identifier values.
    :return: Normalized string identifiers with empty missing values.
    """
    return series.map(normalize_identifier).fillna("")


def is_general_first_name(name: str, context: ScoringContext | None) -> bool:
    """Return whether a shared first name should receive the lower common-name score.

    :param name: Given-name token to check.
    :param context: Optional corpus frequency index.
    :return: Whether the token is classified as general/common.
    """
    canonical = canonical_first_name(name)
    if not canonical or context is None or context.first_name_index is None:
        return True
    return canonical in context.first_name_index.general_names
