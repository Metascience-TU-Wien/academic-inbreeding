# Copyright © 2026 TU Wien

"""Name matching algorithms and researcher-to-author orchestration."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd
import structlog
from fuzzywuzzy import fuzz

from acib.common.names import (
    compound_match,
    normalize_person_name,
    split_name,
    surname_match_keys,
)
from acib.constants import (
    AFFILIATION_ID,
    AUTHOR_ID,
    EID,
    GIVEN_NAME,
    RESEARCHER_ID,
    SURNAME,
    UNIVERSITY_NAME,
)
from acib.name_matching.constants import (
    _SPLIT_GIVEN_NAME_COLUMN,
    _SPLIT_SURNAME_COLUMN,
    AUTHOR_COLUMNS,
    AUTHOR_NAME,
    AUTHOR_NAME_PREFIX,
    MAPPING_ID,
    MATCH_PRIORITY,
    NAME,
    PARENT_AFFILIATION_COLUMNS,
    PERSON_FULLNAME,
    RESEARCHER_COLUMNS,
    RESEARCHER_GIVEN_NAME,
    RESEARCHER_NAME,
    RESEARCHER_SURNAME,
    SURNAME_AS_NAME_UNIVERSITIES,
)

_logger = structlog.getLogger(__name__)
_NORMALIZED_NAME_COLUMN = "norm_name"


def normalize_name(name: str) -> list[str]:
    """Tokenize a name for the fuzzy candidate guardrails."""
    name = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", name.lower())).strip()
    stopwords = {
        "prof",
        "dr",
        "ing",
        "dipl",
        "von",
        "der",
        "van",
        "de",
        "la",
        "und",
        "für",
        "applied",
        "mathematics",
        "rezensiert",
    }
    return [token for token in name.split() if len(token) >= 3 and token not in stopwords]


def token_set_ratio(left: str, right: str) -> float:
    """Return the legacy FuzzyWuzzy token-set score."""
    return float(fuzz.token_set_ratio(left, right))


def names_match(
    name_a: str,
    name_b: str,
    min_shared_tokens: int = 2,
    min_fuzzy_ratio: int = 90,
) -> tuple[bool, float]:
    """Apply shared-token, containment, and fuzzy score conditions.

    :param name_a: First candidate name.
    :param name_b: Second candidate name.
    :param min_shared_tokens: Minimum number of shared normalized tokens.
    :param min_fuzzy_ratio: Minimum final token-set similarity score.
    :return: Whether the names pass all matching conditions and their score.
    """
    if not name_a or not name_b:
        return False, 0
    tokens_a, tokens_b = set(normalize_name(name_a)), set(normalize_name(name_b))
    if not tokens_a or not tokens_b:
        return False, 0
    shared = tokens_a & tokens_b
    shorter = min(len(tokens_a), len(tokens_b))
    if len(shared) < min_shared_tokens or len(shared) / shorter < 0.8:
        return False, 0
    ratio = token_set_ratio(" ".join(tokens_a), " ".join(tokens_b))
    return ratio >= min_fuzzy_ratio, ratio


def find_fuzzy_matches(
    name: str,
    author_df: pd.DataFrame,
    name_col: str,
    min_ratio: float = 100,
    author_names: list[str] | None = None,
    author_tokens_by_row: list[set[str]] | None = None,
    author_token_index: dict[str, list[int]] | None = None,
) -> list[str]:
    """Return names passing the legacy guarded token-set matching rule.

    ``min_ratio`` and the optional blocking index are retained for API
    compatibility; the historical call path applies :func:`names_match`'s
    default score threshold.
    """
    del min_ratio
    if not name or not name.strip():
        return []
    if author_names is None or author_tokens_by_row is None or author_token_index is None:
        author_names = [getattr(row, name_col) for row in author_df.itertuples()]
        candidate_indices: range | list[int] = range(len(author_names))
    else:
        tokens = set(normalize_name(name))
        if not tokens:
            return []
        counts: Counter[int] = Counter()
        for token in tokens:
            counts.update(author_token_index.get(token, ()))
        candidate_indices = []
        for index, count in counts.items():
            tokens_for_author = author_tokens_by_row[index]
            if (
                tokens_for_author
                and count >= 2
                and count / min(len(tokens), len(tokens_for_author)) >= 0.8
            ):
                candidate_indices.append(index)
        candidate_indices.sort()
    matches: list[str] = []
    for index in candidate_indices:
        candidate = author_names[index]
        if candidate and candidate.strip() and names_match(name, " ".join(candidate.split()))[0]:
            matches.append(candidate)
    return matches


def fuzzy_name_merge(
    researcher_df: pd.DataFrame,
    author_df: pd.DataFrame,
    author_col: str = AUTHOR_NAME,
    researcher_col: str = RESEARCHER_NAME,
) -> pd.DataFrame:
    """Join researcher and author rows whose names pass guarded fuzzy matching.

    :param researcher_df: Researcher records.
    :param author_df: Author candidate records.
    :param author_col: Candidate-name column.
    :param researcher_col: Researcher-name column.
    :return: Merged rows for accepted fuzzy matches.
    """
    key = _NORMALIZED_NAME_COLUMN
    researcher, author = (
        researcher_df.reset_index(drop=True).copy(),
        author_df.reset_index(drop=True).copy(),
    )
    researcher[key] = researcher[researcher_col].str.lower().str.strip()
    author[key] = author[author_col].str.lower().str.strip()
    author_names = author[key].fillna("").astype(str).tolist()
    tokens_by_row = [set(normalize_name(name)) for name in author_names]
    token_index: dict[str, list[int]] = defaultdict(list)
    for index, tokens in enumerate(tokens_by_row):
        for token in tokens:
            token_index[token].append(index)
    researcher[key] = researcher[key].apply(
        lambda value: find_fuzzy_matches(
            value,
            author,
            key,
            author_names=author_names,
            author_tokens_by_row=tokens_by_row,
            author_token_index=token_index,
        )
    )
    researcher = researcher.explode(key).reset_index(drop=True)
    result = researcher.merge(author, on=key, how="inner", suffixes=("_researcher", "_author"))
    return result.drop(columns=[key])


def exact_name_merge(researcher_df: pd.DataFrame, author_df: pd.DataFrame) -> pd.DataFrame:
    """Merge names equal after lowercase and surrounding-whitespace normalization.

    :param researcher_df: Researcher records.
    :param author_df: Author candidate records.
    :return: Inner-joined exact matches.
    """
    researcher, author = researcher_df.copy(), author_df.copy()
    EXACT_NAME_COLUMN = "_exact"
    researcher[EXACT_NAME_COLUMN] = researcher[RESEARCHER_NAME].str.lower().str.strip()
    author[EXACT_NAME_COLUMN] = author[AUTHOR_NAME].str.lower().str.strip()
    return researcher.merge(author, on=EXACT_NAME_COLUMN, how="inner")


def _split_name_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty:
        frame[[_SPLIT_SURNAME_COLUMN, _SPLIT_GIVEN_NAME_COLUMN]] = None
    else:
        frame[[_SPLIT_SURNAME_COLUMN, _SPLIT_GIVEN_NAME_COLUMN]] = frame[column].apply(
            lambda value: pd.Series(split_name(value))
        )
    return frame


def containment_name_merge(researcher_df: pd.DataFrame, author_df: pd.DataFrame) -> pd.DataFrame:
    """Match equal surnames when one given-name token set contains the other.

    :param researcher_df: Researcher records.
    :param author_df: Author candidate records.
    :return: Rows satisfying the containment rule.
    """
    researcher = _split_name_column(researcher_df.copy(), RESEARCHER_NAME)
    author = _split_name_column(author_df.copy(), AUTHOR_NAME)
    candidates = researcher.merge(author, on=_SPLIT_SURNAME_COLUMN, how="inner")
    if candidates.empty:
        return candidates
    contained = candidates.apply(
        lambda row: _given_name_contained(
            row[f"{_SPLIT_GIVEN_NAME_COLUMN}_x"], row[f"{_SPLIT_GIVEN_NAME_COLUMN}_y"]
        ),
        axis=1,
    )
    return candidates.loc[contained]


def _given_name_contained(left: str, right: str) -> bool:
    left_tokens, right_tokens = set(left.split()), set(right.split())
    return bool(
        left_tokens
        and right_tokens
        and (left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens))
    )


def compound_name_merge(
    researcher_df: pd.DataFrame,
    author_df: pd.DataFrame,
    *,
    enable_given_name_aliases: bool = False,
) -> pd.DataFrame:
    """Match compound surnames, multi-part given names, and initials.

    :param researcher_df: Researcher records.
    :param author_df: Author candidate records.
    :param enable_given_name_aliases: Enable the source's opt-in alias rules.
    :return: Rows satisfying compound-name compatibility.
    """
    authors, researchers = author_df.copy(), researcher_df.copy()
    if authors.empty or researchers.empty:
        return authors.iloc[0:0].copy()
    SURNAME_TEMP_COLUMN = "_s"
    GIVEN_TEMP_COLUMN = "_g"
    SURNAME_TOKEN_TEMP_COLUMN = "_st"
    GIVEN_TOKEN_TEMP_COLUMN = "_g_split"
    authors[[SURNAME_TEMP_COLUMN, GIVEN_TEMP_COLUMN]] = authors[AUTHOR_NAME].apply(
        lambda value: pd.Series(split_name(value))
    )
    researchers[[SURNAME_TEMP_COLUMN, GIVEN_TEMP_COLUMN]] = researchers[RESEARCHER_NAME].apply(
        lambda value: pd.Series(split_name(value))
    )
    authors[SURNAME_TOKEN_TEMP_COLUMN] = authors[SURNAME_TEMP_COLUMN].apply(
        lambda value: list(surname_match_keys(value)) if isinstance(value, str) else []
    )
    researchers[SURNAME_TOKEN_TEMP_COLUMN] = researchers[SURNAME_TEMP_COLUMN].apply(
        lambda value: list(surname_match_keys(value)) if isinstance(value, str) else []
    )
    author_expanded = authors.explode(SURNAME_TOKEN_TEMP_COLUMN)
    researcher_expanded = researchers.explode(SURNAME_TOKEN_TEMP_COLUMN)
    swapped = author_expanded.copy()
    swapped[GIVEN_TOKEN_TEMP_COLUMN] = swapped[GIVEN_TEMP_COLUMN].str.split(" ")
    swapped = (
        swapped.explode(GIVEN_TOKEN_TEMP_COLUMN)
        .drop(columns=SURNAME_TOKEN_TEMP_COLUMN)
        .rename(columns={GIVEN_TOKEN_TEMP_COLUMN: SURNAME_TOKEN_TEMP_COLUMN})
    )
    author_expanded = pd.concat([author_expanded, swapped], ignore_index=True)
    candidates = researcher_expanded.merge(
        author_expanded, on=SURNAME_TOKEN_TEMP_COLUMN, how="inner", suffixes=("_r", "_a")
    )
    mask = candidates.apply(
        lambda row: compound_match(
            row[RESEARCHER_NAME],
            row[AUTHOR_NAME],
            enable_given_name_aliases=enable_given_name_aliases,
        ),
        axis=1,
    )
    matches = candidates.loc[mask] if not candidates.empty else candidates.copy()
    return matches.drop(
        columns=[column for column in matches.columns if column.startswith("_")]
    ).reset_index(drop=True)


def simple_name_merge(
    researcher_df: pd.DataFrame,
    author_df: pd.DataFrame,
    *,
    enable_given_name_aliases: bool = False,
) -> pd.DataFrame:
    """Combine exact, containment, fuzzy, and compound matches by legacy priority.

    :param researcher_df: Researcher records.
    :param author_df: Author candidate records.
    :param enable_given_name_aliases: Enable the source's opt-in alias rules.
    :return: Deduplicated matches, with the strongest strategy ranked first.
    """
    exact = exact_name_merge(researcher_df, author_df)
    containment = containment_name_merge(researcher_df, author_df)
    fuzzy = fuzzy_name_merge(researcher_df, author_df)
    compound = compound_name_merge(
        researcher_df, author_df, enable_given_name_aliases=enable_given_name_aliases
    )
    for frame, priority in ((exact, 1), (containment, 2), (fuzzy, 3), (compound, 4)):
        frame[MATCH_PRIORITY] = priority
    result = pd.concat([exact, containment, fuzzy, compound], ignore_index=True).loc[
        :, lambda frame: ~frame.columns.str.startswith("_")
    ]
    dedupe = [column for column in (RESEARCHER_ID, AUTHOR_ID, AFFILIATION_ID) if column in result]
    if dedupe:
        result = result.drop_duplicates(dedupe)
    entities = [column for column in (RESEARCHER_ID, EID) if column in result]
    result = result.sort_values(MATCH_PRIORITY)
    if entities:
        result = result.drop_duplicates(subset=entities)
    return result.sort_values(RESEARCHER_ID).reset_index(drop=True)


@dataclass(frozen=True)
class ResearcherMatchResult:
    """Matching results for matched, unmatched, and candidate researcher rows."""

    matched: pd.DataFrame
    possibilities: pd.DataFrame
    unmatched: pd.DataFrame

    @property
    def has_unmatched(self) -> bool:
        """Return whether any researcher was left unmatched."""
        return not self.unmatched.empty


def _preprocess_name_frame(frame: pd.DataFrame, *, researcher: bool) -> pd.DataFrame:
    """Construct stage-normalized comma-separated names from source fields."""
    result = frame.copy()
    if researcher:
        if PERSON_FULLNAME in result and NAME not in result:
            result[NAME] = result[PERSON_FULLNAME]
        if UNIVERSITY_NAME in result:
            special = result[UNIVERSITY_NAME].isin(SURNAME_AS_NAME_UNIVERSITIES)
            result.loc[special, [RESEARCHER_GIVEN_NAME, RESEARCHER_SURNAME]] = result.loc[
                special, [RESEARCHER_SURNAME, RESEARCHER_GIVEN_NAME]
            ].to_numpy()
        given, surname = RESEARCHER_GIVEN_NAME, RESEARCHER_SURNAME
        name_column = RESEARCHER_NAME
    else:
        given, surname, name_column = GIVEN_NAME, SURNAME, AUTHOR_NAME
    if name_column not in result and NAME in result:
        result[name_column] = result[NAME].map(_western_name_to_comma_format)
    if surname in result and given in result:
        result[name_column] = (
            result[surname].map(
                lambda value: normalize_person_name(str(value)) if not pd.isna(value) else ""
            )
            + ", "
            + result[given].map(
                lambda value: normalize_person_name(str(value)) if not pd.isna(value) else ""
            )
        ).str.strip(", ")
    result[name_column] = result[name_column].map(lambda value: _normalize_comma_name(value))
    return result.loc[result[name_column].ne("")].reset_index(drop=True)


def _western_name_to_comma_format(value: object) -> str:
    """Convert source display names to the legacy ``surname, given`` form."""
    if pd.isna(value) or not str(value).strip():
        return ""
    name = re.sub(r"\s+", " ", str(value).strip())
    name = re.sub(r"(?<=\.)(?=[A-ZÄÖÜ])", " ", name)
    if "," in name:
        surname, given = name.split(",", 1)
        return f"{surname.strip()}, {given.strip()}"
    tokens = name.split()
    if len(tokens) == 1:
        return f"{tokens[0]}, "
    particles = {"van", "von", "de", "der", "den", "la", "le", "di", "da", "del"}
    if tokens[0].lower() in particles:
        index = 1
        while index < len(tokens) and tokens[index].lower() in particles:
            index += 1
        if index < len(tokens) and tokens[index][0].isupper() and not tokens[index].endswith("."):
            index += 1
        return f"{' '.join(tokens[:index])}, {' '.join(tokens[index:])}"
    surname_tokens = [tokens[-1]]
    index = len(tokens) - 2
    while index >= 0 and tokens[index].lower() in particles:
        surname_tokens.insert(0, tokens[index])
        index -= 1
    return f"{' '.join(surname_tokens)}, {' '.join(tokens[:index + 1])}"


def _normalize_comma_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if "," not in text:
        return normalize_person_name(text)
    surname, given = text.split(",", 1)
    return f"{normalize_person_name(surname)}, {normalize_person_name(given)}".strip(", ")


def _possibilities(researchers: pd.DataFrame, authors: pd.DataFrame) -> pd.DataFrame:
    """Construct prefix-blocked author possibilities for each researcher."""
    frame = researchers.copy()
    frame[AUTHOR_NAME_PREFIX] = frame[RESEARCHER_NAME].str[:3]
    for column in (AUTHOR_ID, AUTHOR_NAME):
        frame[column] = frame[AUTHOR_NAME_PREFIX].apply(
            lambda prefix, column=column: authors.loc[
                authors[AUTHOR_NAME].str.startswith(prefix, na=False), column
            ].tolist()
        )
    frame = frame.explode([AUTHOR_ID, AUTHOR_NAME]).reset_index(drop=True)
    repeated_columns = frame.columns.difference([RESEARCHER_ID, AUTHOR_NAME, AUTHOR_ID])
    repeated = frame.groupby(RESEARCHER_ID).cumcount().ne(0)
    frame.loc[repeated, repeated_columns] = pd.NA
    return frame


def _safe_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate non-empty result frames, returning an empty frame if needed.

    :param frames: Per-organization result frames.
    :return: Their concatenated rows, or an empty frame when all are empty.
    """
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        return pd.DataFrame()
    return pd.concat(non_empty_frames, ignore_index=True)


def match_researchers(
    researcher_df: pd.DataFrame,
    author_df: pd.DataFrame,
    org_mapping: dict[str, list[int | str]] | None = None,
    *,
    enable_given_name_aliases: bool = False,
    enable_organisation_filter: bool = True,
    max_workers: int = 3,
) -> ResearcherMatchResult:
    """Match researcher names to source-author profiles.

    :param researcher_df: Researcher rows containing names and organization.
    :param author_df: Candidate author rows.
    :param org_mapping: Optional organization-to-affiliation identifiers.
    :param enable_given_name_aliases: Enable conservative given-name aliases.
    :param enable_organisation_filter: Restrict candidates to mapped organizations.
    :param max_workers: Retained pipeline setting; matching currently runs in-process.
    :return: Matched, candidate, and unmatched row sets.
    """
    del max_workers
    researchers = _preprocess_name_frame(researcher_df, researcher=True)
    authors = _preprocess_name_frame(author_df, researcher=False)
    if org_mapping and enable_organisation_filter:
        if AFFILIATION_ID in authors:
            parent_ids = pd.Series(pd.NA, index=authors.index)
            for column in PARENT_AFFILIATION_COLUMNS:
                if column in authors:
                    parent_ids = parent_ids.combine_first(authors[column])
            authors[MAPPING_ID] = parent_ids.combine_first(authors[AFFILIATION_ID])
        results = []
        for organization, mapped_ids in org_mapping.items():
            _logger.info("Matching researchers for organization", organization=organization)
            selected_researchers = (
                researchers.loc[researchers[UNIVERSITY_NAME].str.lower().eq(organization.lower())]
                if UNIVERSITY_NAME in researchers
                else researchers.iloc[0:0]
            )
            if selected_researchers.empty:
                _logger.info(
                    "Skipping organization without researchers",
                    organization=organization,
                )
                continue
            selected_authors = (
                authors.loc[authors[MAPPING_ID].isin(mapped_ids)]
                if MAPPING_ID in authors
                else authors.iloc[0:0]
            )
            selected_researchers = selected_researchers.loc[
                :, selected_researchers.columns.intersection(RESEARCHER_COLUMNS)
            ]
            selected_authors = selected_authors.loc[
                :, selected_authors.columns.intersection(AUTHOR_COLUMNS)
            ]
            result = _match_frames(
                selected_researchers, selected_authors, enable_given_name_aliases
            )
            _logger.info(
                "Finished organization matching",
                organization=organization,
                matched_researchers=len(result.matched),
                total_researchers=len(result.matched) + len(result.unmatched),
            )
            results.append(result)
        _logger.info("Aggregating organization match results", organization_count=len(results))
        matched = _safe_concat([result.matched for result in results])
        unmatched = _safe_concat([result.unmatched for result in results])
        possibilities = _safe_concat([result.possibilities for result in results])
        return ResearcherMatchResult(matched, possibilities, unmatched)
    researchers = researchers.loc[:, researchers.columns.intersection(RESEARCHER_COLUMNS)]
    authors = authors.loc[:, authors.columns.intersection(AUTHOR_COLUMNS)]
    result = _match_frames(researchers, authors, enable_given_name_aliases)
    _logger.info(
        "Finished matching",
        matched_researchers=len(result.matched),
        total_researchers=len(result.matched) + len(result.unmatched),
    )
    return result


def _match_frames(
    researchers: pd.DataFrame, authors: pd.DataFrame, aliases: bool
) -> ResearcherMatchResult:
    matched = simple_name_merge(researchers, authors, enable_given_name_aliases=aliases)
    matched_ids = matched[RESEARCHER_ID].dropna().unique() if RESEARCHER_ID in matched else []
    unmatched = researchers.loc[~researchers[RESEARCHER_ID].isin(matched_ids)]
    return ResearcherMatchResult(matched, _possibilities(researchers, authors), unmatched)
