# Copyright © 2026 TU Wien

"""Shared person-name normalization and compatibility rules.

These primitives are intentionally independent of pipeline stages and tabular
schemas so matching and disambiguation can use the same name semantics.
"""

from __future__ import annotations

import re
import unicodedata

try:
    import regex
except ModuleNotFoundError:  # pragma: no cover - optional acceleration
    regex = None

_FUSED_GIVEN_NAME_PREFIXES = {"maria": "maria", "mari": "maria"}
_FUSED_GIVEN_NAME_SUFFIXES = {
    "carmen": ("carmen",),
    "cruz": ("cruz",),
    "delcarmen": ("del", "carmen"),
    "delacruz": ("de", "la", "cruz"),
    "luz": ("luz",),
    "paz": ("paz",),
}
_GIVEN_NAME_ALIASES = {"gertrude": {"gerti"}, "gerti": {"gertrude"}}


def normalize_person_name(name: str) -> str:
    """Normalize a name using the pipeline's transliteration rules.

    :param name: Raw person-name component.
    :return: Lowercase normalized text with punctuation and diacritics removed.
    """
    if not name or not name.strip():
        return ""
    try:
        name = name.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    name = unicodedata.normalize("NFKD", name)
    replacements = {
        "ä": "a",
        "ö": "o",
        "ü": "u",
        "Ä": "A",
        "Ö": "O",
        "Ü": "U",
        "ß": "ss",
        "ć": "c",
        "Ş": "S",
        "ae": "a",
        "oe": "o",
        "ue": "u",
        "Ae": "A",
        "Oe": "O",
        "Ue": "U",
    }
    for source, replacement in replacements.items():
        name = name.replace(source, replacement)
    for source, replacement in (
        ("æ", "a"),
        ("ø", "o"),
        ("å", "a"),
        ("Æ", "A"),
        ("Ø", "O"),
        ("Å", "A"),
    ):
        name = name.replace(source, replacement)
    name = "".join(character for character in name if not unicodedata.combining(character))
    if regex is not None:
        name = regex.sub(r"\p{Pd}+", " ", name)
        name = regex.sub(r"\s+", " ", name)
    else:
        name = re.sub(r"[-\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]+", " ", name)
        name = re.sub(r"\s+", " ", name)
    name = "".join(
        character
        for character in name
        if unicodedata.category(character).startswith("L") or character.isspace()
    )
    return name.lower()


def split_name(name: str, separator: str = ",") -> tuple[str | None, str | None]:
    """Split a normalized ``surname, given`` name.

    :param name: Full name in comma-separated surname-first form.
    :param separator: Delimiter used between surname and given name.
    :return: Lowercase surname and given name, or ``(None, None)`` if unsplittable.
    """
    if not isinstance(name, str) or not name or "," not in name:
        return None, None
    surname, given = name.split(sep=separator, maxsplit=1)
    return surname.strip().lower(), given.strip().lower()


def surname_tokens(surname: str) -> set[str]:
    """Return separator-delimited surname tokens."""
    return set(re.split(r"[-\s]+", surname))


def _fused_surname_form(surname: str) -> str:
    return "".join(token for token in re.split(r"[-\s]+", surname) if token)


def _is_exact_fused_surname_match(left: str, right: str) -> bool:
    fused_left = _fused_surname_form(left)
    fused_right = _fused_surname_form(right)
    if not fused_left or fused_left != fused_right:
        return False
    count_left = len([token for token in re.split(r"[-\s]+", left) if token])
    count_right = len([token for token in re.split(r"[-\s]+", right) if token])
    return count_left > 1 or count_right > 1


def surname_match_keys(surname: str) -> set[str]:
    """Return conservative tokens for candidate surname blocking.

    :param surname: Raw surname.
    :return: Normalized surname tokens and, for compounds, the fused form.
    """
    normalized = normalize_person_name(surname)
    if not normalized:
        return set()
    keys = {token for token in surname_tokens(normalized) if token}
    if len(keys) > 1:
        keys.add(_fused_surname_form(normalized))
    return keys


def surname_compatible(left: str, right: str) -> bool:
    """Apply the pipeline's exact and subset surname compatibility rules."""
    if not left or not right:
        return False
    normalized_left = normalize_person_name(left)
    normalized_right = normalize_person_name(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True
    if _is_exact_fused_surname_match(normalized_left, normalized_right):
        return True
    tokens_left = surname_tokens(normalized_left)
    tokens_right = surname_tokens(normalized_right)
    return tokens_left.issubset(tokens_right) or tokens_right.issubset(tokens_left)


def given_tokens(given_name: str) -> set[str]:
    return {token.rstrip(".") for token in re.split(r"[-\s]+", given_name) if token}


def normalize_given_tokens(given_name: str) -> tuple[set[str], set[str]]:
    """Return full given-name tokens and initials, including joined forms."""
    tokens = given_tokens(given_name)
    full = {token for token in tokens if len(token) > 1}
    initials = {token[0] for token in tokens if len(token) == 1}
    if len(full) > 1:
        full.add("".join(sorted(full)))
    return full, initials


def split_fused_given_name(token: str) -> tuple[str, ...] | None:
    """Expand supported high-confidence fused given names."""
    for prefix, canonical_prefix in sorted(
        _FUSED_GIVEN_NAME_PREFIXES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if token.startswith(prefix):
            suffix_tokens = _FUSED_GIVEN_NAME_SUFFIXES.get(token[len(prefix) :])
            if suffix_tokens:
                return (canonical_prefix, *suffix_tokens)
    return None


def expand_given_name_variants(given_name: str) -> set[str]:
    """Return conservative opt-in alias and fused-name variants."""
    tokens = given_tokens(given_name)
    if not tokens:
        return set()
    variants = set(tokens)
    for token in tuple(tokens):
        variants.update(_GIVEN_NAME_ALIASES.get(token, set()))
        expanded = split_fused_given_name(token)
        if expanded:
            variants.update(expanded)
    expanded_joined = split_fused_given_name("".join(tokens))
    if expanded_joined:
        variants.update(expanded_joined)
    return variants


def given_name_compatible(
    left: str, right: str, *, enable_given_name_aliases: bool = False
) -> bool:
    """Compare given names using full-token, initial, and optional alias rules."""
    if not left or not right:
        return False
    full_left, initials_left = normalize_given_tokens(left)
    full_right, initials_right = normalize_given_tokens(right)
    if not (full_left or initials_left) or not (full_right or initials_right):
        return False
    if full_left & full_right:
        return True
    if initials_left and full_right and initials_left.issubset({token[0] for token in full_right}):
        return True
    if initials_right and full_left and initials_right.issubset({token[0] for token in full_left}):
        return True
    if not enable_given_name_aliases:
        return False
    alias_left = full_left | expand_given_name_variants(left)
    alias_right = full_right | expand_given_name_variants(right)
    return bool(
        alias_left & alias_right
        or initials_left
        and alias_right
        and initials_left.issubset({token[0] for token in alias_right})
        or initials_right
        and alias_left
        and initials_right.issubset({token[0] for token in alias_left})
    )


def incompatible_full_given_names(left: str, right: str) -> bool:
    """Identify disjoint fully spelled given names, ignoring initials."""
    full_left = {token for token in given_tokens(left) if len(token) > 1}
    full_right = {token for token in given_tokens(right) if len(token) > 1}
    if len(full_left) > 1:
        full_left.add("".join(sorted(full_left)))
    if len(full_right) > 1:
        full_right.add("".join(sorted(full_right)))
    return bool(full_left and full_right and full_left.isdisjoint(full_right))


def compound_match(left: str, right: str, *, enable_given_name_aliases: bool = False) -> bool:
    """Apply compound surname, given-name, and swapped-name matching rules."""
    surname_left, given_left = split_name(left)
    surname_right, given_right = split_name(right)
    if not all((surname_left, given_left, surname_right, given_right)):
        return False
    assert surname_left is not None and given_left is not None
    assert surname_right is not None and given_right is not None
    candidates = (
        (surname_left, given_left, surname_right, given_right),
        (given_left, surname_left, surname_right, given_right),
        (given_right, surname_right, surname_left, given_left),
    )
    for (
        candidate_surname_left,
        candidate_given_left,
        candidate_surname_right,
        candidate_given_right,
    ) in candidates:
        if not surname_compatible(candidate_surname_left, candidate_surname_right):
            continue
        if (
            enable_given_name_aliases
            or not incompatible_full_given_names(candidate_given_left, candidate_given_right)
        ) and given_name_compatible(
            candidate_given_left,
            candidate_given_right,
            enable_given_name_aliases=enable_given_name_aliases,
        ):
            return True
    return False
