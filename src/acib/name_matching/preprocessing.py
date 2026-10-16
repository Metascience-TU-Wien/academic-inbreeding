# Copyright © 2026 TU Wien

"""Source-compatible person-name preparation for name matching."""

from __future__ import annotations

import re

import pandas as pd

from acib.constants import EID, GIVEN_NAME, SURNAME
from acib.name_matching.constants import INITIALS, IS_ALTERNATIVE, NAME, NAME_VARIANTS

_INITIAL_FIX = re.compile(r"(?<=\.)(?=[A-ZÄÖÜ])")
_SURNAME_PARTICLES = {"van", "von", "de", "der", "den", "la", "le", "di", "da", "del"}
_IS_ALTERNATIVE = IS_ALTERNATIVE


def split_western_name(name: str) -> tuple[str, str]:
    """Split a display name using the original pipeline's parsing rules.

    :param name: Unprocessed display name.
    :return: Surname and given-name components.
    """
    if not name or not name.strip():
        return "", ""
    name = re.sub(r"\s+", " ", _INITIAL_FIX.sub(" ", name.strip()))
    if "," in name:
        surname, given = name.split(",", 1)
        return surname.strip(), given.strip()
    tokens = name.split()
    if len(tokens) == 1:
        return tokens[0], ""
    if tokens[0].lower() in _SURNAME_PARTICLES:
        index = 1
        while index < len(tokens) and tokens[index].lower() in _SURNAME_PARTICLES:
            index += 1
        if index < len(tokens) and tokens[index][0].isupper() and not tokens[index].endswith("."):
            index += 1
        return " ".join(tokens[:index]), " ".join(tokens[index:])
    surname_tokens = [tokens[-1]]
    index = len(tokens) - 2
    while index >= 0 and tokens[index].lower() in _SURNAME_PARTICLES:
        surname_tokens.insert(0, tokens[index])
        index -= 1
    return " ".join(surname_tokens), " ".join(tokens[: index + 1])


def _is_initials(value: object) -> bool:
    tokens = re.split(r"[.\s\-]+", str(value))
    letters = ["".join(character for character in token if character.isalpha()) for token in tokens]
    letters = [token for token in letters if token]
    return bool(letters) and all(len(token) == 1 for token in letters)


def _resolve_given_names(parsed: pd.Series, structured: pd.Series | None) -> pd.Series:
    if structured is None:
        return parsed
    result = parsed.copy()
    structured_text = structured.map(lambda value: "" if pd.isna(value) else str(value).strip())
    replace = parsed.map(_is_initials) & structured_text.ne("")
    full = structured_text.map(_is_initials).eq(False)
    mask = replace & full
    result.loc[mask] = structured_text.loc[mask]
    return result


def format_names(frame: pd.DataFrame, *, use_name_variants: bool = False) -> pd.DataFrame:
    """Format display names using the original matching pipeline's semantics.

    :param frame: Records containing ``name`` and optionally structured name fields.
    :param use_name_variants: Whether to expand and clean alternate-name values.
    :return: Records with normalized ``name``, ``surname``, ``given_name``, and ``initials``.
    """
    result = frame.copy()
    if GIVEN_NAME in result:
        structured = result[GIVEN_NAME].map(
            lambda value: "" if pd.isna(value) else str(value).strip()
        )
    else:
        structured = None
    result[_IS_ALTERNATIVE] = False
    if use_name_variants and NAME_VARIANTS in result:
        variants = result.copy()
        variants[NAME_VARIANTS] = variants[NAME_VARIANTS].map(
            lambda value: value if isinstance(value, list) else []
        )
        variants = variants.explode(NAME_VARIANTS).drop(columns=[NAME], errors="ignore")
        variants = variants.rename(columns={NAME_VARIANTS: NAME})
        variants[_IS_ALTERNATIVE] = True
        result = pd.concat([result, variants], ignore_index=True)
    parts = result[NAME].astype(str).map(split_western_name)
    result[SURNAME] = parts.map(lambda pair: pair[0])
    parsed_given = parts.map(lambda pair: pair[1])
    result[GIVEN_NAME] = _resolve_given_names(parsed_given, structured)
    valid = result[SURNAME].str.len().gt(0) & result[GIVEN_NAME].str.len().gt(0)
    result = result.loc[valid].copy()
    result[INITIALS] = result[GIVEN_NAME].map(_is_initials)
    if use_name_variants:
        result[INITIALS] = result[GIVEN_NAME].map(_is_initials)
    result[NAME] = result[SURNAME] + ", " + result[GIVEN_NAME]
    if use_name_variants and EID in result:
        result = result.loc[~(result[_IS_ALTERNATIVE] & result[INITIALS])]
        result = result.groupby(EID, group_keys=False, sort=False).apply(
            lambda group: (
                group.loc[~((~group[INITIALS]).any() & ~group[_IS_ALTERNATIVE] & group[INITIALS])]
                if (~group[INITIALS]).any()
                else group
            ),
            include_groups=True,
        )
    return result.reset_index(drop=True)
