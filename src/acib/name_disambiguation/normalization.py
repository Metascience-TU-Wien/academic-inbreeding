# Copyright © 2026 TU Wien

"""Name normalization with the exact legacy disambiguation behavior."""

from __future__ import annotations

import re
import unicodedata

try:
    import regex as _unicode_regex
except ModuleNotFoundError:  # pragma: no cover - optional acceleration
    _unicode_regex = None

_CHARACTER_REPLACEMENTS = {
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
_NORDIC_REPLACEMENTS = {
    "æ": "a",
    "ø": "o",
    "å": "a",
    "Æ": "A",
    "Ø": "O",
    "Å": "A",
}


def normalize_disambiguation_name(name: str) -> str:
    """Normalize a name in the same order as the original disambiguator.

    :param name: Source name text.
    :return: Lowercase transliterated name with legacy whitespace semantics.
    """
    if not name or not name.strip():
        return ""
    try:
        name = name.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    name = unicodedata.normalize("NFKD", name)
    for source, replacement in _CHARACTER_REPLACEMENTS.items():
        name = name.replace(source, replacement)
    for source, replacement in _NORDIC_REPLACEMENTS.items():
        name = name.replace(source, replacement)
    name = "".join(char for char in name if not unicodedata.combining(char))
    if _unicode_regex is not None:
        name = _unicode_regex.sub(r"\p{Pd}+", " ", name)
        name = _unicode_regex.sub(r"\s+", " ", name).strip()
    else:
        name = re.sub(r"[-\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]+", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
    name = "".join(
        char for char in name if unicodedata.category(char).startswith("L") or char.isspace()
    )
    return name.lower()
