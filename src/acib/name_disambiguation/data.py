# Copyright © 2026 TU Wien

"""DataFrame preparation helpers for disambiguation inputs."""

from __future__ import annotations

import pandas as pd

from acib.constants import EID, RESEARCHER_ID
from acib.name_disambiguation.constants import (
    CANONICAL_GIVEN_NAME,
    CANONICAL_INITIALS,
    CANONICAL_NAME,
    CANONICAL_SURNAME,
    SUBJECT_AREAS_PARSED,
)
from acib.name_disambiguation.parsing import prepare_author_frame

_PREPARED_AUTHOR_COLUMNS = {
    CANONICAL_NAME,
    CANONICAL_SURNAME,
    CANONICAL_GIVEN_NAME,
    CANONICAL_INITIALS,
    SUBJECT_AREAS_PARSED,
}


def prepare_author_candidates(author_df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize candidates and retain only ambiguous researcher batches.

    :param author_df: Matched researcher-author candidate rows.
    :return: Prepared rows belonging to researcher IDs with multiple candidates.
    """
    prepared = prepare_disambiguation_author_frame(author_df)
    if RESEARCHER_ID not in prepared.columns:
        return prepared.reset_index(drop=True)
    ambiguous = prepared.groupby(RESEARCHER_ID)[EID].transform("size").gt(1)
    return prepared.loc[ambiguous].reset_index(drop=True)


def prepare_disambiguation_author_frame(author_df: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical author frame, preserving already-prepared inputs.

    :param author_df: Matched author rows, possibly containing nullable numeric columns.
    :return: Canonical author rows with missing source values represented as empty strings.
    """
    if _PREPARED_AUTHOR_COLUMNS.issubset(author_df.columns):
        return author_df.copy().reset_index(drop=True)
    # Stringify before filling so nullable integers accept the empty-string sentinel.
    string_frame = author_df.astype("string").fillna("").astype(str)
    return prepare_author_frame(string_frame)
