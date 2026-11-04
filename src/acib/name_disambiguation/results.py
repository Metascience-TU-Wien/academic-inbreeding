# Copyright © 2026 TU Wien

"""Typed in-memory inputs and results for name disambiguation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from acib.name_disambiguation.models import BatchDisambiguationResult


@dataclass(frozen=True, slots=True)
class NameDisambiguationInputs:
    """Validated input tables for researcher-author disambiguation.

    :param matched_researchers: Producer-owned matching rows and their metadata.
    :param article_authors: Publication-author evidence used during scoring.
    :param article_references: Citation edges used during scoring.
    """

    matched_researchers: pd.DataFrame
    article_authors: pd.DataFrame
    article_references: pd.DataFrame


@dataclass(frozen=True, slots=True)
class NameDisambiguationResult:
    """Complete tabular and per-batch results from disambiguation.

    :param batch_results: Per-researcher decisions, clusters, selections, and scores.
    :param pair_decisions: Collected pair-level decision artifact table.
    :param clusters: Collected author-cluster membership table.
    :param final_author_selection: Collected researcher-selection table.
    :param final_author_selection_scores: Collected cluster-score diagnostics.
    """

    batch_results: list[BatchDisambiguationResult]
    pair_decisions: pd.DataFrame
    clusters: pd.DataFrame
    final_author_selection: pd.DataFrame
    final_author_selection_scores: pd.DataFrame
