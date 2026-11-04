# Copyright © 2026 TU Wien

"""Input validation and execution boundary for name disambiguation."""

from __future__ import annotations

import sys
from dataclasses import asdict

import pandas as pd

from acib.name_disambiguation.config import NameDisambiguationConfig
from acib.name_disambiguation.constants import (
    ORCID,
)
from acib.name_disambiguation.data import prepare_disambiguation_author_frame
from acib.name_disambiguation.input import (
    NameDisambiguationInputs,
    load_name_disambiguation_inputs,
    read_dataframe as _read_dataframe,
    validate_matching_artifact as _validate_matching_artifact,
)
from acib.name_disambiguation.output import (
    collect_name_disambiguation_result,
    save_name_disambiguation_results,
)
from acib.name_disambiguation.reporting import render_text_report
from acib.name_disambiguation.results import NameDisambiguationResult
from acib.name_disambiguation.service import disambiguate_batches
from acib.name_matching.constants import MATCH_PRIORITY
from acib.name_matching.workflow import add_inferred_asjc_columns
from acib.schemas.authors import DisambiguationAuthorSchema

_RESEARCHER_EMAIL_COLUMN = "person_email"

# Preserve the original workflow module's import paths for these public helpers.
read_dataframe = _read_dataframe
validate_matching_artifact = _validate_matching_artifact


def prepare_scoring_author_frame(matching: pd.DataFrame) -> pd.DataFrame:
    """Build the legacy-compatible scientific evidence view of matcher rows.

    The matching artifact retains corrected researcher metadata. The original
    matcher only carried researcher email on pre-resolved rows, omitted ORCID,
    and inferred missing ASJC codes from organization fields before matching.
    Apply those evidence semantics to a copy so the canonical matcher artifact
    remains unchanged.

    :param matching: Validated matcher-output rows, including preserved metadata.
    :return: A scoring frame with source-compatible researcher evidence.
    """
    evidence = add_inferred_asjc_columns(matching)
    if MATCH_PRIORITY in evidence and _RESEARCHER_EMAIL_COLUMN in evidence:
        matched_rows = evidence[MATCH_PRIORITY].ne(0)
        evidence.loc[matched_rows, _RESEARCHER_EMAIL_COLUMN] = ""
    if ORCID in evidence:
        evidence[ORCID] = ""
    return evidence


def compute_name_disambiguation(
    inputs: NameDisambiguationInputs,
    config: NameDisambiguationConfig,
) -> NameDisambiguationResult:
    """Compute disambiguation results from validated in-memory inputs.

    This function performs no file reads, writes, or console reporting. Construct
    inputs with :func:`prepare_name_disambiguation_inputs` or
    :func:`load_name_disambiguation_inputs`.

    :param inputs: Validated matcher and publication-evidence tables.
    :param config: Scoring thresholds and worker settings.
    :return: Complete per-batch decisions and collected artifact tables.
    """
    scoring_frame = prepare_scoring_author_frame(inputs.matched_researchers)
    authors = prepare_disambiguation_author_frame(scoring_frame)
    authors = DisambiguationAuthorSchema.validate(authors, lazy=True)
    batch_results = disambiguate_batches(
        authors,
        article_author_df=inputs.article_authors,
        article_reference_df=inputs.article_references,
        same_author_score_threshold=config.same_author_score_threshold,
        possible_author_score_threshold=config.possible_author_score_threshold,
        max_workers=config.max_workers,
    )
    return collect_name_disambiguation_result(batch_results)


def run_name_disambiguation(
    config: NameDisambiguationConfig,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run configured disambiguation and persist its legacy output artifacts.

    This compatibility wrapper retains the original DataFrame return, optional
    console reports, and configured persistence behavior.

    :param config: Resolved disambiguation paths and scoring options.
    :param overwrite: Permit replacing existing configured output artifacts.
    :return: Final selections as one row per processed researcher batch.
    :raises FileNotFoundError: If an upstream or external input is absent.
    :raises ValueError: If an input fails validation.
    :raises FileExistsError: If outputs exist and overwrite is disabled.
    """
    config.check_output_paths(overwrite=overwrite)
    inputs = load_name_disambiguation_inputs(config)
    result = compute_name_disambiguation(inputs, config)
    if config.enable_reporting:
        for batch_result in result.batch_results:
            sys.stdout.write(
                render_text_report(
                    batch_result.decisions,
                    batch_result.clusters,
                    batch_result.final_selection,
                    batch_result.final_selection_scores,
                )
            )
    save_name_disambiguation_results(result, config, overwrite=overwrite)
    return pd.DataFrame(
        [asdict(batch_result.final_selection) for batch_result in result.batch_results]
    )
