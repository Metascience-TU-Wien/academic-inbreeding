# Copyright © 2026 TU Wien

"""Load, analyze, and persist the standalone mobility stage."""

from __future__ import annotations

import time
from pathlib import Path

import structlog

from acib.mobility.config import MobilityConfig
from acib.mobility.inbreeding import (
    InbreedingResult,
    analyze_academic_inbreeding,
    write_inbreeding_outputs,
)
from acib.mobility.input import (
    MobilityInputs,
    enrich_authors_with_researcher_metadata,
    load_mobility_inputs,
)
from acib.mobility.professors import prepare_professor_author_profiles
from acib.mobility.schemas import (
    MobilityAuthorAffiliationsOutputSchema,
    MobilityAuthorArticlesOutputSchema,
    MobilityAuthorInbreedingOutputSchema,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_logger = structlog.getLogger(__name__)


def run_mobility(config: MobilityConfig) -> None:
    """Run input enrichment, professor preparation, analysis, and output writing.

    :param config: Resolved mobility inputs and analysis options.
    :raises FileNotFoundError: If a configured mobility input does not exist.
    :raises ImportError: If GeoPandas is unavailable for the GeoPackage input.
    :raises ValueError: If mobility algorithm parameters are invalid.
    """
    started_at = time.perf_counter()
    config.check_input_paths()
    source_repository = (_PROJECT_ROOT.parent / "name-matching-playground").resolve()
    source_outputs = [
        path.resolve()
        for path in config.outputs.all_paths()
        if path.resolve().is_relative_to(source_repository)
    ]
    if source_outputs:
        raise ValueError(
            "Refusing to write mobility outputs inside the read-only source repository: "
            + ", ".join(map(str, source_outputs))
        )
    _logger.info("Loading mobility inputs", author_file_path=str(config.authors))
    inputs = load_mobility_inputs(config)
    _logger.info(
        "Loaded mobility inputs",
        author_rows=len(inputs.authors),
        article_rows=len(inputs.articles),
        affiliation_rows=len(inputs.affiliations),
    )
    result = compute_mobility(inputs, config)
    _logger.info(
        "Writing mobility outputs",
        output_paths=[str(path) for path in config.outputs.all_paths()],
    )
    write_inbreeding_outputs(result, config.outputs)
    _logger.info(
        "Finished mobility run",
        output_paths=[str(path) for path in config.outputs.all_paths()],
        total_runtime_seconds=round(time.perf_counter() - started_at, 3),
    )


def compute_mobility(inputs: MobilityInputs, config: MobilityConfig) -> InbreedingResult:
    """Compute validated mobility results without writing files.

    :param inputs: Loaded and validated mobility tables.
    :param config: Scientific mobility options from the shared configuration.
    :return: Validated in-memory mobility output tables.
    :raises ValueError: If mobility algorithm parameters or output schemas are invalid.
    """
    authors = enrich_authors_with_researcher_metadata(
        authors=inputs.authors,
        researcher_metadata=inputs.researcher_metadata,
    )
    resolved_authors = authors
    if config.enable_coaffiliated_professor_merge:
        resolved_authors = prepare_professor_author_profiles(
            authors,
            professor_title_pattern=config.professor_title_pattern,
        )
        _logger.info(
            "Prepared merged professor author profiles",
            input_author_rows=len(authors),
            professor_profile_rows=len(resolved_authors),
        )
    _logger.info(
        "Starting academic inbreeding analysis",
        max_workers=config.max_workers,
        first_affiliation_publication_count=config.first_affiliation_publication_count,
        early_affiliation_method=config.early_affiliation_method,
        first_affiliation_year_count=config.first_affiliation_year_count,
        min_publication_history_years=config.min_publication_history_years,
    )
    result = analyze_academic_inbreeding(
        author_affiliations=inputs.affiliations,
        articles=inputs.articles,
        resolved_authors=resolved_authors,
        enable_university_affiliation_inbreeding_detection=(
            config.enable_university_affiliation_inbreeding_detection
        ),
        first_affiliation_publication_count=config.first_affiliation_publication_count,
        early_affiliation_method=config.early_affiliation_method,
        first_affiliation_year_count=config.first_affiliation_year_count,
        min_publication_history_years=config.min_publication_history_years,
        max_workers=config.max_workers,
    )
    output_frames = (
        (result.author_affiliations, MobilityAuthorAffiliationsOutputSchema),
        (result.author_summary, MobilityAuthorInbreedingOutputSchema),
        (result.author_articles, MobilityAuthorArticlesOutputSchema),
    )
    for frame, schema in output_frames:
        if not frame.empty or len(frame.columns) > 0:
            schema.validate(frame, lazy=True)
    return result
