# Copyright © 2026 TU Wien

"""Load, analyze, and persist the standalone mobility stage."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import structlog

from acib.mobility.config import MobilityConfig
from acib.mobility.inbreeding import (
    analyze_academic_inbreeding,
    unique_affiliations,
    write_inbreeding_outputs,
)
from acib.mobility.input import enrich_authors_with_researcher_metadata
from acib.mobility.professors import prepare_professor_author_profiles
from acib.mobility.schemas import (
    MobilityAffiliationSchema,
    MobilityArticleAuthorSchema,
    MobilityAuthorAffiliationsOutputSchema,
    MobilityAuthorArticlesOutputSchema,
    MobilityAuthorInbreedingOutputSchema,
    MobilityUniqueAffiliationsOutputSchema,
)
from acib.schemas.matching import MatchingOutputSchema
from acib.schemas.researchers import ResearcherDisambiguationSelectionSchema

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_logger = structlog.getLogger(__name__)
_RESEARCHER_METADATA_STRING_DTYPES = {
    "person_functions": "string",
    "institute_link": "string",
    "person_location_link": "string",
    "orcid": "string",
}


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
    _logger.info(
        "Loading mobility inputs",
        author_file_path=str(config.authors),
        researcher_metadata_file_path=(
            str(config.researcher_metadata) if config.researcher_metadata is not None else None
        ),
        article_author_file_path=str(config.article_author),
        author_affiliations_file_path=str(config.affiliations),
    )
    authors = ResearcherDisambiguationSelectionSchema.validate(
        _read_dataframe(config.authors), lazy=True
    )
    articles = MobilityArticleAuthorSchema.validate(
        _read_dataframe(config.article_author), lazy=True
    )
    affiliations = _read_geodataframe(config.affiliations)
    geometry_column = affiliations.geometry.name
    affiliation_attributes = pd.DataFrame(
        affiliations.drop(columns=[geometry_column], errors="ignore")
    )
    MobilityAffiliationSchema.validate(affiliation_attributes, lazy=True)
    authors = enrich_authors_with_researcher_metadata(
        authors=authors,
        researcher_metadata=MatchingOutputSchema.validate(
            _read_dataframe(
                config.researcher_metadata,
                dtype=_RESEARCHER_METADATA_STRING_DTYPES,
            ),
            lazy=True,
        ),
    )
    _logger.info(
        "Loaded mobility inputs",
        author_rows=len(authors),
        article_rows=len(articles),
        affiliation_rows=len(affiliations),
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
        author_affiliations=affiliations,
        articles=articles,
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
    unique_affiliation_frame = unique_affiliations(author_affiliations=result.author_affiliations)
    if not unique_affiliation_frame.empty or len(unique_affiliation_frame.columns) > 0:
        MobilityUniqueAffiliationsOutputSchema.validate(unique_affiliation_frame, lazy=True)
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


def _read_dataframe(path: Path, *, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    """Read CSV or Parquet input using source suffix-based dispatch.

    :param path: Existing tabular input file.
    :param dtype: Optional CSV column dtypes for mixed-format metadata.
    :return: Loaded input dataframe.
    """
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False, dtype=dtype)


def _read_geodataframe(path: Path) -> pd.DataFrame:
    """Read the configured GeoPackage without eagerly requiring GeoPandas.

    :param path: Existing GeoPackage file.
    :return: Geospatial affiliation rows.
    :raises ImportError: If GeoPandas is not installed.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("geopandas is required to read mobility affiliation files") from exc
    return gpd.read_file(path)
