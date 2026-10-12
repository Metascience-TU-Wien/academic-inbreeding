# Copyright © 2026 TU Wien

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from acib.mobility.config import MobilityConfig, MobilityOutputs
from acib.mobility.workflow import run_mobility


def test_mobility_runs_independently_and_writes_only_configured_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    author_path = tmp_path / "selection.csv"
    metadata_path = tmp_path / "resolved.csv"
    article_path = tmp_path / "article.parquet"
    affiliation_path = tmp_path / "affiliations.gpkg"
    for path in (author_path, metadata_path, article_path, affiliation_path):
        path.touch()

    authors = pd.DataFrame(
        {
            "sc_id": ["1"],
            "status": ["assigned_single_candidate"],
            "selected_disambiguated_cluster_id": ["1001"],
            "selected_author_ids": [[1001]],
            "possible_clusters": [[]],
            "inferred_asjc_codes": [[]],
            "scopus_asjc_codes": [[]],
            "inferred_asjc_labels": [[]],
            "reason": ["single candidate"],
            "researcher_name": ["Example, Professor"],
            "selected_history": [{"1001": []}],
        }
    )
    metadata = pd.DataFrame(
        {
            "sc_id": [1],
            "author_id": [1001],
            "priority": [1],
            "eid": ["1"],
            "researcher_name": ["Example, Professor"],
            "author_name": ["Professor, Example"],
            "name": ["Professor, Example"],
            "indexed_name": ["Example P"],
            "initials": ["P"],
            "given_name": ["Professor"],
            "surname": ["Example"],
            "publication_range": ["2010-2020"],
            "name_variants": ["[]"],
            "subject_areas": ["[]"],
            "person_title": ["Prof."],
            "university_name": ["University of Vienna"],
        }
    )
    articles = pd.DataFrame(
        {
            "eid": ["paper-1", "paper-2"],
            "author_ids": ["1001", "1001"],
            "author_afids": ["60025988", "60025988"],
            "cover_date": ["2010-01-01", "2020-01-01"],
        }
    )
    affiliations = gpd.GeoDataFrame(
        {
            "affiliation_id": [60025988],
            "affiliation_id_parent": pd.Series([pd.NA], dtype="Int64"),
            "preferred_name": ["University of Vienna"],
            "type": ["education"],
            "address_part": ["Universitätsring 1"],
            "city": ["Vienna"],
            "country": ["Austria"],
        },
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    frames = {
        author_path: authors,
        metadata_path: metadata,
        article_path: articles,
    }
    monkeypatch.setattr(
        "acib.mobility.workflow._read_dataframe",
        lambda path, **_kwargs: frames[path].copy(),
    )
    monkeypatch.setattr("acib.mobility.workflow._read_geodataframe", lambda _path: affiliations)

    output_dir = tmp_path / "mobility-output"
    outputs = MobilityOutputs(directory=output_dir)
    config = MobilityConfig(
        authors=author_path,
        researcher_metadata=metadata_path,
        article_author=article_path,
        affiliations=affiliation_path,
        outputs=outputs,
        enable_university_affiliation_inbreeding_detection=True,
        enable_coaffiliated_professor_merge=True,
        first_affiliation_publication_count=1,
        max_workers=1,
    )

    run_mobility(config)

    assert all(path.is_file() for path in outputs.all_paths())
    assert not (output_dir / "professor_author_profiles.csv").exists()
    summary = pd.read_csv(outputs.author_inbreeding_proxy)
    assert summary.loc[0, "sc_id"] == 1
    assert summary.loc[0, "title"] == "Full Prof."
    articles_output = pd.read_csv(outputs.author_articles)
    assert articles_output["eid"].tolist() == ["paper-1", "paper-2"]
