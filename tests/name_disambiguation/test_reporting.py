# Copyright © 2026 TU Wien

import json
from pathlib import Path

import pandas as pd

from acib.name_disambiguation.config import (
    NameDisambiguationConfig,
    NameDisambiguationOutputs,
)
from acib.name_disambiguation.output import write_all_outputs
from acib.name_disambiguation.reporting import render_json_report, render_text_report
from acib.name_disambiguation.service import disambiguate_batches


def test_reports_expose_canonical_names_and_representative_author_id(
    author_frame: pd.DataFrame,
) -> None:
    authors = author_frame.copy()
    authors.loc[0, "author_id"] = "57198158890"
    authors.loc[0, "orcid"] = "0000-0001-2345-6789"
    authors.loc[1, "orcid"] = "0000-0001-2345-6789"
    result = disambiguate_batches(authors)[0]
    json_report = render_json_report(
        result.decisions, result.clusters, result.final_selection, result.final_selection_scores
    )
    payload = json.loads(json_report)
    assert "left_canonical_name" in json_report
    assert "right_canonical_name" in json_report
    assert "canonical_name" in json_report
    assert "canonical_indexed_name" in json_report
    assert "disambiguated_cluster_id" in json_report
    text = render_text_report(
        result.decisions, result.clusters, result.final_selection, result.final_selection_scores
    )
    expected_id = str(result.clusters["disambiguated_cluster_id"].iloc[0])
    assert expected_id == "57198158890"
    assert f"disambiguated_cluster_id={expected_id}" in text
    assert payload["clusters"]


def test_text_report_formats_target_and_cluster_subject_areas() -> None:
    from acib.name_disambiguation.models import FinalAuthorSelection

    clusters = pd.DataFrame(
        [
            {
                "cluster_id": 1,
                "disambiguated_cluster_id": "a1",
                "author_id": "a1",
                "name": "Müller, Sandra",
                "indexed_name": "Müller S.",
                "publication_start": 2020,
                "publication_end": 2024,
                "subject_areas": ["MATH"],
            }
        ]
    )
    selection = FinalAuthorSelection(
        sc_id="12",
        status="manual_selection_required",
        selected_cluster_id=None,
        selected_disambiguated_cluster_id="",
        selected_author_ids=[],
        possible_clusters=[["a1"]],
        inferred_asjc_codes=[2603, 2604],
        scopus_asjc_codes=[],
        inferred_asjc_labels=[],
        reason="multiple clusters remain",
    )
    scores = pd.DataFrame(
        [
            {
                "cluster_id": 1,
                "disambiguated_cluster_id": "a1",
                "cluster_size": 1,
                "author_ids": ["a1"],
                "cluster_subject_codes": [2603, 2604],
                "cluster_level_two_subject_codes": [2600],
                "overlap_codes": [2600],
                "overlap_count": 1,
                "inferred_coverage": 1.0,
            }
        ]
    )
    report = render_text_report(pd.DataFrame(), clusters, selection, scores)
    assert "target_subject_areas=[2603 (Analysis), 2604 (Applied Mathematics)]" in report
    assert "target_level2_subject_areas=[2600 (General Mathematics)]" in report
    assert "cluster_subject_areas=[2603 (Analysis), 2604 (Applied Mathematics)]" in report
    assert "cluster_level2_subject_areas=[2600 (General Mathematics)]" in report
    assert "overlap_subject_areas=[2600 (General Mathematics)]" in report


def test_write_outputs_validates_and_writes_all_six_artifacts(
    tmp_path: Path,
    author_frame: pd.DataFrame,
) -> None:
    results = disambiguate_batches(author_frame)
    root = tmp_path / "disambiguation"
    outputs = NameDisambiguationOutputs(
        pair_decisions=root / "pairs.csv",
        clusters=root / "clusters.csv",
        final_author_selection=root / "selection.csv",
        final_author_selection_scores=root / "scores.csv",
        report_txt=root / "report.txt",
        report_jsonl=root / "report.jsonl",
    )
    config = NameDisambiguationConfig(
        matching_output=tmp_path / "matching.csv",
        article_author_file_path=tmp_path / "articles.csv",
        article_reference_file_path=tmp_path / "references.parquet",
        outputs=outputs,
    )
    write_all_outputs(results, config)
    assert all(path.is_file() for path in outputs.all_paths())
    assert "Author Name Disambiguation" in outputs.report_txt.read_text(encoding="utf-8")
    assert (
        json.loads(outputs.report_jsonl.read_text(encoding="utf-8").splitlines()[0])["sc_id"] == "1"
    )
