# Copyright © 2026 TU Wien

"""End-to-end coverage for the configured name-disambiguation workflow."""

from ast import literal_eval
from pathlib import Path

import pandas as pd

from acib.constants import AUTHOR_ID, RESEARCHER_ID
from acib.name_disambiguation.config import (
    NameDisambiguationConfig,
    NameDisambiguationOutputs,
)
from acib.name_disambiguation.constants import DECISION_LABEL, DISAMBIGUATED_CLUSTER_ID
from acib.name_disambiguation.workflow import run_name_disambiguation


def test_workflow_reads_matcher_and_publication_inputs_and_writes_all_outputs(
    tmp_path: Path,
) -> None:
    """Run the stage from small CSVs through all six configured artifacts.

    :param tmp_path: Isolated temporary workspace supplied by pytest.
    """
    matching_path = tmp_path / "matching.csv"
    article_path = tmp_path / "article_authors.csv"
    reference_path = tmp_path / "references.csv"
    output_directory = tmp_path / "outputs"

    pd.DataFrame(
        [
            {
                RESEARCHER_ID: 1,
                "author_id": author_id,
                "priority": 1,
                "eid": eid,
                "researcher_name": "Müller, Thomas",
                "author_name": "Müller, Thomas",
                "name": "Müller, Thomas",
                "surname": "Müller",
                "given_name": "Thomas",
                "indexed_name": "Müller T",
                "initials": "T",
                "publication_range": "{2020,2024}",
                "name_variants": "[]",
                "subject_areas": '[{"area":"Software","code":1712}]',
                "orcid": "0000-0001-2345-6789",
                "person_email": "researcher@example.org",
                "inferred_asjc_codes": "[]",
                "organization": "Wirtschaft",
                "affiliation_id": 60018163,
                "affiliation_id_parent": parent_id,
                "person_title": "Prof.",
                "university_name": "TU Wien",
                "affiliation_ids_current": "[60018163]",
            }
            for author_id, eid, parent_id in (
                (10, "P1", 60018163),
                (20, "P2", pd.NA),
            )
        ]
    ).to_csv(matching_path, index=False)
    pd.DataFrame(
        [
            {"eid": "P1", "author_ids": "10", "subject_areas": "[]"},
            {"eid": "P2", "author_ids": "20", "subject_areas": "[]"},
        ]
    ).to_csv(article_path, index=False)
    pd.DataFrame(columns=["citing_eid", "cited_eid"]).to_csv(reference_path, index=False)

    outputs = NameDisambiguationOutputs(directory=output_directory)
    config = NameDisambiguationConfig(
        matching_output=matching_path,
        article_author_file_path=article_path,
        article_reference_file_path=reference_path,
        outputs=outputs,
        enable_reporting=False,
    )

    selections = run_name_disambiguation(config)

    assert selections["status"].tolist() == ["assigned_single_cluster"]
    pair_decisions = pd.read_csv(outputs.pair_decisions)
    clusters = pd.read_csv(outputs.clusters)
    assert pair_decisions[DECISION_LABEL].tolist() == ["same_author"]
    components = literal_eval(pair_decisions.iloc[0]["components"])
    component_labels = {component["label"] for component in components}
    assert "email_exact" not in component_labels
    assert "orcid_match" not in component_labels
    assert selections.iloc[0]["inferred_asjc_codes"] == [1400]
    assert clusters[DISAMBIGUATED_CLUSTER_ID].nunique() == 1
    assert clusters[AUTHOR_ID].astype(str).tolist() == ["10", "20"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.all_paths())
