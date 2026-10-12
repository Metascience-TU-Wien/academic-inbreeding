# Copyright © 2026 TU Wien

import pandas as pd
import pytest

from acib.name_disambiguation.output import (
    collect_final_selections,
    collect_pair_decisions,
)
from acib.name_disambiguation.service import disambiguate_batches


def test_disambiguation_keeps_batch_order_and_singleton_status(
    author_frame: pd.DataFrame,
) -> None:
    results = disambiguate_batches(author_frame)
    selections = collect_final_selections(results)
    assert selections["sc_id"].tolist() == ["1", "2"]
    assert selections.iloc[-1]["status"] == "assigned_single_candidate"
    assert selections.iloc[-1]["selected_author_ids"] == ["30"]
    assert (
        selections.iloc[-1]["reason"]
        == "only one candidate author exists for this sc_id; disambiguation not required"
    )
    assert len(collect_pair_decisions(results)) == 1


def test_empty_author_batch_is_rejected() -> None:
    with pytest.raises(ValueError):
        disambiguate_batches(pd.DataFrame())


def test_orcid_anchored_inline_batch_preserves_fixture_cluster_shape(
    author_frame: pd.DataFrame,
) -> None:
    # The original author.csv is unavailable. The ORCID anchor makes the source test's
    # expected three-row/two-cluster output deterministic on a small inline fixture.
    authors = author_frame.copy()
    authors["sc_id"] = "fixture-replacement"
    authors["orcid"] = ["0000-0001-2345-6789", "0000-0001-2345-6789", ""]
    results = disambiguate_batches(authors)
    result = results[0]
    assert len(result.authors) == 3
    assert len(result.decisions) == 3
    assert len(result.clusters) == 3
    assert result.clusters["cluster_id"].nunique() == 2
    assert sorted(result.clusters["cluster_size"].tolist()) == [1, 2, 2]
    assert (result.decisions["label"] == "same_author").sum() == 1


def test_sequential_and_parallel_workers_are_deterministic(
    author_frame: pd.DataFrame,
) -> None:
    sequential = disambiguate_batches(author_frame, max_workers=1)
    parallel = disambiguate_batches(author_frame, max_workers=2)
    pd.testing.assert_frame_equal(
        collect_final_selections(sequential), collect_final_selections(parallel)
    )
    pd.testing.assert_frame_equal(
        collect_pair_decisions(sequential), collect_pair_decisions(parallel)
    )
