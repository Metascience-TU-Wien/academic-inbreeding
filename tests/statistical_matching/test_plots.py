# Copyright © 2026 TU Wien

"""Stage-local matching plot writer coverage."""

from pathlib import Path

import pandas as pd

from acib.statistical_matching.plots import write_balance_plot, write_matching_plots


def _balance() -> pd.DataFrame:
    covariates = [
        "academic_age",
        "first_year",
        "title[Ao. Prof.]",
        "title[Ass. Prof.]",
        "title[FH-Prof.]",
        "title[Full Prof.]",
        "life_sci_share",
        "health_sci_share",
        "phys_sci_share",
        "soc_hum_share",
        "multi_area_share",
    ]
    return pd.DataFrame(
        [
            {
                "method": method,
                "covariate": covariate,
                "diff_unadjusted": 0.1,
                "diff_adjusted": 0.02,
                "abs_diff_adjusted": 0.02,
                "balanced": True,
            }
            for method in ("PSM", "IPTW", "CEM", "EB")
            for covariate in covariates
        ]
    )


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": method,
                "n_non_inbred": 50,
                "n_inbred": 50,
                "ess_non_inbred": 45.0,
            }
            for method in ("PSM", "IPTW", "CEM", "EB")
        ]
    )


def test_balance_plot_retains_reference_terms_all_methods_and_writes_configured_file(
    tmp_path: Path,
) -> None:
    data = _balance()
    assert {"title[Full Prof.]", "phys_sci_share"} <= set(data.covariate)
    path = write_balance_plot(data, _summary(), tmp_path, ("PSM", "IPTW", "CEM", "EB"), 0.10)
    assert path == tmp_path / "covariate_balance.png"
    assert path.is_file()


def test_balance_only_writer_does_not_touch_propensity_plots(tmp_path: Path) -> None:
    pooled = tmp_path / "propensity_score_overlap.png"
    method = tmp_path / "propensity_score_overlap_by_method.png"
    pooled.write_bytes(b"keep pooled")
    method.write_bytes(b"keep method")
    write_balance_plot(_balance(), _summary(), tmp_path, ("PSM", "IPTW", "CEM", "EB"), 0.10)
    assert pooled.read_bytes() == b"keep pooled"
    assert method.read_bytes() == b"keep method"


def test_complete_plot_writer_emits_propensity_plots(tmp_path: Path) -> None:
    propensity = pd.DataFrame(
        {
            "is_inbred": [False, False, True, True],
            "propensity_score": [0.12, 0.34, 0.55, 0.78],
        }
    )
    by_method = pd.concat(
        [propensity.assign(method=method, weights=1.0) for method in ("PSM", "IPTW", "CEM", "EB")],
        ignore_index=True,
    )
    paths = write_matching_plots(
        _balance(),
        _summary(),
        propensity,
        by_method,
        pd.DataFrame(),
        (),
        {},
        ("PSM", "IPTW", "CEM", "EB"),
        "is_inbred",
        0.10,
        tmp_path,
    )
    assert paths[1:] == (
        tmp_path / "propensity_score_overlap.png",
        tmp_path / "propensity_score_overlap_by_method.png",
    )
    assert all(path.is_file() for path in paths)
    assert len(by_method) == 16
