# Copyright © 2026 TU Wien

"""Scientific evidence preparation at the disambiguation boundary."""

import pandas as pd

from acib.name_disambiguation.workflow import prepare_scoring_author_frame


def test_scoring_projection_matches_source_without_mutating_matcher_metadata() -> None:
    """Keep preserved metadata out of scoring unless the source used it."""
    matching = pd.DataFrame(
        [
            {
                "sc_id": 1,
                "priority": 1,
                "person_email": "matched@example.org",
                "orcid": "0000-0001-2345-6789",
                "inferred_asjc_codes": "[]",
                "organization": "Wirtschaft",
            },
            {
                "sc_id": 2,
                "priority": 0,
                "person_email": "known@example.org",
                "orcid": "0000-0002-3456-7890",
                "inferred_asjc_codes": "1700",
                "organization": "Wirtschaft",
            },
        ]
    )
    original = matching.copy(deep=True)

    evidence = prepare_scoring_author_frame(matching)

    assert evidence["person_email"].tolist() == ["", "known@example.org"]
    assert evidence["orcid"].tolist() == ["", ""]
    assert evidence["inferred_asjc_codes"].tolist() == ["1400", "1700"]
    pd.testing.assert_frame_equal(matching, original)
