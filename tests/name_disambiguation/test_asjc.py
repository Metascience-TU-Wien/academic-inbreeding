# Copyright © 2026 TU Wien

from acib.name_disambiguation.asjc import (
    ASJC_TAXONOMY,
    level_two_asjc_code,
    level_two_asjc_codes,
)


def test_level_two_code_collapses_specific_asjc_codes_and_rejects_broad_domains() -> None:
    assert level_two_asjc_code(2604) == 2600
    assert level_two_asjc_code(2613) == 2600
    assert level_two_asjc_code(2207) == 2200
    assert level_two_asjc_code(1708) == 1700
    assert level_two_asjc_code(2600) == 2600
    assert level_two_asjc_code(26) is None
    assert level_two_asjc_code(17) is None
    assert level_two_asjc_codes({2604, 2613, 2207, 1708}) == {2600, 2200, 1700}


def test_every_taxonomy_code_maps_to_its_level_two_category() -> None:
    for code in ASJC_TAXONOMY:
        expected = None if code < 1000 else (code // 100) * 100
        assert level_two_asjc_code(code) == expected
