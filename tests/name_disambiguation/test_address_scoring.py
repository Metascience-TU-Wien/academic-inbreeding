# Copyright © 2026 TU Wien

"""Address-evidence routing tests for Scopus and OpenAlex."""

import pandas as pd

from acib.name_disambiguation.constants import OPENALEX_AUTHOR_AFFILIATION_EVIDENCE
from acib.name_disambiguation.models import AuthorRecord
from acib.name_disambiguation.parsing import author_from_row
from acib.name_disambiguation.scoring import score_pair
from acib.name_disambiguation.scoring_context import build_scoring_context


def _author(author_id: str) -> AuthorRecord:
    return author_from_row(
        {
            "author_id": author_id,
            "name": "Doe, Jane",
            "surname": "Doe",
            "given_name": "Jane",
            "initials": "J",
            "indexed_name": "Doe J",
            "name_variants": "[]",
            "subject_areas": "[]",
        }
    )


def test_openalex_author_linked_address_scores_four_without_double_counting() -> None:
    """Score author-specific Works affiliations through the +4 address layer."""
    left = _author("https://openalex.org/A1")
    right = _author("https://openalex.org/A2")
    article_authors = pd.DataFrame(
        [
            {
                "eid": "https://openalex.org/W1",
                "author_id": left.author_id,
                OPENALEX_AUTHOR_AFFILIATION_EVIDENCE: [
                    {
                        "author_id": left.author_id,
                        "cities": ["Vienna"],
                        "countries": ["Austria"],
                    }
                ],
            },
            {
                "eid": "https://openalex.org/W2",
                "author_id": right.author_id,
                OPENALEX_AUTHOR_AFFILIATION_EVIDENCE: [
                    {
                        "author_id": right.author_id,
                        "cities": ["Vienna"],
                        "countries": ["Austria"],
                    }
                ],
            },
        ]
    )

    decision = score_pair(
        left,
        right,
        context=build_scoring_context(
            authors=[left, right],
            article_author_df=article_authors,
        ),
    )
    address_components = [
        component
        for component in decision.components
        if component.label in {"author_address_match", "publication_address_match"}
    ]

    assert [(component.label, component.weight) for component in address_components] == [
        ("author_address_match", 4.0)
    ]


def test_scopus_article_affiliation_address_remains_publication_address_two_points() -> None:
    """Keep Scopus article-linked addresses on the existing +2 scoring layer."""
    left = _author("100")
    right = _author("200")
    article_authors = pd.DataFrame(
        [
            {
                "eid": "S1",
                "author_id": left.author_id,
                "affiliation_city": "Vienna",
                "affiliation_country": "Austria",
            },
            {
                "eid": "S2",
                "author_id": right.author_id,
                "affiliation_city": "Vienna",
                "affiliation_country": "Austria",
            },
        ]
    )

    decision = score_pair(
        left,
        right,
        context=build_scoring_context(
            authors=[left, right],
            article_author_df=article_authors,
        ),
    )
    address_components = [
        component
        for component in decision.components
        if component.label in {"author_address_match", "publication_address_match"}
    ]

    assert [(component.label, component.weight) for component in address_components] == [
        ("publication_address_match", 2.0)
    ]
