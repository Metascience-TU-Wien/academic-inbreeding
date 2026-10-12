# Copyright © 2026 TU Wien

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from acib.mobility.schemas import (
    MobilityAffiliationSchema,
    MobilityArticleAuthorSchema,
    MobilityAuthorAffiliationsOutputSchema,
    MobilityAuthorArticlesOutputSchema,
    MobilityAuthorInbreedingOutputSchema,
    MobilityUniqueAffiliationsOutputSchema,
)
from acib.schemas.researchers import ResearcherDisambiguationSelectionSchema


def test_article_author_schema_preserves_complex_fields_and_extra_metadata() -> None:
    frame = pd.DataFrame(
        {
            "eid": ["paper-1"],
            "author_ids": ["1001;1002"],
            "author_afids": ["10-20;30"],
            "cover_date": ["2020-01-01"],
            "subject_areas": ['[{"code": 1502}]'],
        }
    )

    result = MobilityArticleAuthorSchema.validate(frame, lazy=True)

    assert result.equals(frame)
    assert result.loc[0, "author_afids"] == "10-20;30"
    assert "subject_areas" in result.columns


def test_article_author_schema_requires_processing_fields() -> None:
    frame = pd.DataFrame({"eid": ["paper-1"], "author_ids": ["1001"]})

    with pytest.raises(SchemaErrors):
        MobilityArticleAuthorSchema.validate(frame, lazy=True)


def test_article_author_schema_accepts_empty_input_with_schema_columns() -> None:
    frame = pd.DataFrame(
        {
            "eid": pd.Series(dtype="object"),
            "author_ids": pd.Series(dtype="object"),
            "author_afids": pd.Series(dtype="object"),
            "cover_date": pd.Series(dtype="object"),
        }
    )

    result = MobilityArticleAuthorSchema.validate(frame, lazy=True)

    assert result.empty
    assert result.columns.tolist() == frame.columns.tolist()


def test_affiliation_schema_accepts_nullable_parent_and_extra_metadata() -> None:
    frame = pd.DataFrame(
        {
            "affiliation_id": pd.Series([10, 20], dtype="int64"),
            "affiliation_id_parent": [pd.NA, 100],
            "preferred_name": ["Child", "Parent"],
            "extra_metadata": ["first", "second"],
        }
    )

    result = MobilityAffiliationSchema.validate(frame, lazy=True)

    assert result.equals(frame)
    assert result["affiliation_id_parent"].isna().tolist() == [True, False]
    assert "extra_metadata" in result.columns


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"affiliation_id": [None]}),
        pd.DataFrame({"affiliation_id": ["not-an-integer"]}),
    ],
)
def test_affiliation_schema_rejects_missing_or_invalid_ids(frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaErrors):
        MobilityAffiliationSchema.validate(frame, lazy=True)


def test_affiliation_history_output_schema_requires_integral_identifiers() -> None:
    frame = pd.DataFrame(
        {
            "author_ids": pd.Series([1001], dtype="int64"),
            "affiliation_id": pd.Series([10], dtype="int64"),
            "affiliation_type": ["education"],
            "affiliation_id_parent": pd.Series([pd.NA], dtype="Int64"),
            "date_first": ["2020-01-01"],
            "additional": ["retained"],
        }
    )

    result = MobilityAuthorAffiliationsOutputSchema.validate(frame, lazy=True)

    assert result.equals(frame)
    assert result.loc[0, "additional"] == "retained"

    invalid = frame.assign(affiliation_id=["bad-id"])
    with pytest.raises(SchemaErrors):
        MobilityAuthorAffiliationsOutputSchema.validate(invalid, lazy=True)


def test_unique_affiliation_schema_rejects_invalid_identifier() -> None:
    with pytest.raises(SchemaErrors):
        MobilityUniqueAffiliationsOutputSchema.validate(
            pd.DataFrame({"affiliation_id": ["bad-id"]}), lazy=True
        )


def test_inbreeding_output_schema_allows_unclassified_proxy_and_extra_fields() -> None:
    frame = pd.DataFrame(
        {
            "author_ids": pd.Series([1001, 1002], dtype="int64"),
            "is_inbreeding_proxy": pd.Series([True, pd.NA], dtype="boolean"),
            "inbreeding_detection_status": ["assigned", "manual_review"],
        }
    )

    result = MobilityAuthorInbreedingOutputSchema.validate(frame, lazy=True)

    assert result.equals(frame)
    assert result["is_inbreeding_proxy"].isna().tolist() == [False, True]


def test_author_articles_schema_requires_eid_and_preserves_article_fields() -> None:
    frame = pd.DataFrame({"eid": ["paper-1"], "doi": ["10/example"]})

    result = MobilityAuthorArticlesOutputSchema.validate(frame, lazy=True)

    assert result.equals(frame)
    with pytest.raises(SchemaErrors):
        MobilityAuthorArticlesOutputSchema.validate(
            pd.DataFrame({"doi": ["10/example"]}), lazy=True
        )


def test_final_selection_schema_preserves_integral_numeric_identifier_dtypes() -> None:
    frame = pd.DataFrame(
        {
            "sc_id": pd.Series([10], dtype="int64"),
            "status": ["assigned_single_candidate"],
            "selected_cluster_id": pd.Series([pd.NA], dtype="Float64"),
            "selected_disambiguated_cluster_id": pd.Series([1001.0], dtype="float64"),
            "selected_author_ids": [[1001]],
            "possible_clusters": [[]],
            "inferred_asjc_codes": [[]],
            "scopus_asjc_codes": [[]],
            "inferred_asjc_labels": [[]],
            "reason": ["selected"],
        }
    )

    validated = ResearcherDisambiguationSelectionSchema.validate(frame, lazy=True)

    assert validated["sc_id"].dtype == "int64"
    assert validated["selected_disambiguated_cluster_id"].dtype == "float64"
    assert validated.loc[0, "selected_disambiguated_cluster_id"] == 1001.0


@pytest.mark.parametrize(
    ("column", "value"),
    [("sc_id", 1.5), ("selected_disambiguated_cluster_id", 1001.5)],
)
def test_final_selection_schema_rejects_fractional_identifiers(column: str, value: float) -> None:
    frame = pd.DataFrame(
        {
            "sc_id": [1],
            "status": ["assigned_single_candidate"],
            "selected_disambiguated_cluster_id": [1001],
            "selected_author_ids": [[1001]],
            "possible_clusters": [[]],
            "inferred_asjc_codes": [[]],
            "scopus_asjc_codes": [[]],
            "inferred_asjc_labels": [[]],
            "reason": ["selected"],
        }
    )
    frame[column] = value

    with pytest.raises(SchemaErrors):
        ResearcherDisambiguationSelectionSchema.validate(frame, lazy=True)
