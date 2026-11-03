# Synthetic pipeline fixtures

These files contain fictional people, author profiles, publications, and
citations. They are small illustrative inputs, not research data. One
institution mapping value is intentionally real for the current-affiliation
selection example; that exception is described below.

## Files and relationships

- `researchers.csv`: 19 fictional researcher records, including missing matches
  and a pre-existing Scopus author ID.
- `scopus_authors.csv`: 21 candidate profile/publication rows. EIDs connect to
  `article_authors.csv`; repeated names on different author IDs and
  affiliations intentionally create ambiguous candidates.
- `article_authors.csv`: one article-author evidence row for each synthetic EID,
  with serialized author IDs, author names, and subject-area metadata.
- `article_references.csv`: synthetic citation edges between existing EIDs.

The name cases include ordinary exact matches, initials, multi-part and
hyphenated given names, compound surnames, diacritics, apostrophes, particles,
same-name profiles, cross-affiliation profiles, known-ID passthrough data, and
researchers without candidate profiles. Not every name variation is expected to
match: the fixtures are examples for exploring the matching rules, not labels
that every possible candidate is correct.

Researcher `990019` (`Ari Example`) has two same-name profiles and is deliberately
configured with `TU Wien` as the current university. The university name and
Scopus affiliation ID `60018163` are real mapping values used only to exercise
the existing current-affiliation selection rule; the researcher, author
profiles, and publication records are fictional. The second profile has a
different synthetic affiliation and non-overlapping publication history.

The fictional organizations are intentionally absent from the production
organization mapping. Examples that use `match_researchers` directly should
disable organization filtering; workflow-level examples can separately
illustrate existing-ID passthrough and configuration behavior. The remaining
CSV files provide small linked inputs for later disambiguation notebooks. They
do not include the geospatial affiliation dataset needed by the mobility stage
or the impact datasets used downstream.

`config/pipeline_synthetic.toml` is the canonical example configuration for
the name-matching and name-disambiguation notebooks. It declares these fixtures
as external inputs and stores each stage's artifacts in its own directory under
`data/output/synthetic/`, separate from earlier synthetic-run outputs.
