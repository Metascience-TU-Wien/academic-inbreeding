# Synthetic pipeline fixtures

These files contain fictional people, author profiles, publications, and
citations. They are small illustrative inputs, not research data. The real TU
Wien mapping is intentionally used for current-affiliation and mobility examples;
the people and publication records remain fictional.

## Files and relationships

- `researchers.csv`: 19 fictional researcher records, including missing matches,
  a pre-existing Scopus author ID, and academic titles for selected researchers.
- `scopus_authors.csv`: 21 candidate profiles with one row per unique author ID.
  EIDs connect to `article_authors.csv`; repeated names on different author IDs
  and affiliations intentionally create ambiguous candidates. Publication
  histories for mobility are stored in `article_authors.csv`.
- `article_authors.csv`: one article-author evidence row for each synthetic EID,
  with author IDs, author-aligned affiliation IDs, cover dates, author names,
  and subject-area metadata. Author and affiliation ID lists use semicolons as
  delimiters in the mobility input format.
- `article_references.csv`: synthetic citation edges between existing EIDs.
- `author_affiliations.gpkg`: a geospatial mapping containing only fictional
  synthetic affiliation IDs and parent institution metadata.

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

Most fictional organizations are absent from the production university mapping.
The mobility example enables university-affiliation detection and uses the real
TU Wien affiliation ID for Ada Lovelace and Jamie Lee. Their fictional
publication histories include TU Wien in both the configured first-two-year
window and their latest publication, demonstrating two positive inbreeding
proxies. John Doe's fictional institution is unmapped and remains unclassified.
Examples that use `match_researchers` directly should disable organization
filtering; workflow examples can separately illustrate existing-ID passthrough
and configuration behavior. The fixtures are sufficient for the synthetic
mobility stage, but not for downstream impact analyses.

`config/pipeline_synthetic.toml` is the canonical example configuration for
the synthetic matching, disambiguation, and mobility workflows. It declares
these fixtures as external inputs and stores each stage's artifacts in its own directory under
`data/output/synthetic/`, separate from earlier synthetic-run outputs.
