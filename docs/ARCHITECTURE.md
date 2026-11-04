# Architecture

## Shared DataFrame schemas

Reusable DataFrame contracts live in `src/acib/schemas/` and are organized by
data entity rather than processing stage. Pandera `DataFrameModel` classes use
the column constants in `acib.constants` and `acib.name_matching.constants` as
field aliases. Schema configs use `strict = False`, so validating a frame does
not discard additional researcher or author metadata.

Implemented contracts:

- `ResearcherInputSchema` (`schemas/researchers.py`) requires a non-null,
  unique `sc_id`. University and name-source columns are optional nullable
  fields so the schema can describe supported name inputs; the workflow
  separately requires `university_name` when organization filtering is on.
  `scopus_author_id`, when present, is validated as a nullable, integral,
  exactly representable CSV numeric value and is not coerced in the preserved
  metadata frame.
- `AuthorAffiliationSchema` (`schemas/affiliations.py`) requires non-null
  `author_id` and `affiliation_id`. Author IDs are deliberately not unique
  because candidate rows represent multiple profile affiliations. Parent
  affiliation columns are optional nullable pandas `Int64` fields. The CSV
  boundary parses their integral text spellings, including `.0` suffixes,
  directly to integers without passing through floating point. Invalid,
  fractional, or out-of-range values are rejected. Additional affiliation
  metadata is allowed.
- `ScopusAuthorInputSchema` (`schemas/authors.py`) extends the affiliation
  contract. It requires the EID, indexed-name, initials, publication-range,
  name-variant, and subject-area columns used by the downstream disambiguation
  parser; values may be null. A name-source column or given-name/surname pair
  must be available for matching. Complex serialized fields are not coerced.
- `MatchingOutputSchema` (`schemas/matching.py`) requires non-null `sc_id`,
  `author_id`, and `priority`; priorities are restricted to 0–4. It does not
  impose uniqueness on either ID. Passthrough and incomplete candidate details
  can be null. Numeric author IDs are converted to `int64` from integer values
  or numeric strings; nullable affiliation IDs use pandas `Int64`. Other
  researcher metadata is retained as allowed extra columns.
- `DisambiguationAuthorSchema` (`schemas/authors.py`) validates prepared
  author rows at the pair-scoring boundary. The same module defines contracts
  for the four tabular disambiguation artifacts and extends the Scopus input
  contract used by matching.
- `ArticleAuthorEvidenceSchema` and `ArticleReferenceSchema`
  (`schemas/publications.py`) validate publication-evidence inputs while
  allowing serialized fields and additional source metadata to pass through.
- `ResearcherDisambiguationSelectionSchema` (`schemas/researchers.py`)
  validates the final per-researcher selection artifact.

Input validation occurs once after CSV reading and explicit parent-ID parsing,
before matching or affiliation preparation. The returned validated frames are
used by the workflow. The final output is validated after researcher metadata
restoration and before CSV writing. Temporary matching frames are not repeatedly
validated.

Statistical-matching contracts live in `acib.statistical_matching.schemas`;
they validate researcher identity plus dynamically configured treatment,
covariates, and outcomes, while preserving extra producer columns. Regression
contracts live in `acib.regression.schemas` and describe the shared article
input, external impact/disruption inputs, mobility-proxy fields required by
regression, and regression-owned input/build artifacts. The regression input
contract validates the one-row-per-researcher artifact consumed by downstream
statistical matching; the professor-article contract covers the table used by
the regression suite's plot outputs.
Mobility consumes a GeoPackage and keeps that geospatial affiliation contract
distinct from the flattened Scopus author-affiliation rows validated here.

The standalone mobility command resolves its final researcher-selection
artifact from `name_disambiguation.outputs`, researcher metadata from
`name_matching.outputs.resolved_researchers`, and article-author and
GeoPackage datasets from `[data]`. It validates the tabular input boundaries,
optionally restores title/university metadata by researcher ID, then delegates
to `acib.mobility.inbreeding.analyze_academic_inbreeding`.
That function invokes stage-local input preparation before classification.
Results are written to the three explicit mobility output paths as author
affiliation history, inbreeding proxies, and author articles. The author-articles
artifact contains one row per mobility author profile and EID, with resolved
`author_afids` and original `unmapped_id` values joined by semicolons. The
approved decision is that `professor_author_profiles.csv` must never be written
to disk, including when professor merging is enabled. The mobility workflow
remains a standalone command rather than a step in the minimal
matching/disambiguation runner.

## Dependency-Aware Pipeline Configuration

This section is the authoritative target architecture for pipeline paths and
dependencies. Stage migration documentation may record transitional behavior,
but must not establish a competing path declaration or dependency model.

### Canonical paths and ownership

Every external input and generated artifact has exactly one canonical
configuration path. External inputs belong in `[data]`; stage settings belong
in `[stage]`; generated artifacts belong in `[stage.outputs]`. A consumer
refers to an upstream artifact through its single owning output declaration.
Do not add duplicate aliases, legacy path declarations, implicit conventional
output names, or speculative configuration fields for the same path. Treat
compatibility declarations as transitional and resolve them deliberately
during migration.

External inputs and generated artifacts have different location rules. Dataset
paths are user-configured under `[data]`; they may be relative or absolute and
may point to any accessible directory, including the legacy repository. The
application must not hardcode external dataset locations. Generated artifacts
are owned by their producing stage, whose `[stage.outputs]` declarations are
the only paths downstream consumers may use. Do not read a generated upstream
artifact from a legacy-repository path.

The intended migration sequence is:

```text
name_matching
      |
      v
name_disambiguation
      |
      v
    mobility
      |
      v
   regression
      |
      v
statistical_matching
```

This sequence specifies migration order, not data dependencies. Actual
dependencies must be established by inspecting each stage's implementation;
do not infer artifact dependencies from sequence alone. The audited source
regression implementation consumes a mobility inbreeding artifact and external
article data; statistical matching consumes the regression input artifact.

The currently established artifact dependencies are:

- `name_disambiguation` consumes
  `name_matching.outputs.resolved_researchers`.
- `mobility` consumes
  `name_disambiguation.outputs.final_author_selection`,
  `name_matching.outputs.resolved_researchers`, and the external
  article-author and author-affiliation datasets.
- `regression` consumes `mobility.outputs.author_inbreeding_proxy`, the external
  `data.article_author_file_path`, the external
  `data.article_impact_file_path` dataset, and an optional
  `data.disruption_indicator_file_path`. It does not consume any
  `mobility_impact` artifact.
- `statistical_matching` consumes `regression.outputs.regression_input`.

`mobility_impact` is excluded from the migration scope. Do not add it as a
planned stage or infer dependencies on it. Regression and statistical-matching
packages/configurations migrate in the listed order. Statistical matching is
configured as a downstream consumer of regression. Regression owns
its outcome groupings, labels, and aliases; it must not require the downstream
statistical-matching configuration to run. Artifact-level dependencies are
recorded explicitly; they are not inferred from stage order. Users invoke each
stage explicitly, but a stage command ensures its required generated upstream
artifacts exist by running their producer stages when they are missing. This
dependency execution follows the actual artifact graph and does not introduce
a scheduler or general-purpose DAG runner.

Each `[stage.outputs]` table contains exactly one `directory`. Stage constants
define fixed filenames, and output dataclasses expose derived artifact
properties; consumers use those properties without reconstructing paths.
Regression's LaTeX and plot roots are `outputs.directory / "latex"` and
`outputs.directory / "plots"`; dynamic filename patterns remain unchanged.
Regression's fixed generated artifacts are owned under its output directory,
including `regression_input` (the statistical-matching input),
`professor_article_input`, diagnostics, model tables, and text reports. Regression outcome
groups and display aliases are owned by `[regression.outcome_metadata]`; the
regression stage does not read them from downstream statistical-matching
configuration. Downstream stages reference the producer's derived properties;
no regression path is duplicated in statistical-matching configuration.

Canonical ownership is illustrated by these declarations:

```toml
[data]
article_author_file_path = "data/input/article.parquet"
article_reference_file_path = "data/input/article_reference.parquet"
author_affiliations_file_path = "data/input/author_affiliation.gpkg"

[name_matching.outputs]
directory = "data/slurm/output/name_matching"

[name_disambiguation]
max_workers = 3
same_author_score_threshold = 6.0
possible_author_score_threshold = 3.0

[name_disambiguation.outputs]
directory = "data/slurm/output/name_disambiguation"

[mobility.outputs]
directory = "data/slurm/output/mobility"
```

### Paths, ownership, and contracts

Resolve relative configured paths from the repository root, not the process
working directory. Explicit external absolute paths remain absolute. The
user is responsible for configuring accessible external dataset paths; no
dataset copy or relocation is required by the architecture. The producer owns
each generated artifact and defines its output contract;
consumers read it via that owner's declaration and validate it against the
applicable consumer contract/schema. Validate at meaningful stage boundaries
and preserve additional metadata where the schema permits it.

Keep path resolution, existence, validity, freshness, and execution distinct.
Resolution determines a path; existence checks presence; validity checks the
expected format/schema/contract; freshness asks whether the artifact reflects
current inputs and settings; execution runs a stage. Stale-artifact detection
is absent, and presence or schema validity does not establish freshness.
Automatic dependency execution is triggered only by a missing required generated
artifact; existing artifacts are reused without freshness checks. Missing
external inputs remain errors and never trigger a stage. If a required artifact
is missing while other outputs of its producer already exist, automatic
execution refuses to overwrite the partial output set unless the producer's
explicit overwrite option was supplied.

Preserve scientific reproducibility: configuration/path work must not change
algorithms, thresholds, classifications, statistical methods, randomness, or
output semantics without explicit approval. Record and test approved behavior
changes under the migration gates. The approved decision for
`professor_author_profiles.csv` is that it must never be written to disk; this
is an explicitly approved output behavior change. This records the decision,
not verification of its implementation.

`acib.pipeline.config` resolves external inputs from `[data]` and generated
artifacts from their producing stage's `[stage.outputs].directory`. Path aliases
and legacy duplicate declarations are rejected; stage-specific settings remain
in their respective packages. Disambiguation consumes the configured
name-matching output plus the article-author and citation-edge datasets. It
writes pair, cluster, final-selection, selection-score, text-report, and
JSONL-report artifacts under its single configured output directory. Stage CLIs
ensure their required generated dependencies before execution: disambiguation
ensures matching; mobility ensures matching and disambiguation; regression
ensures mobility when rebuilding its inputs; and statistical matching builds
the regression input when absent (ensuring mobility first). Matching has no
upstream stage dependency. Missing external datasets are reported as errors.
The explicit `run-acib-pipeline` command remains a convenience for the matching
and disambiguation sequence, not a general DAG scheduler.

The disambiguation workflow validates the matcher artifact, prepares canonical
author fields, validates publication-evidence tables, executes researcher
batches, validates tabular outputs, and persists results. No freshness or stale
artifact detection is performed; an existing artifact is checked only for
presence and schema validity before runner reuse.

Disambiguation tabular field names are centralized in
`acib.name_disambiguation.constants`; researcher, author, publication, and
affiliation fields confirmed to have cross-stage meaning come from
`acib.constants` or the existing matching schema constants. Private
implementation columns (for example, cluster sorting helpers) stay local to
their module, while labels and nested report keys remain values rather than
DataFrame schema constants.

## Identifier and validation considerations

The configured researcher input contains 93,855 rows with integral `int64`
`sc_id` values. The Scopus candidate input contains 679,325 rows; its
`author_id` and `affiliation_id` columns are integral `int64`, while nullable
parent-affiliation columns are read as float by pandas by default. Their
observed values are integral and well within exact integer range. The workflow
therefore reads those columns as text and parses them directly to `Int64`,
avoiding precision loss. It does not coerce the original researcher metadata.

Pandera validation adds a pass over each input frame and the final output.
Validation is limited to declared contract columns; `strict = False` preserves
additional columns. The Scopus author file is large, so validation is kept to
these stage boundaries and is not repeated on intermediate frames.

### Statistical-matching dependency and artifacts

Statistical matching is explicitly invoked with
`scripts/statistical_matching_analysis.py`, `statistical-matching-analysis`, or
`python -m acib.statistical_matching`. Its sole input is
`regression.outputs.regression_input`; it has no local input path and ensures
that regression's input artifact exists before running (building that artifact
without fitting regression models when absent). The ordered selected subset is
`[statistical_matching].outcomes`; outcome family and label come only from
`[regression.outcome_metadata]`. Nine CSV artifacts and `matching_formulas`
derive from `[statistical_matching.outputs].directory` and producer-owned
filename constants. LaTeX and plot files retain source filenames under
`directory / "latex"` and `directory / "plots"`. The plot utility reads these
canonical artifacts. Presence
and schema checks do not establish freshness.

The nine configured CSV artifacts are `balance_summary`, `covariate_balance`,
`outcome_effects_long`, `adjusted_outcome_means`, `outcome_effects_wide`,
`outcome_standard_errors_wide`, `method_diagnostics`, `propensity_scores`, and
`propensity_scores_by_method`. The LaTeX collection conditionally contains
`matching_results_relative.tex` and `matching_results_absolute.tex`; the plots
collection contains the balance and propensity figures, plus
`absolute_outcome_effects.png` when absolute outcomes are selected.

Matching validates dynamically configured treatment, covariates, and outcomes
and retains extra regression columns. It preserves current-source PSM, IPTW,
CEM, and entropy-balancing rules; balance diagnostics use the original pooled
within-group population SD and expand selected field/domain share families.
Outcome models use weighted least squares and configured robust covariance,
with separate relative/absolute LaTeX tables, PP-only percentage scaling, and
an absolute-outcome plot. Shared inbreeding/outcome/academic-age/year and
field/domain share fields are centralized in `acib.constants`.
