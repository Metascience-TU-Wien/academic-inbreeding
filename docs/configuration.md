# Configuration reference

ACIB reads TOML configuration from the repository root. Relative paths resolve
against that root, independently of the current directory; absolute paths are
retained. Environment variables and `~` are expanded. External datasets belong
in `[data]`, stage settings in `[stage]`, and generated artifacts under the
producer's `[stage.outputs].directory`. Each output table accepts only the
directory; filenames are fixed by stage. See [Architecture](ARCHITECTURE.md)
for path ownership, dependencies, and artifact-contract rules.


## External inputs: `[data]`

| Key | Used by | Description and contract |
| --- | --- | --- |
| `researcher_file_path` | Name matching | Researcher/staff CSV. `sc_id` must be present, non-null, and unique; a supported name representation is required. Organization filtering also requires `university_name`. Other metadata is retained. |
| `candidate_authors_file` | Name matching/disambiguation | Candidate Scopus-author CSV. Candidate author IDs can occur on multiple affiliation rows. Required fields are defined in `acib.schemas.authors`; serialized fields are not generically coerced. |
| `article_author_file_path` | Disambiguation, mobility, regression build | Publication-author evidence. Stage-specific schemas apply. |
| `article_reference_file_path` | Disambiguation | Citation/reference edges used as evidence. |
| `author_affiliations_file_path` | Mobility | External geospatial affiliation GeoPackage, distinct from the flattened Scopus affiliation rows. |
| `article_impact_file_path` | Regression (optional input) | Article impact indicators, validated against regression's input contract when used. |
| `disruption_indicator_file_path` | Regression (optional) | External disruption indicators. |

Paths are strings. Requiredness depends on the invoked stage and selected
workflow. Schemas generally permit additional columns; see `src/acib/schemas/`,
`src/acib/mobility/schemas.py`, and `src/acib/regression/schemas.py` for exact
contracts. A valid path does not guarantee valid file contents.

## Name matching

Requires `[name_matching]`, its output directory, and the two researcher and
candidate-author paths above. Output: `resolved_researchers.csv`.

| Key | Type; default | Behavior |
| --- | --- | --- |
| `enable_given_name_aliases` | Boolean; `true` | Enables alternate given-name forms in matching. |
| `enable_organisation_filter` | Boolean; `true` | Filters using organization/affiliation mapping; requires researcher `university_name` and compatible mapping data. Synthetic config disables it for fictional organizations. |
| `existing_scopus_author_id_column` | String; optional | Researcher input column with known IDs, handled as pre-resolved/pass-through records. |
| `max_workers` | Positive integer; `3` | Maximum parallel matching workers. |
| `encoding` | Nonempty string; `"utf-8-sig"` | CSV encoding. |
| `[name_matching.outputs].directory` | Path; required | Directory for the fixed output artifact. |

## Author disambiguation

Requires the name-matching artifact, article-author and article-reference data,
and an output directory. Pair labels are not final researcher assignments;
clustering and selection apply additional evidence and conflict safeguards.

| Key | Type; default | Behavior |
| --- | --- | --- |
| `enable_reporting` | Boolean; `true` | Enables console reporting; report files are still output. |
| `same_author_score_threshold` | Nonnegative number; `6.0` | Pair-score threshold for the strongest same-author category. |
| `possible_author_score_threshold` | Nonnegative number; `3.0` | Lower possible-match threshold; cannot exceed same-author threshold. |
| `max_workers` | Positive integer; `1` | Maximum batch worker processes. |
| `[name_disambiguation.outputs].directory` | Path; required | Output directory. |

Fixed outputs: `pair_decisions.csv`, `author_clusters.csv`,
`final_author_selection.csv`, `final_author_selection_scores.csv`,
`disambiguation_report.txt`, and `disambiguation_report.jsonl`.

## Mobility

Consumes the disambiguation selection and matching metadata, plus external
article-author and affiliation data. Outputs are `author_affiliations.csv`,
`author_inbreeding_proxy.csv`, and `author_articles.csv`.

| Key | Type; default | Behavior |
| --- | --- | --- |
| `enable_university_affiliation_inbreeding_detection` | Boolean; `false` | Enables the stricter university/early-affiliation detection rule. |
| `enable_coaffiliated_professor_merge` | Boolean; `false` | Merges qualifying professor rows for the same author profile in memory. |
| `professor_title_pattern` | String; `"prof"` | Pattern for professor selection. |
| `first_affiliation_publication_count` | Positive integer; `3` | Early-publication window size. |
| `early_affiliation_method` | `first_n_publications` or `first_n_years`; default `first_n_publications` | How the early affiliation window is selected. |
| `first_affiliation_year_count` | Positive integer; `2` | Early-year window size. |
| `min_publication_history_years` | Nonnegative integer or null; `null` | Optional minimum publication-history duration. No minimum by default. |
| `max_workers` | Positive integer; `3` | Worker processes. |
| `[mobility.outputs].directory` | Path; required | Output directory. |

Classification uses publication-affiliation histories rather than a doctoral
institution-only rule. The exact processing, including parent organizations and
multiple profiles, is implemented in `acib.mobility.inbreeding` and its input
preparation. Mobility's inbreeding proxy is consumed by regression.

## Regression

Regression uses the mobility proxy and external article data to construct
`author_regression_input.csv`. With `rebuild_input = false`, that file must
already exist under `[regression.outputs].directory`; it is reused without
freshness checks. Statistical matching consumes the same artifact.

| Key | Type; default | Behavior |
| --- | --- | --- |
| `mle_failure_policy` | `error` or `warn`; `error` | Handling of maximum-likelihood diagnostic failure. |
| `mle_gradient_tolerance` | Positive number; `0.0001` | Per-observation score-norm tolerance for applicable likelihood fits. |
| `print_model_summaries` | Boolean; `false` | Print fitted summaries to standard output. |
| `drop_missing` | Boolean; `true` | Drop missing observations needed by the individual model formula; samples may vary by model/outcome. |
| `min_publications` | Positive integer; `4` | Publication threshold in input/sample construction. |
| `min_first_year` | Nonnegative integer or null; `null` | Optional inclusive cutoff for first publication year. |
| `professor_title_pattern` | Nonempty string; `"prof"` | Case-insensitive professor-title selection pattern. |
| `exclude_titles` | String array; `["FH-Ass. Prof."]` | Titles excluded from model fitting. |
| `rebuild_input` | Boolean; `true` | Build regression input; false requires an existing file. Building ensures mobility artifact as needed. |
| `covariance_type` | String or null; null | Optional covariance estimator override where supported. |
| `model_specs` | Array of tables; source M1–M10 defaults | Optional ordered replacement. Each table requires unique `model_id` and nonempty string-array `terms`. |
| `zinb_inflation_formulas` | Outcome-to-formula table; `{}` | Optional ZINB inflation formulas. |
| `zinb_regularization` | Nonnegative number; `3.0` | Initialization regularization. |
| `zinb_fit_method` | Nonempty string; `"bfgs"` | ZINB optimizer. |
| `zinb_max_iterations` | Positive integer; `1000` | ZINB iteration limit. |
| `zinb_dispersion_power` | Positive integer; `2` | ZINB negative-binomial variance power. |
| `[regression.outputs].directory` | Path; required | Root for fixed artifacts and `latex/`, `plots/`. |

The default specifications are source-defined M1–M10. Optional custom model
terms can replace them; field-share markers expand using available fields and
an omitted reference category. Formula encoding and reference categories follow
the implemented statsmodels/Patsy specifications.

### Outcomes and estimators

`[regression.dependent_variables]` maps internal outcome names to one model
family. Supported identifiers:

| ID | Estimator and interpretation |
| --- | --- |
| `ols` | Ordinary least squares; coefficients. |
| `olshc3` | OLS with HC3 covariance; coefficients. |
| `huber` | Huber-T robust linear regression. |
| `frac` | Binomial-logit fractional response; average marginal effects (AMEs) and delta-method inference. |
| `ppml` | Poisson pseudo-maximum-likelihood; AMEs and delta-method inference; nonnegative fractional responses are retained. |
| `nb` | Negative-binomial likelihood; nonlinear AME reporting. Fractional count responses are rounded on a model-owned copy as recorded in `MIGRATION.md`. |
| `pr` | Poisson regression (legacy estimator). |
| `zinb` | Zero-inflated negative-binomial (legacy estimator). |

Nonlinear model parameters and reported AMEs are distinct. Continuous terms use
average derivatives; categorical contrasts use average prediction differences.
Estimator-specific response-domain and convergence rules apply. The regression
configuration provides outcome metadata independently of statistical matching.

Each `[regression.outcome_metadata.<outcome>]` table requires exactly these
fields:

| Key | Type; allowed values | Purpose |
| --- | --- | --- |
| `outcome_type` | String; `relative` or `absolute` | Output grouping and estimator/report conventions. |
| `alias` | Nonempty string | Human-readable label; internal variable key remains unchanged. |

The example outcomes are `hazen_perc_med` (mwPR), `pp_top_10`, `pp_top_5`,
`pp_top_1` (relative), and `P_top10`, `P_top5`, `P_top1`, `n_public` (absolute).
The list is an example, not a loader-enforced closed outcome list.

## Statistical matching

Consumes the regression-owned input artifact, never a separately declared path.
If absent, it builds regression input without fitting regression models.

| Key | Type; default | Behavior |
| --- | --- | --- |
| `outcomes` | Nonempty unique string array; eight example outcomes | Outcome columns selected from regression input; each needs regression metadata. |
| `covariates` | Nonempty string array; `academic_age`, `title`, `primary_field`, `first_year` | Observed characteristics used in design and balance. Data presence is checked at runtime. |
| `methods` | Nonempty unique string array; `PSM`, `IPTW`, `CEM`, `EB` | Supported methods: propensity-score matching, inverse-probability weighting, coarsened exact matching, entropy balancing. |
| `treatment_variable` | Nonempty string; `is_inbred` | Treatment column. |
| `treated_value` | Boolean; `true` | Value identifying the treated group. |
| `estimand` | Nonempty string; `ATT` | The implemented estimand is ATT; this observational estimand does not itself establish causal identification. |
| `exclude_titles` | String array; `["FH-Ass. Prof."]` | Title values excluded before method analysis. |
| `balance_threshold` | Positive number; `0.10` | Diagnostic threshold on absolute standardized mean differences; exceeding it is not an automatic run failure. |
| `covariance_type` | Nonempty string; `"HC3"` | Covariance estimator for outcome models. |
| `drop_missing` | Boolean; `true` | Missing analysis values are dropped before method calculations. |
| `[statistical_matching.cem_cutpoints]` | Numeric-array table; defaults for academic age and first year | Per-variable explicit CEM boundaries. |
| `[statistical_matching.outputs].directory` | Path; required | Output root. |

PSM fits a logistic propensity model, clips scores to `[1e-6, 1 - 1e-6]`, and
matches each treated observation to its nearest available control by absolute
score distance, without replacement. Ties follow input row order; there is no
configured caliper or common-support restriction. Unpaired observations are
not included. IPTW uses treated weight 1 and control weight `p / (1 - p)` for
ATT; it clips scores as above and does not trim or truncate weights. CEM creates
right-closed intervals (first interval includes its lowest endpoint), with
infinite outer bounds; out-of-range finite values remain in outer bins. Matched
strata retain treated weights of 1 and control weights `n_treated / n_control`;
unmatched strata are excluded. EB uses encoded covariate columns and balances
their means. It drops columns whose treated mean is outside control support or
constant incompatibly, then uses exponential-tilt weights optimized by BFGS
(gradient tolerance `1e-8`, iteration limit `10000`); failure beyond the
gradient guard raises an error. Control weights are normalized to the treated
count. Output outcome models use weighted least squares and the configured
covariance type.

CEM cutpoint arrays are checked to contain numbers, but the loader does not
validate ordering, uniqueness, or range. Sorted boundaries are expected for
meaningful bins. A configured numeric column uses the listed boundaries;
variables without a cutpoint entry are treated as string-valued categories,
not automatically quantile-coarsened. The default cutpoint table provides
`academic_age` and `first_year` boundaries.

## Artifacts, execution, and reproducibility

| Producer | Main artifacts and consumer |
| --- | --- |
| Name matching | `resolved_researchers.csv` → disambiguation and mobility. |
| Disambiguation | Pair decisions, clusters, final selection, scores, text/JSONL reports → mobility uses final selection. |
| Mobility | Affiliation history, inbreeding proxy, author articles → regression uses proxy. |
| Regression | `author_regression_input.csv`, diagnostics, model tables/reports/plots → statistical matching uses input. |
| Statistical matching | Nine CSV tables, formula report, LaTeX and plots. |

Users invoke each stage explicitly. A stage may invoke configured producers for
missing generated artifacts; missing external inputs are errors. Existing
artifacts are reused without freshness detection. Only matching and
disambiguation expose `--overwrite`, and it is local to that stage. The synthetic
fixtures are illustrative fictional inputs, not complete research datasets;
see [`../data/test/synthetic/README.md`](../data/test/synthetic/README.md).

Statistical matching's CSV outputs are `balance_summary.csv`,
`covariate_balance.csv`, `outcome_effects_long.csv`,
`adjusted_outcome_means.csv`, `outcome_effects_wide.csv`,
`outcome_standard_errors_wide.csv`, `method_diagnostics.csv`,
`propensity_scores.csv`, and `propensity_scores_by_method.csv`; it also writes
`matching_formulas.txt`, plus generated files under `latex/` and `plots/`.
Regression's named fixed files are defined in
`acib.regression.constants`; its outputs also include those under `latex/` and
`plots/`.

## Command-line options

All five commands accept `--config PATH`; the documented repository-root
commands are in the [README](../README.md#running-the-pipeline). The default
configuration is stage-specific and can be seen in each command's `--help`.
Matching and disambiguation accept `--overwrite`, which permits replacement of
that stage's existing output set. Matching, mobility, regression, and
statistical matching accept `--validate-paths`; regression also accepts
`--print-model-summaries`. Disambiguation has no path-validation flag. None of
the other stages has an overwrite flag. `-h`/`--help` prints usage and exits.
