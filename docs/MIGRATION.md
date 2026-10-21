# Migration plan and results

## Approved regression estimator refactor

- This section supersedes the earlier regression-robustness-bundle description
  below. Each outcome now selects one estimator independently. Supported values
  are `ols`, `olshc3`, `huber`, `frac`, `ppml`, `nb`, `pr`, and `zinb`;
  `fractional_logit` is renamed `frac`, and the former `nbhc3` behavior is
  represented by `nb` (negative-binomial likelihood with HC3 covariance).
  Combined values `huberfrac`, `huberppml`, and `nbppml` are rejected. `pr` and
  `zinb` remain available as independent legacy estimators.
- `frac`, `ppml`, and `nb` preserve raw fitted parameters in result objects and
  add separately identified average marginal effects. Continuous predictors
  use average derivatives; categorical predictors use average predicted
  differences from the Patsy reference category. Delta-method standard errors
  and p-values use the fitted covariance matrix. Fractional-logit effects are
  restored to the outcome's original scale. Model specifications 1–10 retain
  their existing formulas and sample selection. The age effect differentiates
  the complete formula, including a quadratic age term; the separate squared
  term row remains a placeholder for nonlinear estimators.
- `frac` requires outcomes in [0, 1] or [0, 100], retaining exact endpoints;
  observed 0–100 outcomes are divided by 100 for fitting and results are
  restored to the original scale. `ppml` preserves nonnegative outcomes,
  including fractional publication counts and zeros. `nb` requires
  nonnegative counts; fractional `P_top10`, `P_top5`, and `P_top1` responses
  are rounded to nearest integer using half-up ties (for example, 2.5 becomes
  3), on an estimator-owned copy of the data. This NB approximation affects
  only the response passed to NB and is identified in result metadata.
- LaTeX presentation retains existing templates and filenames. OLS-family,
  Huber, and nonlinear estimators use estimator-appropriate effect headings;
  the compact model-comparison table keeps its original stars-based p-value
  presentation. Huber's Model 1–10 fitting is generic; existing M9/M10-specific
  supplementary reports remain intentionally tied to those specifications.
- Approval of this estimator redesign, its half-up count rounding, the
  M9-specific supplementary reports, and the typed result metadata was granted
  during the design review. The user separately authorized Phase 4 regression
  test migration and verification.

## Regression estimator refactor — Phase 4 verification

- Migrated/updated regression tests under `tests/regression/`, including
  independent selection of all eight supported estimators, removal of combined
  identifiers, input reuse workflow coverage, outcome-domain checks, exact
  fractional endpoints, estimator-specific transformations, and standard
  output filenames.
- Added independent numerical AME references for fractional logit and PPML
  binary contrasts/delta-method inference, NB binary contrasts and title
  contrasts with robust standard errors and p-values, and a continuous Model 7
  age derivative that includes the quadratic term. The table keeps the age²
  coefficient row as a placeholder for nonlinear fits.
- Added half-up NB tie/boundary checks (`2.5 -> 3`), actual NB response checks,
  original-input immutability checks, PPML fractional-response preservation,
  mixed-estimator headings/notes, and M9/M10 supplementary output-file checks.
- Compared source and target Model 1–10 specifications using the original
  repository's implementations at
  `/home/chri/PycharmProjects/name-matching-playground`: all formula strings,
  complete-case row indexes, Patsy design columns, and design-matrix values
  matched exactly. Sample sizes were 119 for Models 1–8 and 118 for Models 9–10
  in the deterministic comparison fixture. Design dimensions were 6, 7, 8, 8,
  9, 10, 11, 36, 37, and 36 respectively.
- Additional source-vs-target estimator comparisons matched estimates and
  standard errors for OLS, OLS-HC3, Poisson regression, and ZINB on deterministic
  synthetic data. NB parameter estimates matched; its covariance intentionally
  differs because the requested target behavior uses HC3 while the source's
  default `nb` uses conventional covariance. Huber matched the direct statsmodels
  HuberT/RLM reference. Fractional logit and PPML matched direct Binomial-logit
  and Poisson GLM references.
- `uv run pytest tests/regression/ -q`: **101 passed**, 2 statsmodels ZINB
  optimizer runtime warnings (invalid logarithms during candidate fitting).
  `uv run pytest tests/integration/ -q`: after updating the regression-output
  assertion for estimator-specific inference notes, **24 passed**; 5 pandas
  concatenation FutureWarnings remain in name-matching/mobility tests.
- `uv run pytest --import-mode=importlib -q`: **349 passed, 2 failed**. Both
  failures are in `tests/pipeline/test_dependencies.py` and concern producer
  ordering/missing external inputs for name disambiguation; they are outside the
  regression scope. No regression test failed.
  Default `uv run pytest -q` still stops during collection because test modules
  named `test_workflow.py` collide; importlib mode collects the suite.
- `uv run ruff check` and all configured pre-commit hooks passed, including
  Flake8, Black, mypy, and isort. No research data or original-repository files
  were modified. No estimator-behavior discrepancy was found beyond the
  explicitly requested NB HC3 covariance change.
- Follow-up runtime investigation found that AME delta-method gradients rebuilt
  Patsy design matrices and evaluated parameter perturbations serially. AME
  scenario matrices are now built once, and the same central finite-difference
  parameter perturbations are evaluated as a batch. The scenario data,
  covariance calculation, finite-difference method, and reported estimand are
  unchanged. Regression tests (101 passed) and all configured pre-commit hooks
  passed after this optimization.

## Fractional-logit Model 9 independent validation

- Validated the completed run configured by
  `config/pipeline_regression_frac.toml`, reading its resolved researcher input
  and regression outputs without modifying either. Model 9's source and target
  formula terms matched; for each of mwPR and the three PP outcomes, the
  complete-case row indexes and Patsy design matrices matched the original
  repository exactly (7,842 rows, 38 columns).
- Independent direct statsmodels GLM fits used Binomial with the Logit link and
  HC3 covariance. For each outcome, fitted parameters and standard errors
  matched the saved target coefficients to maximum absolute differences below
  `5e-16` and `1e-16`, respectively. All four fits converged with finite
  estimates/covariances. Predicted probabilities stayed in `[0, 1]`; none were
  below `1e-6` or above `1 - 1e-6`.
- Independent Patsy scenario construction, SciPy inverse-logit predictions,
  parameter finite differences, and HC3 delta-method calculations reproduced
  all 37 saved AMEs per outcome. Maximum estimate differences were below
  `5e-15`, standard-error differences below `3e-11`, and p-value differences
  below `3e-12`. This includes inbreeding discrete effects, continuous
  derivatives, and categorical contrasts. The formula's Patsy treatment
  baseline for title is Ao. Prof.; the paper table separately contrasts each
  title against Full Prof., matching the original reporting convention.
- Only mwPR was rescaled from 0–100 to 0–1 for fitting and returned to
  percentile-point units. The three PP outcomes remained on their original
  0–1 fractional scale for fitting and their AMEs were expressed in percentage
  points. Their nonbinary responses and observed endpoints were retained; the
  actual sample contains 6,336/5,531/3,367 nonbinary observations for PP(top
  10%)/PP(top 5%)/PP(top 1%), respectively. PP(top 10%) has 1,486 zeros and 20
  ones; PP(top 5%) has 2,305 zeros and 6 ones; PP(top 1%) has 4,475 zeros and
  no ones. mwPR has no exact endpoints in this sample.
- For inbreeding, independently calculated AMEs (HC3 delta SE, z-test p-value)
  were: mwPR `-1.84454` (`0.32422`, `p=1.28e-8`, `***`); PP(top 10%) `-1.01886`
  (`0.31526`, `p=0.00123`, `**`); PP(top 5%) `-0.43251` (`0.21728`,
  `p=0.04653`, `*`); PP(top 1%) `-0.10233` (`0.07654`, `p=0.18121`, no star).
  The M9 LaTeX table labels estimates AME, gives percentile/percentage-point
  units and robust SEs, and keeps raw logit coefficients in separate CSV rows.
- Numerical-differentiation checks found the production `1e-5` parameter step
  stable against `1e-6`: inbreeding SE differences were at most `0.00004`
  percentage/percentile points. `1e-4` had larger discrepancies (up to about
  2.6% for PP(top 1%)); this is recorded as sensitivity to an overly large
  gradient step, not evidence of fit instability. Continuous age AMEs were
  stable across `1e-4`, `1e-5`, and `1e-6` predictor steps.
- OLS-HC3 M9 inbreeding benchmarks were `-1.84907`, `-1.02272`, `-0.42838`,
  and `-0.07309` in corresponding percentile/percentage-point units. These
  are descriptive linear-model comparisons, not a correctness criterion for
  the nonlinear AMEs; independent prediction and delta-method calculations
  above provide that validation. The original implementation has no
  Fractional-Logit estimator to compare directly.

## Negative-binomial AME overflow correction

- The configured NB failure was traced to an estimation/prediction-coordinate
  mismatch. NB fitting centers `first_year`, but AME scenarios had been rebuilt
  from the uncentered source frame. For Model 9/10, this produced invalid or
  overflowing mean predictions, non-finite delta-method variances, and a later
  LaTeX formatting failure on `nan`/`inf` standard errors.
- NB AME scenarios now use the same centered predictor coordinate as the fit.
  The NB delta-method gradient uses the exact derivative of its log-link mean,
  avoiding repeated parameter perturbation predictions; the AME definition,
  fitted likelihood, covariance, model formula, and sample remain unchanged.
  Non-finite gradient/covariance/variance inputs now fail with an AME-specific
  error rather than emitting matmul warnings and failing later in formatting.
- Validation with the configured 7,842-row researcher artifact: one Model 9
  `P_top10` NB fit plus its 37 AMEs completed in 0.62 seconds, with centered
  `first_year` mean approximately zero and all estimates, SEs, and p-values
  finite. The NB Model 9 regression test independently checks the centered
  prediction frame and compares the analytic log-link delta SE to numerical
  finite differences.

## Estimator-specific LaTeX effect headings

- The main relative/absolute performance, Model 9 supplementary, and Model 10
  templates already choose `AME` versus `Coef.` per outcome column. Added an
  explicit effect-type header beneath each M1–M10 model heading in the compact
  model-comparison tables, so NB model columns are labeled `AME` while linear
  model columns remain labeled `Coef.`. Estimate selection, standard errors,
  significance stars, and existing table filenames are unchanged.
- The negative-binomial table test asserts one `AME` label per model column.
- The Model 10 comparison table now reports the inbreeding estimate, robust
  standard error, and p-value, followed by estimator and effect rows populated
  from each outcome's configured model family. For the documented OLS-HC3/NB
  configuration this identifies OLS coefficients and the NB AME explicitly;
  numerical estimates and inference are unchanged. Its caption and estimator-
  specific note follow the requested research-performance table wording.

## Regression robustness estimators

- The `huberfrac`, `huberppml`, and `nbppml` bundles are selected directly as
  values in `[regression.dependent_variables]`. For each selected outcome and
  configured formula specification, the bundle fits its two named estimators
  under distinct model IDs. Model 9 therefore keeps the same formula, ASJC
  controls, and complete-case sample as that outcome's existing Model 9.
  Huber RLM retains its HuberT covariance; fractional logit, PPML, and NBHC3
  bundle fits use HC3. Fractional logit rescales mwPR to [0, 1] for fitting.
  Nonlinear GLM outputs include discrete average effects for binary inbreeding.
- The existing `nb` estimator retains its previous covariance behavior. The new
  `nbhc3` estimator uses the same negative-binomial likelihood and fitting process
  but explicitly uses HC3 covariance for standard errors, confidence intervals,
  and Wald p-values. This keeps HC3 selection separate from the existing `nb`
  specification.

- Regression dependent-variable configuration selects one existing model family
  per outcome or one explicitly named two-estimator bundle; selecting a bundle
  does not implicitly fit another estimator. Supported additions are
  `huber`, `fractional_logit`, and `ppml`. Huber uses the existing configured
  formula, covariates, and analytical sample. Fractional logit is limited to
  relative outcomes; PPML is limited to absolute outcomes. Both GLM estimators
  use HC3 covariance by default and reject non-converged or non-finite fits.
- Selecting a bundle in `[regression.dependent_variables]` replaces that
  outcome's previous model-family selection for all configured model
  specifications; it does not retain the previous estimator in parallel.
  This is the requested configuration behavior. For example,
  `pp_top_10 = "huberfrac"` fits Huber and fractional logit for that outcome.
- Bundles use the existing LaTeX filenames and layouts. The standard tables
  select one presentation estimator per outcome (Huber for `huberfrac` and
  `huberppml`; NBHC3 for `nbppml`), retaining the established one-fit-per-outcome
  table structure. Companion fractional-logit and PPML fits remain in the
  robustness CSV outputs rather than producing alternate suffixed LaTeX tables.
- Fractional-logit mwPR is divided by 100 for fitting because its input is stored
  on a 0–100 scale; the three configured top-percentile shares are already on
  the 0–1 scale. The academic-inbreeding marginal effect is a discrete mean
  prediction contrast (inbred versus non-inbred) with delta-method inference;
  mwPR marginal effects are reported on the original 0–100 scale.
- OLS and OLS-HC3 retain their existing coefficient/fit-statistics files. The
  selected non-OLS models write coefficient (including marginal-effect rows)
  and fit-statistics tables to separate robustness files, all beneath the
  user-configured `[regression.outputs].directory`. No estimator-specific path
  declaration duplicates the canonical output directory.
- RLM does not provide likelihood-based AIC/BIC. Such values remain missing;
  best-by-AIC/BIC reporting now returns an empty selection when every candidate
  metric is missing rather than failing the regression output write. No AIC/BIC
  values are fabricated for Huber fits.
- This is an additive estimator capability and changes results only when the
  user selects one of the added model types.
- Phase 4 was approved and completed with focused independent statsmodels
  references: Huber coefficients/standard errors match formula-API RLM with
  `HuberT`; fractional-logit and PPML coefficients, HC3 errors, confidence
  intervals, and p-values match direct Binomial-logit and Poisson GLMs. Tests
  also cover mwPR scaling, relative/absolute estimator restrictions, identical
  complete-case samples, nonlinear AMEs and delta-method inference for both GLMs,
  PPML nonnegative/convergence/finite-value checks, unchanged OLS-HC3 behavior,
  and output/config separation. `uv run pytest tests/regression/` passed **82
  tests** (16 warnings: statsmodels' documented GLM BIC `FutureWarning` and two
  existing ZINB optimizer `RuntimeWarning`s).
- Ruff and the complete pre-commit hook set, including mypy, passed on all
  changed files. A direct `uv run mypy` invocation is unavailable in the
  project environment; the configured pre-commit mypy hook succeeded. The
  default full-suite `uv run pytest` cannot collect because two test modules
  named `test_workflow.py` collide during import. Using
  `uv run pytest --import-mode=importlib` collected and ran the suite: **329
  passed, 2 failed**. Both failures are in `tests/pipeline/test_dependencies.py`
  and concern upstream dependency execution; no regression test failed. These
  unrelated failures were left unchanged.
- Created and independently executed these full pipeline configurations:
  `config/pipeline_regression_ols_hc3.toml`, `pipeline_regression_huber.toml`,
  `pipeline_regression_fractional_logit.toml`, and `pipeline_regression_ppml.toml`.
  They preserve all base settings except estimator selections and the
  regression output directory. OLS-HC3 was selected for all eight primary
  outcomes, including publications; fractional logit includes only the three
  top-percentile shares; PPML includes publications and the three absolute
  top-percentile counts. All four commands completed successfully. Their
  generated regression-input CSVs have identical SHA-256
  `6fdc85ee52a38ac77b7c5deb927e0000c923c2bb4a1024f8f2fa767c45e16733`, and all
  shared Model 9 formulas match exactly. Model 9 has 7,842 observations for
  every fitted outcome in every configuration.
- GLM convergence guards passed for all 30 fractional-logit and 40 PPML fits.
  A diagnostic re-fit of the 80 Huber models checked statsmodels RLM's default
  deviance tolerance (`1e-8`) against its final two deviance values: 66 converged
  before the 50-iteration limit; 14 reached the limit above tolerance. The
  affected pairs are `P_top1` m10/m3/m4/m5/m7/m8, `pp_top_1`
  m10/m2/m3/m4/m6/m8/m9, and `pp_top_5` m5. The Model 9 Huber effect for
  `pp_top_1` is therefore flagged as non-converged in the comparison output.
- Model 9 Supplementary Material tables, with linear coefficients distinguished
  from nonlinear average marginal effects, are saved at
  `data/output/regression/model9_estimator_comparisons/`. The directory also
  contains the per-model Huber convergence diagnostics. The four estimator
  results are isolated in their requested directories; the previous
  `data/output/regression/` main-paper coefficient and fit-statistics files
  retained their pre-run modification times and were not overwritten.

## Full-data name-disambiguation equivalence correction

- The migrated name-matching artifact correctly retains every field from the
  corrected researcher input; this artifact and the corrected source dataset
  were not changed. The source matcher, however, did not expose all of those
  fields on matched (`priority != 0`) rows to disambiguation. In particular,
  target-only `person_email` and researcher `orcid` values created email/ORCID
  pair evidence, while raw `[]` inferred-ASJC values prevented the source
  organization's ASJC inference from running.
- The candidate-author, article-author, citation-edge, affiliation, and impact
  inputs were SHA-256 identical between repositories. The 93,855-row researcher
  files differ only in `person_name`/`person_surname` for 884 rows; their
  `person_fullname` values and row order are identical. The target retained the
  corrected fields and added no matching-algorithm name swap.
- Disambiguation now creates a separate evidence copy of the matcher artifact:
  it applies the existing source-compatible ASJC inference helper; retains
  email only on pre-resolved (`priority == 0`) rows, as the source output did;
  and clears researcher ORCID, which the source matcher did not pass through.
  It does not mutate or rewrite the canonical name-matching artifact. No name
  swapping was added to matching; the corrected input remains authoritative.
- With identical current article/reference datasets, `PYTHONHASHSEED=0`, and
  five workers, the corrected target evidence projection matched the freshly
  generated original disambiguation outputs exactly across all tabular
  artifacts: 32,953 final selections, 13,847 pair decisions, 40,371 cluster
  rows, and 8,279 selection-score rows. Keys and all values matched.
- The existing target name-matching artifact's inferred codes, after applying
  the source helper to a copy, matched the source-generated codes for all
  32,953 researcher IDs. The full target disambiguation run then matched the
  fresh source run exactly despite retaining the corrected metadata in its
  upstream matcher artifact.
- Mobility was rerun to a temporary output directory with the matched
  disambiguation results. Its four output artifacts matched the fresh original
  outputs by row values, except one grant-number serialization difference in
  the mobility proxy (`DFF\\u2fff...` versus the corresponding Unicode
  character); article rows and all classification/history fields matched.
  Regression inputs exclude that field.
- Regression was rerun to temporary output using the temporary mobility
  artifact. The 7,842-row `author_regression_input.csv` matched the fresh
  original by researcher ID and every column value; the 653,163-row
  professor-article input also matched exactly. All 80 model/outcome fit rows
  had the same keys, formulas, and observation counts. Across all coefficient
  rows, maximum absolute differences were `1.271e-9` for estimates,
  `3.153e-11` for standard errors, `1.320e-9` for confidence interval bounds,
  and `2.124e-9` for p-values. These are floating-point differences, not
  estimator or specification changes. M9 used 7,842 observations for every configured
  outcome, with equivalent coefficients, standard errors, confidence intervals,
  and p-values.
- One regression invocation initially used a malformed temporary-config path
  replacement and wrote its newly generated outputs to the configured target
  directory. The previous 7,835-row target regression output was regenerated
  from the unchanged prior target mobility artifact and restored. The corrected
  7,842-row regression verification is in
  `/tmp/opencode/equivalent_pipeline/regression_corrected/`; mobility outputs
  are in `/tmp/opencode/equivalent_pipeline/mobility/`, and disambiguation
  outputs are in `/tmp/opencode/disambiguation_equivalence/target_projected/`.
  No original-repository files were written, and no commit was created.

## Name-disambiguation equivalence remediation

- Controlled source/target execution found that the migrated workflow passed every
  column from the article parquet into evidence indexing, while the source loader
  projects to its declared article-evidence column list. Unprojected funding and
  other metadata columns were therefore interpreted as additional grant evidence,
  changing pair scores and researcher selections. The migrated workflow now applies
  the source projection before validating the article-evidence contract; unknown
  parquet fields remain ignored as in the source. Researcher metadata and the
  name-matching output contract are not altered by this correction.
- The observed affiliation float/integer representation difference is normalized
  by affiliation-ID extraction to the same identifier string. It was not the cause
  of the decision differences; no blanket float coercion was added.
- With `PYTHONHASHSEED=0`, equal worker counts, the same matcher artifact, and the
  same article/reference paths, complete-data source/target outputs matched exactly:
  32,953 selections, 13,847 pair decisions, 40,371 cluster rows, and 7,876 final
  selection-score rows. Composite researcher/pair/author/cluster keys matched, and
  every output column had zero differences, including selected IDs, statuses,
  scores, confidence, labels, histories, and ASJC-derived fields.
- `gender_given_tokens` consumes a set returned by `given_tokens`, so Python hash
  randomization can change which name token is examined first. Unseeded independent
  runs consequently showed a small number of differing pair and selection results.
  The equivalence claim is therefore for controlled executions using the same
  `PYTHONHASHSEED`; the source pipeline does not enforce a seed. This existing
  nondeterminism is documented rather than hidden by altering the original.

## Scope

The source project is `name-matching-playground`, managed with Poetry. The target is this repository, `academic-inbreeding`. Migration is incremental: dependency/build setup comes first; application code and tests are not part of Phase 1.

## Phase 1 — Poetry to uv

### Results

- Replaced Poetry project configuration with PEP 621 project metadata in `pyproject.toml` and the `uv_build` backend.
- Carried over all 33 direct dependency declarations and their version bounds from the Poetry project. The source declares no dependency groups or optional extras, so dependencies remain in the main project dependency list.
- Preserved the source Python support range. Poetry's `>=3.12,<3.14.1 || >3.14.1,<4.0` is represented as the equivalent PEP 440 constraint `>=3.12,<4.0,!=3.14.1`.
- Added a standard `src/academic_inbreeding/` package scaffold; no application modules have been migrated.
- Added `.gitignore` entries for Python artifacts, uv's `.venv`, test/tool caches, and PyCharm's `.idea` directory.
- Carried over the existing `.pre-commit-config.yaml` and `mypy.ini` tool configurations.
- `uv.lock` is generated by uv and is the target's dependency lockfile. The source `poetry.lock` is not copied.
- The uv lockfile preserves the Poetry-locked version of all 33 direct dependencies while leaving the original version ranges in `pyproject.toml` unchanged.

### Deferred or changed configuration

- Poetry-specific `[tool.poetry]` package declarations are replaced by the `uv_build` src-layout backend configuration.
- Console scripts from the source project are deferred because their referenced `scripts` package is not part of this phase; registering them before migrating that code would create broken entry points.
- The source's `readme = "README.md"` is omitted because the target does not yet contain that file.
- The source had no dependency groups, optional dependency declarations, or tool-specific sections in `pyproject.toml` to transfer. Existing standalone tool configurations are retained as separate files.

### Verification

- `uv lock` completed successfully and resolved 86 packages.
- `uv sync` completed successfully under CPython 3.13.9, built and installed the local package, and installed 85 packages into `.venv`.
- `uv lock --check` confirmed the lockfile is current, and `uv pip check` reported all installed packages compatible.
- No dependency resolution conflicts were reported.
- Comparing lock selections, all 33 direct dependency versions match `poetry.lock`. The uv resolver selected newer versions for 30 shared transitive packages (for example, `numpy` 2.5.3 instead of 2.4.1) and the resolved graph has 86 packages versus 83 in Poetry's lock. These are allowed by the declared dependency ranges; transitive runtime compatibility remains to be covered by later application tests.

## Later phases

Migrate application modules and tests only after the dependency foundation is verified. Preserve matching and disambiguation behavior, and compare outputs against the source implementation using representative fixtures and datasets before considering each stage complete.

### Stage migration status and approved order

The project owner has decided that `mobility_impact` is excluded from the
migration scope. It is not pending, is not required before regression, and no
`mobility_impact` migration is planned. Historical references to this stage
below are retained where they document prior dependency analysis.

| Stage | Status |
| --- | --- |
| `name_matching` | COMPLETED |
| `name_disambiguation` | COMPLETED |
| `mobility` | COMPLETED |
| `mobility_impact` | EXCLUDED |
| `regression` | COMPLETED (PHASE 4) |
| `statistical_matching` | COMPLETED (PHASE 4) |

The approved migration order is `name_matching` → `name_disambiguation` →
`mobility` → `regression` → `statistical_matching`, one stage per migration.
This is a migration sequence only; actual artifact dependencies must be
established by inspecting implementations.

## Phase 2 — `name_matching` code migration

### Status

- The matching package, stage-specific TOML configuration, and runnable CLI workflow have been migrated. Phase 4 tests for `name_matching` have now been migrated and run; later stages remain untouched.
- The original repository remains read-only. Existing target changes were left untouched.

### Architectural decisions

- Added the stage package `src/acib/name_matching/` for matching algorithms and its TOML configuration model.
- Added `scripts/match_author_names.py` as the legacy-style script entry point, the `match-author-names` project command, and `python -m acib.name_matching`. Added a target `config/pipeline.toml` containing only the original `[name_matching]` table; later stages' settings are intentionally not copied in this stage.
- Added `src/acib/common/names.py` for name normalization, splitting, surname compatibility/blocking, given-name compatibility, aliases, and compound matching. These primitives have no stage imports and use only the standard library plus optional `regex` (with the original standard-library fallback).
- Kept stage-specific token fuzzy matching, dataframe joins, priority assignment, and result structures inside `acib.name_matching`.
- Did not add imports from future pipeline stage packages. Existing target repository uses an `acib` package scaffold; no `AGENTS.md` or `docs/ARCHITECTURE.md` was present during this migration.

### Implementation and behavior notes

- Migrated the exact, containment, guarded fuzzy, and conservative compound matching paths, including match priorities (1–4), the 90 fuzzy score threshold, the two-token and 0.8 overlap guards, and researcher/author de-duplication.
- Added the name-matching TOML settings model and loader. It retains the source keys/defaults, expands environment/user paths, and resolves relative input/output paths from the project root.
- The workflow supports Scopus inputs, Scopus parent-affiliation normalization, existing Scopus ID passthrough, matching, stable output sorting, and CSV writing. It does not yet reproduce the complete Scopus name-formatting path or inferred ASJC enrichment; these remain code-migration limitations, and full behavior equivalence is not claimed.
- Default and relative config paths resolve from the project root rather than the process working directory, so the script works when launched with `scripts/` as its working directory. The workflow refuses to write output beneath the read-only `name-matching-playground` sibling repository.
- Shared name behavior was checked against its original consumers: the disambiguation parser, name scoring, gender utilities, and matching implementation use the same normalization/splitting/compatibility semantics. The shared module is independent of pandas and stage configuration.
- Reuse opportunity for the next stage: migrate disambiguation imports of `normalize_person_name`, `split_name`, `given_name_compatible`, and `surname_compatible` to `acib.common.names` rather than recopying those implementations. Future-stage organization/path utilities were not generalized.
- Cross-stage schema fields confirmed against source consumers have been moved to `src/acib/constants.py`; `acib.name_matching.constants` now imports those shared definitions rather than redeclaring them. Fields whose semantics are stage-specific remain in the name-matching constants module.
- The name-matching workflow intentionally separates researcher input metadata from a minimal matching projection. It validates non-null unique `sc_id` values and restores metadata by joining only match-owned result columns, so original metadata takes explicit precedence and merge suffix columns are not produced. The configured existing-Scopus-ID input column is omitted from output; derived `researcher_name` and candidate-author fields remain separate from researcher metadata. This improves the original output behavior, which dropped researcher metadata during its matching-column projection, without changing match priorities or organization filtering.
- Restored structured `structlog` progress events for input loading, existing-ID passthrough counts, per-organization matching results, output writing, and total runtime. The CLI logging configuration alone was not sufficient: the migration had omitted the original workflow and matching logger calls.
- Test migration exposed that target organization aggregation raised when all per-organization frames of a result type were empty. Restored the source implementation's safe aggregation behavior, returning an empty DataFrame instead; this does not alter matching criteria or priorities.
- Fixed the confirmed empty-researcher compound-match bug: organization groups with no researchers are skipped, and `compound_name_merge` now returns an empty frame when either side is empty. In the configured full output, the old implementation had 719 unlinked priority-4 rows, all mapped to FH Johanneum, for which the researcher input contained no researchers. The same early-return bug existed in the original implementation.
- Added reusable Pandera DataFrameModel contracts in `src/acib/schemas/` for researchers, Scopus author/affiliation input, and matching output. Input and output validation use the returned DataFrames; additional metadata remains allowed. Parent-affiliation ID CSV values are parsed lexically into nullable `Int64` without an intermediate float conversion; matching output normalizes numeric Scopus author IDs to `int64` after validation.
- Migrated the relevant matching, affiliation-preparation, existing-ID, and mapping assertions from the original name-matching tests into `tests/name_matching/`, and added a temporary-file end-to-end workflow test under `tests/integration/`. Original tests for inferred ASJC enrichment and formatting helpers were not ported because those features are not implemented in the target stage.
- Matching regression expectations cover exact, containment, guarded fuzzy, and compound priorities and organization-filter decisions, based on the original matching tests. The integration case verifies original metadata restoration, Scopus-ID passthrough, output row count, and the organization-filtered match. Full-dataset comparison against the original was not run.
- A direct representative-fixture comparison of original and migrated `simple_name_merge` results produced identical `(sc_id, eid, priority)` tuples for exact, containment, fuzzy, and compound cases.

### Approved full-data name-preparation correction

- After a same-input full-data source/target comparison, the user approved corrections to researcher and candidate-author name preparation. Complete structured given-name/surname fields take precedence; incomplete fields fall back to usable display-name fields. Names with no recoverable given or surname are excluded, and normalized names use exactly one space after the comma.
- Candidate-author fallback specifically checks `name` when `author_name` is empty. This restores 9,730 source-retained candidate rows observed in the controlled comparison; the target previously dropped them. The target previously retained 1,049 rows that source preparation rejected, including surname-only values; those are now excluded unless a fallback yields both components.
- Structured researcher fields preserve compound surnames that the source display-name parser split incorrectly (for example, `Diez Sanjuan, Lucia`). This is an intentional behavior difference approved by the user, not a claim that every changed match is more accurate.
- Full-data name-preparation comparison: source retained 92,161 researcher rows and target retained 92,162; 5,295 shared researcher names differed after whitespace normalization. Source retained 678,263 candidate rows and target retained 678,757; among 678,262 shared source-row identities, 64,881 normalized candidate names differed. One source-only candidate and 495 target-only candidates remained; target-only names now require both components. Comparison files are in `/tmp/opencode/name_preparation_comparison/`.
- The controlled full-data matching rerun used identical raw inputs and `max_workers = 1`, with source and target outputs written separately under `/tmp/opencode/name_matching_comparison/`. The source rerun exactly matched the saved playground output by SHA-256. The corrected target produced 41,029 rows versus 40,371 source rows. Of 39,615 shared researcher/author pairs, 3,106 had different priorities; 864 shared researchers had different candidate-author sets. Thus source/target full-data equivalence is not established; changed decisions include the approved structured-name behavior and require case-level assessment before target results are treated as source-equivalent.
- Added regression coverage for structured compound surnames, display-name fallback, candidate `name` fallback when `author_name` is empty, initials-only structured names, rejection of incomplete names, and end-to-end metadata/output behavior. `uv run pytest tests/name_matching/ tests/integration/test_name_matching_workflow.py` passed (40 tests; six existing pandas `FutureWarning`s).

### Verification

- `./.venv/bin/ruff check src/acib/common src/acib/name_matching` passed.
- `uv run --no-project --with mypy mypy --config-file mypy.ini src/acib/common src/acib/name_matching` passed with no type errors. Mypy reported unused module-specific configuration sections because this scoped invocation did not analyze those modules.
- The module entry point's `--help` smoke check passed using temporary package injection for missing runtime dependencies; it did not run the matching pipeline or validate output equivalence.
- Configured `tool.uv.build-backend.module-name = "acib"` to match the existing package scaffold. `uv build --wheel --out-dir /tmp/opencode` and `uv run /home/chri/PycharmProjects/academic-inbreeding/.venv/bin/python /home/chri/PycharmProjects/academic-inbreeding/scripts/match_author_names.py --help` both completed successfully. The help invocation verifies startup only; it does not execute or verify pipeline outputs.
- `uv run pytest tests/name_matching/` passed (25 tests); `uv run pytest tests/integration/` passed (4 tests). These focused tests emitted pandas `FutureWarning`s from `combine_first` when affiliation fields contain empty entries.
- `./.venv/bin/ruff check src/acib/name_matching tests/name_matching tests/integration` passed. `uv run --no-project --with mypy mypy --config-file mypy.ini src/acib/name_matching tests/name_matching tests/integration` passed with no type errors; mypy reported unused unrelated configuration sections.

### Full-data output-equivalence reconciliation

- The corrected researcher CSV is retained as the authoritative metadata source. Matching results are joined without overlapping match-side researcher fields, and the configured `scopus_author_id` control column is excluded from the resolved output. Metadata is read lexically for output so values such as Scopus IDs and scores retain their CSV spelling.
- Matched candidate output now retains the source's historical identifier, affiliation IDs/parent IDs, affiliation name, current-affiliation IDs, and all matching fields in source column order. Pre-resolved rows use the source-compatible `is_alternative=False` and `initials=False` values; other `is_alternative` values remain null. Affiliation IDs retain the source CSV `.0` spelling on serialization.
- The source full-data baseline and target output were compared as serialized string-valued CSVs after the final four-worker run. Both contain **40,371 rows and the same 57 columns in the same order**. Pair membership, row-key order, and priorities match exactly: zero assignment differences and zero priority differences. Every non-metadata output value matches the source baseline; all retained metadata values match the corrected input exactly. No unexplained column or value differences remain.
- The intentional metadata-retention differences versus source output occur only in researcher metadata columns; the per-column difference counts are: `person_fullname` 40,371; `person_role` 31,116; `person_link` 38,506; `person_functions` 1,142; `institute_link` 25,329; `person_grade` 8,503; `person_email` 14,112; `person_phone` 6,512; `person_location_link` 250; `orcid` 2,033; `scopus` 316; `organization_names` 38,679; `organization_names_relevant` 38,679; `inferred_asjc_evidence` 38,679; `inferred_asjc_codes` 2,984; `inferred_asjc_parent_codes` 38,679; `inferred_asjc_labels` 38,691; `inferred_asjc_parent_labels` 38,679; `has_inferred_asjc` 38,679; `inferred_asjc_max_score` 29,976; `inferred_asjc_confidence` 38,679; the corresponding seven JSON metadata columns have 38,679 differences each; `person_surname` 38,687; and `person_name` 38,762. For every one of these columns, target values were checked against the corrected researcher input and had zero differences.
- Output reconciliation exposed blank/empty author-name edge cases in unit fixtures. Candidate names now fall back to the source name when the alternate display-name field is blank, and empty candidate frames are safe in exact/fuzzy/prefix matching. Focused name-matching and integration tests pass (40 tests). The full target suite ran: **309 passed, 2 failed**. Both failures are in pre-existing pipeline dependency tests (`tests/pipeline/test_dependencies.py`) concerning missing external-input checks and stage invocation; no unrelated dependency code was changed. The full-suite failures do not concern matching output equivalence.

## Phase 2 — `name_disambiguation` code migration

### Status

- The disambiguation implementation, matcher-output dependency, publication-evidence inputs, configured artifact outputs, and minimal matcher-to-disambiguation runner have been migrated. Phase 3 static checks passed; Phase 4 test migration for this stage is now complete.
- The original repository's tests and source files remain unmodified. Target disambiguation tests are organized under `tests/name_disambiguation/`; one full workflow test is under `tests/integration/`.

### Architectural decisions and implementation

- Kept disambiguation algorithms and stage options in `src/acib/name_disambiguation/`; added six explicit outputs (pair decisions, cluster memberships, final selections, selection scores, text reports, and JSONL reports).
- Added `scripts/disambiguate_author_names.py` to pair with `scripts/match_author_names.py`; retained `scripts/run_disambiguation.py` as a backward-compatible alias. The `disambiguate-author-names` command and `python -m acib.name_disambiguation` entry points are also available.
- The stage consumes `name_matching.resolved_matches_output_file_path` as its
  upstream input. Shared external researcher, Scopus-author, article-author,
  and article-reference datasets are represented in `[data]`. Outputs are
  declared individually under `[name_disambiguation.outputs]`; duplicate and
  obsolete stage-local path declarations are rejected rather than translated.
- Implemented a minimal two-stage runner that validates and reuses a present matcher artifact, or runs matching if it is absent, before invoking disambiguation. This is intentionally not a generic pipeline framework.
- Reused `acib.common.names` for name operations and `acib.common.titles` for final-selection title classification. Kept legacy-compatible name normalization in the disambiguation package where whitespace/null semantics differ from the shared matcher helper.
- Added Pandera contracts for prepared author rows, publication evidence and citation edges, four tabular disambiguation outputs, and final researcher selections. Validation is at stage boundaries; unrelated researcher/author metadata is retained.
- Consolidated stable DataFrame field names through stage constants and reused confirmed shared or matching-stage fields rather than duplicating definitions. Temporary clustering columns remain private to the clustering module; decision/status values and nested report-object keys remain values, not DataFrame constants. Added Sphinx/reStructuredText documentation to the public disambiguation APIs and typed models without changing calculations.
- Ported scoring, pair clustering, selection, reporting, and output enrichment including author history and current affiliations. Preserved the 6.0/3.0 default pair thresholds, selection rules/statuses, output artifact names, and legacy report renderers. `enable_reporting` controls console reports; configured report files are still written with the other artifacts.
- Central configuration resolves relative paths from the project root and rejects conflicting duplicate declarations. Existing legacy key spellings are accepted as aliases so migrated TOML can load without changing scientific or path interpretation.

### Reuse opportunities and limitations

- Later mobility stages can reuse `common.titles` and shared author/researcher identity constants. Publication-evidence schemas and citation-edge loading may be reusable only after comparing their semantics with later-stage inputs; no broader evidence abstraction was introduced.
- The shared name-normalization helper is not a drop-in replacement for disambiguation normalization because whitespace and missing-value semantics differ.
- Article datasets and the matching artifact are external inputs; no full-dataset execution or source-output comparison was performed in this code phase. Existing artifacts are checked for presence and schema only; freshness/staleness tracking is not implemented.
- No intentional scientific or output behavior changes are approved or claimed. Input/output validation may reject malformed data earlier than the legacy implementation; scoring and classification behavior were not intentionally changed.

### Verification and approval gate

- `uv run ruff check src/acib/name_disambiguation src/acib/pipeline src/acib/schemas` passed.
- `uv run --no-project --with mypy mypy --config-file mypy.ini src/acib/name_disambiguation src/acib/pipeline src/acib/schemas` passed with no type errors. Mypy printed notices for unused module-specific config sections outside this analysis scope.
- `uv run python -m compileall -q src/acib/name_disambiguation src/acib/pipeline src/acib/schemas` passed. A runtime import/config smoke check resolved the configured paths and thresholds successfully.
- Migrated and grouped the original 87 disambiguation test methods into 66 pytest tests under `tests/name_disambiguation/`, preserving their scientific assertions for parsing, scoring, threshold/classification behavior, gender and temporal blockers, ASJC selection, clustering, histories, reporting, and configuration. Added edge coverage for nullable integer inputs, malformed/missing publication evidence, empty batches, output overwrite checks, and all six output artifacts.
- Added `tests/integration/test_name_disambiguation_workflow.py` with small temporary CSV inputs to verify matcher artifact loading, nullable affiliation IDs, publication evidence, scoring, selection, schema validation, and all six outputs. No tests for other unmigrated stages were added or changed.
- The source-referenced `data/disambiguation/author.csv` fixture is absent from the original repository. Three source tests depending on it were adapted to a small inline profile fixture; the original asserted cluster shape and representative ID checks remain, but full fixture-specific source comparison could not be performed.
- Ran the original test file read-only with bytecode and pytest cache writing disabled: `tests/test_disambiguation.py` collected 87 tests, with 82 passing and 5 failing. Three failures were `FileNotFoundError` for the absent `author.csv`. Two source assertions disagree with the source implementation: the initial-overlap case expects `given_conflict_with_initials_conflict` but emits `given_conflict_with_initial_overlap`; the lowered-threshold case expects `same_author` but returns `possible_same_author` because the initial-only risk cap takes precedence. Migrated assertions use the observed source behavior, and the original test file remains untouched.
- A direct representative score comparison between source and target produced identical decision labels, raw scores, confidence, and component/weight pairs. This does not substitute for comparison on the unavailable source fixture.
- `uv run pytest tests/name_disambiguation/` passed (70 tests); `uv run pytest tests/integration/` passed (5 tests, including the existing matching integration tests). The combined affected-scope command `uv run pytest tests/name_disambiguation/ tests/integration/ tests/schemas/ tests/name_matching/` passed (131 tests). It emitted six existing pandas `FutureWarning`s from name-matching affiliation aggregation; disambiguation tests had no warnings.
- Pre-commit hooks (Flake8, Black, mypy, isort, and repository checks), Ruff, and explicit mypy checks passed for the migrated test files. Mypy emitted notices about unused unrelated configuration sections. No commits were created.

## Phase 2 — `mobility` code migration

### Configuration ownership and independent execution

- Centralized all external inputs under `[data]`: `researcher_file_path`,
  `candidate_authors_file`, `article_author_file_path`,
  `article_reference_file_path`, and `author_affiliations_file_path`. The
  GeoPackage remains the verified source dataset at
  `/home/chri/PycharmProjects/name-matching-playground/data/input/author_affiliation.gpkg`;
  it was not replaced with a different file.
- Mobility resolves its author-selection input from
  `name_disambiguation.outputs.final_author_selection`, researcher metadata from
  `name_matching.resolved_matches_output_file_path`, and article-author and
  GeoPackage inputs from `[data]`. No generated upstream input points to the
  legacy repository. The checked-in sample `[data]` paths point to the verified
  external input datasets under the original repository's `data/input/`
  directory. Input locations are user-configurable and may be absolute or
  relative paths anywhere accessible, including that directory; copying or
  relocating datasets is not required. Application code has no hardcoded
  external input paths.
- Removed duplicate/obsolete input and output-directory declarations from
  `[mobility]`. Matching and disambiguation loaders now require canonical
  shared-data and explicit stage-output declarations; redundant legacy path
  keys are rejected rather than retained as indefinite aliases.
- Defined the four source artifacts individually under `[mobility.outputs]`:
  `author_affiliations.csv`, `author_affiliations_unique.csv`,
  `author_inbreeding_proxy.csv`, and `author_articles.csv`. Mobility checks all
  required inputs and identifies the producing stage for missing generated
  artifacts. It never invokes matching or disambiguation.
- The optional source-runner file `professor_author_profiles.csv` is no longer
  written. This output behavior change was explicitly approved by the user;
  professor profiles remain available in memory for analysis.
- Relative paths continue resolving against the target repository root, not
  the current working directory. No general runner or mobility orchestrator
  was introduced. Configuration architecture and future-stage dependencies are
  described in `docs/ARCHITECTURE.md`; permanent rules were added to
  `.opencode/agents/migration.md` and a concise root `AGENTS.md` was created.

### Implementation and validation

- Reused `acib.common.titles`, shared researcher/author/affiliation fields,
  disambiguation selection contracts, and disambiguation ASJC parsing. Mobility
  keeps its GeoPackage contract and analysis fields stage-local.
- Kept the GeoPackage input field `type` distinct from the mobility output field
  `affiliation_type`, restoring the original output schema rather than leaking
  the input column name into the author-affiliation artifact.
- The workflow applies existing Pandera contracts to the final selection,
  matching output, article-author input, and GeoPackage. Added meaningful
  mobility output contracts for the four written artifacts. ID parsing remains
  non-coercing: final-selection identifiers accept exact integer, integral
  float, or digit-string representations without rewriting their input dtype;
  fractional and malformed identifiers are rejected. This accommodates the
  configured CSV's actual inferred ID dtypes without changing IDs.
- Reproduced the mixed-type warning on the configured target matching artifact:
  columns 27 (`person_functions`), 28 (`institute_link`), 34
  (`person_location_link`), and 35 (`orcid`). These are serialized/free-text,
  link, or identifier strings—not researcher IDs, author IDs, priorities, or
  affiliation IDs. Mobility now uses stable CSV inference and explicit string
  dtypes for those four fields; leading zeros in ORCID values are preserved.
  The schema check on the configured target metadata also passed.
- The configured-data run initially failed in parallel preparation because
  calling `set_index(column_name)` removed the temporary chunk-index column
  before a subsequent `drop`. The original wrapper sets the index from the
  Series and then drops the retained column. The target now matches that
  behavior. Regression tests cover the index-preserving parallel path, the
  missing-index fallback, and a 5,001-row sequential/parallel preparation
  comparison.
- All 38 source test functions were migrated to `tests/mobility/` with their
  scientific assertions intact. Added configuration/dependency, missing-input,
  schema/invalid-ID, DtypeWarning, chunk-index, metadata-join, and standalone
  workflow coverage. The workflow integration test verifies all four
  configured artifacts and confirms the professor-profile file is absent.

### Scientific regression and end-to-end results

- The two original mobility test files were run read-only with bytecode/cache
  writing disabled and `PYTHONPATH` pointed at the original source. Result:
  **38 passed**. The original repository was not modified.
- The migrated and added mobility tests passed, and the complete target suite
  passed: `uv run pytest` — **195 passed**, 24 warnings. Warnings were existing
  pandas `FutureWarning`s in mobility/name-matching affiliation aggregation and
  a multiprocessing `fork()` deprecation warning; no tests failed.
- Executed the configured full mobility analysis using the verified GeoPackage
  and current target-generated upstream artifacts, with all four outputs
  redirected to a temporary directory. Inputs: 33,275 author-selection rows,
  40,984 researcher-metadata rows, 690,089 article rows, and 118,198 GeoPackage
  affiliation rows. The run processed 781,091 article-author rows and 8,238
  author IDs, generated four schema-valid artifacts (46,564 author-affiliation
  rows; 9,267 unique affiliations; 8,279 author summaries; 480,971 author
  articles), and wrote no professor-profile file. Production mobility outputs
  were empty before the run and were not overwritten.
- Ran the original mobility algorithm directly against those same target input
  artifacts and the same scientific settings, writing only to a separate
  temporary directory. The four artifacts had matching columns, row identities,
  and values: zero differences in inbreeding classifications/statuses,
  publication-history and mobility indicators, affiliation histories, unique
  affiliations, and all compared article fields. The source reader emitted the
  mixed-type warning described above. Existing source output files from
  September were also inspected but were not used as the equivalence baseline:
  their upstream inputs are older and the files are not freshness-verified.

### Quality checks and remaining limitations

- `uv run pytest` passed (195 tests).
- `uv run ruff check .` passed after making Ruff and isort configuration
  consistent with `combine_as_imports = true`; four existing disambiguation
  import blocks were reformatted only to align the two linters.
- `uv run --no-project --with mypy mypy --config-file mypy.ini src/acib/`
  passed with no type errors. The specifically requested `uv run mypy` command
  is unavailable because mypy is not a project runtime dependency; the scoped
  uv invocation and pre-commit mypy hook both ran successfully. Mypy emitted
  notices for unrelated unused config sections.
- The first `uv run pre-commit run --all-files` attempt found two Flake8 line
  length issues and Black reformatted `src/acib/schemas/researchers.py` and
  `src/acib/pipeline/config.py`. Those lines were fixed; the full test and
  static checks were rerun after formatting. The final
  `uv run pre-commit run --all-files` passed all hooks: whitespace/end-of-file,
  YAML/AST/JSON/TOML/private-key checks, license insertion, Flake8, Black, mypy,
  and isort.
- The stored legacy mobility outputs do not match the current pipeline inputs
  and are not treated as current results. Same-input current source-vs-target
  execution showed exact equivalence. Stale-artifact detection remains
  unimplemented and is not implied by file existence or schema validity.
- The current configured mobility run and direct source comparison succeeded
  using the external paths in `[data]`. The configured sample uses files under
  the legacy repository, but this is a user-controlled external-data location,
  not a generated-artifact dependency or migration blocker. No dataset copies
  were made.
- No later stage was migrated, no scientific method was changed, and no commit
  was created.

### Approved mobility output follow-up

- The user approved changing `author_articles.csv` from one row per EID to one
  row per mobility author profile and EID. The output now aggregates all
  resolved `author_afids` and original `unmapped_id` values into semicolon-
  delimited columns while retaining the other article fields. Existing
  author/affiliation association and parent-resolution logic remains unchanged;
  this is an output-contract change only. Regression does not consume this
  artifact and continues reading the configured external article dataset.
- The user also approved stopping production of `author_affiliations_unique.csv`.
  Mobility now declares/writes three outputs: `author_affiliations.csv`,
  `author_inbreeding_proxy.csv`, and `author_articles.csv`. Existing files in
  user data directories are not deleted by this code change.
- After explicit Phase 4 approval, the existing mobility tests were updated to
  assert the changed output contract. `uv run pytest tests/mobility/`
  passed (60 tests) and `uv run pytest tests/integration/` passed (24 tests).
  Coverage checks grouped affiliations by author/EID, preserves distinct
  authors on the same EID, serializes resolved and original IDs, validates the
  schema, and confirms the removed unique-affiliation artifact is not written.
- Ran the original mobility tests read-only with bytecode and pytest cache
  writing disabled: 32 passed. A representative source/target comparison of
  author affiliation histories and scientific summary fields
  (`is_inbreeding_proxy`, status, publication-history years, and first/last
  affiliation sets) was identical. The intended CSV contract differences are
  tested separately.
- Final `uv run ruff check src/acib/mobility tests/mobility
  tests/integration/test_mobility_workflow.py` passed;
  `uv run --no-project --with mypy mypy --config-file mypy.ini
  src/acib/mobility tests/mobility tests/integration/test_mobility_workflow.py`
  passed with no type errors (mypy reported existing unused-config notices).
  Pre-commit passed all applicable hooks on changed files. Tests emitted 16
  existing mobility warnings and five warnings in integration tests: pandas
  `FutureWarning`s for affiliation-parent aggregation and multiprocessing's
  `fork()` deprecation warning. No tests failed.

## Phase 2 — `regression` code migration

### Configuration ownership and dependencies

- Regression's external article-author table, article-impact indicators, and
  optional disruption-indicator file are resolved from their canonical `[data]`
  entries. The author-level inbreeding input is resolved directly from
  `mobility.outputs.author_inbreeding_proxy`; it is not duplicated under
  `[regression]` and no upstream stage is automatically executed.
- Regression resolves relative external paths from the target project root and
  retains configured absolute external paths. Generated files never fall back
  to a legacy-repository output path.
- `article_impact_file_path` is an external dataset, not an output of the
  excluded `mobility_impact` stage. Regression has no dependency on
  `mobility_impact`.
- Regression outputs are explicitly declared in `[regression.outputs]`:
  `regression_input`, `professor_article_input`, three build diagnostics, model
  coefficients/statistics/descriptives, AIC/BIC reports, formulas, summaries,
  and the `latex_dir`/`plots_dir` collection roots. The future statistical-
  matching stage consumes `regression.outputs.regression_input` rather than
  declaring a duplicate input path.
- Regression's outcome groupings and aliases were moved from the source's
  `statistical_matching.dependent_variables` coupling to
  `[regression.outcome_metadata]`. The configured relative/absolute groups and
  labels are retained, and regression can run without the statistical-matching
  section. No statistical-matching code or configuration was migrated.
- The site-specific `run_regression.sh` Slurm/Apptainer wrapper was not
  migrated. The Python `regression_analysis.py` and `build_regression_input.py`
  entry points, `python -m acib.regression`, and the `regression-analysis` and
  `build-regression-input` commands are available without cluster-specific
  paths.

### Implementation and scientific behavior

- Added the separate `src/acib/regression/` package for input construction,
  fractional ASJC field shares, model fitting/diagnostics, tables, LaTeX, and
  the regression-suite plots. Reused `acib.common.titles`, mobility's author
  and ASJC summary helpers, and disambiguation's ASJC taxonomy where source
  semantics match.
- Regression validates the mobility proxy, the shared article-author input,
  optional impact/disruption data, generated researcher-level and
  professor-article tables, and coefficient/fit/descriptive outputs at stage
  boundaries. The same source article-author and mobility-proxy contracts are
  reused for those producer artifacts; regression-specific tables have their
  own contracts.
- Preserved the ten model specifications and configured model families: seven
  OLS outcomes and Negative Binomial for `n_public`. Also preserved
  `min_publications = 1`, `rebuild_input = true`, `drop_missing = true`,
  professor-title filtering, empty configured title exclusions, the 1996–2023
  impact eligibility window, fractional top-percentile aggregation, ASJC field
  shares/reference selection, NB/ZINB diagnostic policy `error`, and gradient
  tolerance `0.0001`. No scientific method or model specification was changed.
- The source regression package's module-level documentation said the default
  was nine models, but its actual model constants define ten. The target
  documentation now correctly says ten; model terms remain unchanged.
- The `regression_input` output is the canonical future statistical-matching
  artifact. `professor_article_input` remains available for the regression
  suite's professor-title plot and existing analysis helpers; the two plot
  families are written only under the configured `plots_dir`.
- Approved regression plot fix: the Model 9 prediction plot's non-OLS guard now
  considers only the plot's configured outcomes. The unrelated `n_public` Model
  9 Negative Binomial fit no longer suppresses the four-outcome OLS prediction
  figure. The user reported that the original pipeline produced this plot; the
  regression output run showed Model 9 OLS fits and an NB `n_public` fit, while
  only the professor violin PNG was present. Verification of the fix through
  regression tests and a source-comparison run remains pending Phase 4 approval.
- Fixed generated files retain their source filenames: the two prepared input
  artifacts, three input diagnostics, coefficient and fit-statistics tables,
  descriptives, AIC/BIC outputs, the inbreeding comparison, formula text, and
  summary text. Model-dependent LaTeX and plot files remain grouped under their
  explicitly configured collection directories.
- The regression builder validates the shared mobility article-author schema,
  mobility author-proxy contract, configured external impact/disruption
  contracts, and its researcher-/professor-level output contracts. Model
  coefficient, fit-statistic, and descriptive artifacts are also validated at
  the write boundary. Missing mobility inputs report the mobility producer;
  regression never executes mobility.
- `RegressionConfig` can either rebuild from external/mobility inputs or load
  its own previously generated `regression_input` artifact when
  `rebuild_input = false`. This checks the configured path and contract only;
  freshness is not established.
- The optional `disruption_indicator_file_path` remains unset in the checked-in
  sample, but if configured its sole declaration is `[data]`.
- Regression's fixed model outputs are individually configured, while
  model-dependent LaTeX tables and conditional figures are grouped under the
  configured `latex_dir` and `plots_dir`. No output root or legacy output
  filename is hardcoded in the regression package.

### Phase 3 — code migration report and Phase 4 gate

- `uv run ruff check .` passed.
- `uv run --no-project --with mypy mypy --config-file mypy.ini src/acib/` passed with no type errors across 79 source files; mypy emitted notices for unrelated unused config sections.
- `uv run pre-commit run --all-files` passed all applicable hooks, including Flake8, Black, mypy, isort, AST, TOML, and whitespace checks. The two plot modules intentionally select the Agg backend before importing pyplot; `.flake8` scopes their Flake8 E402 exemption, while Ruff passes without unused `noqa` directives.
- CLI `--help` smoke checks for `python -m acib.regression`, `scripts/regression_analysis.py`, and `scripts/build_regression_input.py` passed. Loading the configured regression settings resolved the mobility proxy, external article/impact paths, outcome groups/aliases, and all explicit outputs. The mobility proxy and external inputs exist; the regression output directory did not exist before implementation checks.
- Path resolution and configured-input existence were checked; output declarations and schemas are implemented, but the regression workflow and model fitting were not run, so end-to-end artifact validity and numerical equivalence remain unverified. Existing artifact presence/schema checks do not establish freshness; stale-artifact detection is not implemented.
- No regression tests were created, migrated, modified, or run during Phase 2, as required by the approval gate. Phase 4 coverage and verification are recorded below.
- The original repository remained read-only. No regression workflow or output write was run during Phase 2; production research outputs were not overwritten. No commit was created.

### Phase 4 — regression test migration and verification

- Migrated regression input aggregation, article/impact joins, professor article
  construction, ASJC label and field-share rules, model specifications and
  families, MLE diagnostics, model reporting, and plot behavior into
  `tests/regression/`. Added temporary-path integration tests for configured
  input building/reuse, LaTeX outputs, CLI/workflow flags, and regression
  artifacts. No tests for other stages were added or modified.
- Added canonical-configuration tests using a temporary TOML file: relative
  generated paths resolve from the target project root, external absolute paths
  remain absolute, and regression configuration loads without statistical-
  matching or mobility-impact sections. The regression input contract rejects
  duplicate researcher IDs and non-boolean inbreeding status values. Workflow
  reuse tests use missing upstream/external paths and verify regression can use
  its own configured input without executing mobility.
- Added coverage for the approved Model 9 plot fix: the fixture fits the four
  configured OLS plot outcomes plus an unrelated Negative Binomial `n_public`
  Model 9; writing to a temporary output directory produces
  `plots/model9_inbreeding_predictions.png` and the professor violin plot.
- Ran the original regression tests read-only with bytecode and pytest cache
  disabled using the target uv environment and `PYTHONPATH` set to the original
  source. `tests/test_regression.py`: **45 passed** (including parametrized
  cases). Original `tests/test_build_regression_input_script.py`,
  `tests/test_model9_inbreeding_prediction_plot.py`, and
  `tests/test_professor_violin.py`: **12 passed**. The source files were not
  edited.
- `uv run pytest tests/regression/` passed: **60 passed**, 3 statsmodels
  warnings (Poisson BIC deprecation and two ZINB numerical warnings).
  `uv run pytest tests/integration/` passed: **14 passed**, 5 pandas
  `FutureWarning`s from existing mobility/name-matching integration tests.
- A deterministic eight-row OLS fixture was fitted separately by the original
  and migrated implementations. Coefficients (`Intercept = -0.125`, inbreeding
  effect `0.75`, age effect `1.05`), R-squared (`0.9436893203883495`), AIC,
  BIC, observation count, and formula matched; the target assertion uses tight
  numeric tolerances. This is a representative model regression comparison,
  not a full-dataset comparison.
- `uv run ruff check tests/regression/ tests/integration/` passed.
  `uv run --no-project --with mypy mypy --config-file mypy.ini
  tests/regression/ tests/integration/` passed with no type errors in 8 test
  source files; mypy noted unrelated unused config sections. Pre-commit passed
  for all five changed regression test files, including Flake8, Black, mypy,
  and isort.
- Plot-specific legacy helper-script functions do not exist in the target
  package. Their input/configuration and file-output behaviors are exercised
  through the migrated plot classes and configured regression writer instead.
  No full research-dataset regression run was performed during Phase 4; the
  configured research outputs were not overwritten. No commit was created.

## Phase 2 — `statistical_matching` code migration

### Configuration, ownership, and implementation

- Added the independent `acib.statistical_matching` package and direct
  `load_statistical_matching_config` loader. The sole input resolves through
  `regression.outputs.regression_input`; selected ordered outcomes are configured
  under `[statistical_matching].outcomes`, and family/label metadata is derived
  only from `[regression.outcome_metadata]`. Matching introduces no local input
  path, aliases/types, or upstream auto-run.
- Declared nine CSV artifacts and `matching_formulas` individually, plus
  `latex_dir` and `plots_dir` collection roots. Added
  `scripts/statistical_matching_analysis.py`, the
  `statistical-matching-analysis` command, the module entry point, and the
  canonical-path plot-only utility.
- The CSV outputs are `balance_summary`, `covariate_balance`,
  `outcome_effects_long`, `adjusted_outcome_means`, `outcome_effects_wide`,
  `outcome_standard_errors_wide`, `method_diagnostics`, `propensity_scores`, and
  `propensity_scores_by_method`. The LaTeX and PNG filenames retain the source
  names within the configured collections.
- Input schema validation requires researcher identity and configured treatment,
  covariates, and outcomes while preserving additional producer columns. Missing
  regression artifacts identify the regression producer. Presence/schema checks
  do not establish freshness.
- Matching reads the producer artifact through regression's existing CSV/Parquet
  table reader and applies `RegressionInputOutputSchema` before configured-column
  checks. Each CSV table is validated against a matching-owned Pandera contract
  before its configured write; dynamic method and diagnostic columns are allowed.
- Centralized confirmed shared inbreeding, outcome, academic-age/year, and field/
  domain-share constants in `acib.constants`; regression and matching use those
  shared definitions directly.

### Behavior and migration gate

- Migrated current-source treatment parsing/preparation, PSM, ATT IPTW, CEM,
  entropy balancing, population-pooled-SD balance diagnostics, weighted outcome
  models and configured robust covariance. Preserved PP-only percentage scaling,
  two-decimal console/LaTeX p-values, relative/absolute LaTeX families, and the
  conditional absolute-outcome plot. `treated_value` remains inert as in source.
- The read-only source worktree had pre-existing changes to matching, outcome
  metadata, family reports, and absolute-outcome plotting when audited. Phase 2
  used that working-tree behavior as the approved reference; those source files
  and changes were not edited.
- Reuse includes regression's producer artifact contract and the lazy formula API
  helper only; matching does not depend on regression algorithms. The plotting
  helper consumes canonical configured outputs without duplicating input paths.
- No matching workflow/model fit against research data was run during Phase 2,
  and configured research outputs were not written. No tests were created,
  edited, migrated, or run during Phase 2. Phase 3 static checks were reported
  before the separate Phase 4 approval; Phase 4 verification is recorded below.

### Phase 4 — statistical-matching tests and verification

- Migrated the source statistical-matching suite, report, balance-diagnostic,
  workflow-output, and plot assertions into `tests/statistical_matching/` and
  `tests/integration/test_statistical_matching_workflow.py`. Added tests for
  canonical input/output path resolution, regression-owned metadata, absence of
  duplicate local paths/metadata, independent execution without mobility,
  producer and output contracts, strict balance threshold, zero-variance SMD,
  PSM tie/input ordering and control exhaustion, and EB's no-feasible-column
  case. No other stages' tests were changed.
- Ran the original selected source tests read-only with bytecode and pytest
  cache disabled through the target uv environment: the original
  `tests/test_statistical_matching.py`,
  `tests/test_matching_covariate_balance_plot.py`, and
  `tests/test_matching_propensity_overlap_plot.py` passed (**16 passed**).
  The selected matching-dependent-variable test from the original
  `tests/test_config.py` passed (**1 passed; 6 deselected**). The original
  repository was not edited.
- `uv run pytest tests/statistical_matching/` passed: **24 passed**, one
  `PerfectSeparationWarning` from the PSM no-control-group error-path case.
  `uv run pytest tests/integration/test_statistical_matching_workflow.py`
  passed: **7 passed**, no warnings. All integration output paths use temporary
  directories; no configured research outputs were written.
- A deterministic synthetic fixture was fitted using the original and target
  statistical-matching implementations. The source-derived PSM mwPR ATT was
  `-3.0`, robust standard error `0.2029223250`, and 95% CI
  `[-3.4002700049, -2.5997299951]`; the target test checks these values with
  justified tolerances. The fixture also verifies treated/control weight totals
  and zero maximum SMD across the four methods. This is representative coverage,
  not a full-data or all-estimator numerical comparison.
- `uv run ruff check tests/statistical_matching/
  tests/integration/test_statistical_matching_workflow.py` passed.
  `uv run --no-project --with mypy mypy --config-file mypy.ini
  tests/statistical_matching/ tests/integration/test_statistical_matching_workflow.py`
  passed with no type errors in five test source files; mypy emitted notices
  about unrelated unused configuration sections. Pre-commit passed for all
  five changed test files, including Flake8, Black, mypy, and isort.
- No full configured research-data matching workflow was run and no commit was
  created. Full-data scientific comparison and numerical comparison of every
  estimator/outcome remain outside this small-fixture verification.

## Final output-directory configuration refactor

The approved Phase 2 refactor changes every migrated stage to declare a single
`[stage.outputs].directory`. Fixed filenames remain owned by the producer's
stage constants and are exposed through output dataclass properties; consumers
reference those producer properties. Regression and statistical-matching
LaTeX/plot roots derive as `directory / "latex"` and `directory / "plots"`,
preserving dynamic filenames. Dependency properties are
`name_matching.outputs.resolved_researchers`,
`name_disambiguation.outputs.final_author_selection`,
`mobility.outputs.author_inbreeding_proxy`, and
`regression.outputs.regression_input`. Relative paths resolve from the project
root, absolute paths remain absolute, stages remain explicitly invoked, and
stale-artifact detection is not implemented. See `docs/ARCHITECTURE.md` for the
canonical specification.

## Approved dependency auto-execution

The project owner approved running a producer stage automatically when a
user-invoked stage is missing one of its configured generated inputs. The
dependency graph is explicit: disambiguation ensures matching; mobility ensures
matching and disambiguation; regression ensures mobility when it rebuilds its
input; statistical matching ensures regression's input artifact, building that
input without fitting the regression model suite. Name matching has no upstream
producer. Missing external datasets remain errors. Existing artifacts are
reused without freshness detection. To prevent accidental replacement,
auto-execution refuses partial producer output sets unless the producer's
explicit overwrite option is supplied. See `docs/ARCHITECTURE.md` and
`acib.pipeline.dependencies` for the canonical behavior.
