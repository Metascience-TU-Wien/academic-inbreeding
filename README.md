# Academic Inbreeding (ACIB)

Academic Inbreeding (ACIB) is a Python package for analysing academic inbreeding and its association with research performance using bibliometric data.

ACIB provides a reproducible, configuration-driven, dependency-aware analysis pipeline comprising five main stages:

1. **Name matching:** Identifies candidate Scopus Author IDs for researchers listed in institutional staff records using affiliation information and exact, fuzzy, and rule-based name matching.
2. **Author disambiguation:** Resolves ambiguous matches and identifies Scopus Author IDs belonging to the same researcher using bibliographic, affiliation, and citation-based evidence.
3. **Mobility analysis:** Reconstructs researchers' institutional affiliation histories to classify academic inbreeding and derive mobility indicators.
4. **Regression analysis:** Examines associations between academic inbreeding and research performance using multivariable regression models, sensitivity analyses, and alternative estimators.
5. **Statistical matching:** Evaluates the robustness of these associations using propensity-score matching, inverse-probability weighting, coarsened exact matching, and entropy balancing.

## Installation

### Install with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available, then run the following command from the repository root:

```bash
uv sync
```

This creates the project environment and installs the dependencies specified in `pyproject.toml` and locked in `uv.lock`.

The project requires Python 3.12 or later, excluding Python 3.14.1.

### Containerized execution

Container use is optional; `uv` is the recommended standard installation method. Docker is convenient for local development, while Apptainer is suited to shared HPC systems where Docker may be restricted. Containers keep configuration and data on mounted host directories, so generated outputs persist outside the image. See [Containerization](docs/CONTAINERS.md) for build, execution, data-mounting, and HPC/SLURM instructions.

## Quick start

To test the pipeline without external research datasets, run the synthetic-data example from the repository root:

```bash
uv run disambiguate-author-names --config config/pipeline_synthetic.toml --overwrite
```

This executes the author disambiguation stage and automatically runs name matching if its required output artifacts are missing.

Results are written to `data/output/synthetic/`. The `--overwrite` option permits replacement of the disambiguation stage's six outputs; it does not request that upstream name-matching outputs be overwritten. This example covers name matching and author disambiguation only. The fixtures do not include the GeoPackage needed for mobility or the research-impact inputs needed for a complete downstream analysis. Regression is configured with `rebuild_input = false`, so it also requires a pre-existing regression input file.

To analyse your own data, configure the input and output paths in `config/pipeline.toml`. Research datasets are not included in the repository.

## Configuration

Pipeline settings are defined in a single TOML configuration file, typically `config/pipeline.toml`.

The configuration is organized into the following sections:

| Configuration section | Purpose |
| --- | --- |
| `[data]` | Paths to external researcher, author, article, affiliation, and citation-impact datasets. |
| `[name_matching]` | Name-matching settings, including organization filtering and worker count. |
| `[name_matching.outputs]` | Output directory for researcher-to-author candidate matches. |
| `[name_disambiguation]` | Author disambiguation settings, including pair-scoring thresholds, reporting, and worker count. |
| `[name_disambiguation.outputs]` | Output directory for pair decisions, author clusters, selected matches, and reports. |
| `[mobility]` | Settings for mobility analysis and academic inbreeding classification. |
| `[mobility.outputs]` | Output directory for mobility and academic inbreeding artifacts. |
| `[regression]` | Regression model settings, sample selection, and input reconstruction. |
| `[regression.dependent_variables]` | Regression estimator selected for each dependent variable. |
| `[regression.outcome_metadata.<outcome>]` | Outcome classification (relative or absolute) and display label. |
| `[regression.outputs]` | Output directory for regression artifacts, including `author_regression_input.csv`. |
| `[statistical_matching]` | Statistical matching settings, including outcomes, covariates, methods, and estimand. |
| `[statistical_matching.outputs]` | Output directory for matching results, reports, and plots. |

### Configuration conventions

Each stage maintains its own configuration settings. Values are not implicitly inherited from other stages.

Output directories are specified using the `directory` field within each `[stage.outputs]` section. Each stage defines the filenames of its generated artifacts, and downstream stages automatically use the corresponding upstream output paths.

Relative paths are resolved from the repository root, independent of the current working directory; absolute paths are retained. Environment-variable and home-directory markers are expanded.

For example, the regression stage generates `author_regression_input.csv` in the directory specified by `[regression.outputs].directory`. The statistical matching stage automatically uses this file as its input.

When `rebuild_input = false`, regression reuses the existing `author_regression_input.csv` instead of regenerating it. If the file does not exist, execution fails.

## Running the pipeline

Each pipeline stage can be executed independently using its corresponding command-line interface.

After configuring the required datasets and output directories in `config/pipeline.toml`, run the desired stage from the repository root.

| Stage | Command |
| --- | --- |
| Name matching | `uv run match-author-names --config config/pipeline.toml --overwrite` |
| Author disambiguation | `uv run disambiguate-author-names --config config/pipeline.toml --overwrite` |
| Mobility analysis | `uv run analyze-mobility --config config/pipeline.toml` |
| Regression analysis | `uv run regression-analysis --config config/pipeline.toml` |
| Statistical matching | `uv run statistical-matching-analysis --config config/pipeline.toml` |

### Dependency-aware execution

Users invoke stages explicitly. A command may run configured producers to create required missing generated artifacts; this is limited dependency handling, not a general workflow scheduler.

| Invoked stage | Generated upstream artifact(s) ensured when absent |
| --- | --- |
| Name matching | None. |
| Author disambiguation | `name_matching.outputs.resolved_researchers`. |
| Mobility | Resolved researchers and `name_disambiguation.outputs.final_author_selection`. |
| Regression | `mobility.outputs.author_inbreeding_proxy` when rebuilding the regression input. |
| Statistical matching | `regression.outputs.regression_input`; it builds this input if absent (ensuring mobility as needed), without fitting regression models. |

**Artifact handling:**

- **Missing generated artifacts:** Their configured producer may run. Missing external inputs remain errors. A producer is not automatically run to replace a partial set of existing outputs unless explicit overwrite is enabled for that producer.
- **Existing artifacts:** Reused without checking whether their source data or configuration has changed. Presence or schema validity does not establish freshness; stale-artifact detection is not implemented.
- **Overwriting outputs:** Only name matching and author disambiguation expose `--overwrite`; it applies to that invoked producer's outputs and does not propagate upstream. Mobility, regression, and statistical matching have no overwrite flag.
- **External datasets:** Must be configured and available before execution.

For regression analysis, setting `rebuild_input = false` explicitly disables reconstruction of `author_regression_input.csv`. In this case, the file must already exist in the configured regression output directory.

Statistical matching uses the same regression input artifact, ensuring consistency between the two analytical stages. For supported settings and defaults, see the [configuration reference](docs/configuration.md). Synthetic fixture contents and limitations are described in [`data/test/synthetic/README.md`](data/test/synthetic/README.md).
