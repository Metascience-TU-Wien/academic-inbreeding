# Containerized execution

Containerization is optional; the standard installation uses `uv` (see the
[README](../README.md#installation)). The Docker image and Apptainer image run
from `/app`. They keep configuration and data outside the image by mounting
host directories:

| Host directory | Container path | Access |
| --- | --- | --- |
| `config/` | `/app/config` | Read-only |
| `data/` (or another data directory) | `/app/data` | Read/write |

The checked-in configs use repository-relative paths such as `data/input` and
`data/output`. In the container, `/app` is the project root, so these resolve
inside the data mount. Results persist on the host as long as output paths point
under that mounted directory. Mount external research datasets in the same data
directory structure or edit a host-side config to use paths available inside
the container. A host path is not automatically visible inside the image unless
it is mounted.

## Docker

Build from the repository root:

```bash
make build
```

This builds `academic-inbreeding:latest` from `Dockerfile`. Set `IMAGE` and
`TAG` to change the image name or tag. The Dockerfile's entry point is
`uv run --no-sync`; the default command prints name-matching help.

Run the synthetic name-matching/disambiguation example:

```bash
docker run --rm \
  --volume "$PWD/config:/app/config:ro" \
  --volume "$PWD/data:/app/data" \
  academic-inbreeding:latest \
  disambiguate-author-names --config /app/config/pipeline_synthetic.toml --overwrite
```

This runs disambiguation, which produces matching output if it is missing.
The synthetic fixture has no mobility GeoPackage, and rebuilding the regression
input is not supported by that config (`rebuild_input = false`); regression
therefore needs an existing regression input artifact. The example does not
represent a full research-data run. `--overwrite` applies to disambiguation
outputs, not upstream matching outputs.

To run an individual production-config stage, `make` targets mount the
repository's `config/` and `data/` directories and pass the correct CLI entry
point:

```bash
make matching ARGS=--overwrite
make disambiguation ARGS=--overwrite
make mobility
make regression
make statistical-matching
```

Set `CONFIG_DIR=/path/to/config` and `DATA_DIR=/path/to/data` to use other host
locations. Their contents are visible in the container at `/app/config` and
`/app/data`; the config must refer to container-visible locations. For a direct
Docker invocation, use the same bind mounts and one of the registered commands
(`match-author-names`, `disambiguate-author-names`, `analyze-mobility`,
`regression-analysis`, `statistical-matching-analysis`) with
`--config /app/config/pipeline.toml`.

## Apptainer

Build a SIF from the repository root:

```bash
make apptainer-build
```

The default output is `academic-inbreeding.sif`; override it with
`APPTAINER_IMAGE=/path/to/image.sif`. The definition bootstraps from
`python:3.12-slim` and installs locked dependencies. The image runscript uses
`/app` as its working directory.

Run the synthetic example with host config and data mounted:

```bash
apptainer exec --pwd /app \
  --bind "$PWD/config:/app/config:ro" \
  --bind "$PWD/data:/app/data" \
  academic-inbreeding.sif \
  disambiguate-author-names --config /app/config/pipeline_synthetic.toml --overwrite
```

`make apptainer-shell` opens an interactive shell with the standard mounts.
Stage targets call the corresponding `scripts/hpc/*.sh` wrapper; for example:

```bash
make apptainer-mobility
make apptainer-regression
```

Those wrappers use the production `pipeline.toml`, bind config read-only and
data read/write, and set the container working directory to `/app`. They accept
additional stage CLI options through `ARGS`, for example
`make apptainer-disambiguation ARGS=--overwrite`.

## HPC and Slurm

The Apptainer wrappers include Slurm resource directives and can be invoked via
`sbatch` or run directly on a machine with Apptainer. They expect the repository
scripts and config directory to be accessible on the host, plus a shared data
directory containing `input/`. Set the data root in `~/.bashrc` (or the job
environment):

```bash
export DATA_DIR_ACADEMIC_INBREEDING="/shared/path/academic_inbreeding/data"
```

Alternatively, `ACIB_DATA_DIR` overrides that variable directly. Config and
image locations default to the repository's `config/` and
`academic-inbreeding.sif`; override them with `ACIB_CONFIG_DIR` and
`ACIB_APPTAINER_IMAGE`. `make apptainer-*` also passes `CONFIG_DIR` and
`APPTAINER_IMAGE` into those settings. The scripts check the configured paths
and require `input/` before proceeding. They create `logs/` under the repository
root, set numerical-library thread counts from `SLURM_CPUS_PER_TASK`, and bind
the data directory read/write at `/app/data`.

To submit the complete five-stage Slurm sequence from the repository root:

```bash
make apptainer-submit
# or
bash scripts/hpc/submit_pipeline.sh
# or
sbatch --chdir="$PWD" scripts/hpc/submit_pipeline.sh
```

The submitter queues name matching, disambiguation, mobility, regression, then
statistical matching with `afterok` dependencies. Stage-level dependency checks
remain a fallback for missing generated artifacts; they do not generate
external datasets. Slurm account, partition, memory, and time directives in
`scripts/hpc/*.sh` are site-specific and should be reviewed before submission.
