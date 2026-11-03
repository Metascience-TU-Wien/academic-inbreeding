# Containerized execution

Containerization is optional; `uv` is the standard installation approach (see
the [README](../README.md#installation)). The primary image is
`ghcr.io/metascience-tu-wien/academic-inbreeding`. Prebuilt images are published
by the repository's GitHub Actions workflow; public access depends on the
maintainer setting the GHCR package visibility to public. No research data or
pipeline config is included in the image.

Both images use `/app` as the project working directory. The normal mounts are:

| Host directory | Container path | Access |
| --- | --- | --- |
| `config/` | `/app/config` | Read-only |
| `data/` or a configured external data root | `/app/data` | Read/write |

The sample config uses paths below `data/`; they resolve beneath `/app` in the
container. Mount external datasets at the same structure or provide a config
whose paths resolve to mounted container paths. Output artifacts persist only
when the configured output directory is on a mounted writable path. Keep
configuration outside the image too, so the exact analysis settings can be
reviewed and versioned separately.

## Docker

### Build locally

From the repository root:

```bash
make build                 # academic-inbreeding:latest
make build TAG=0.1.0       # academic-inbreeding:0.1.0
```

`IMAGE` and `TAG` can select another local image name/tag. The Dockerfile
installs the locked environment with `uv sync --frozen --no-dev`, sets the OCI
source label, and has a noninteractive `uv run --no-sync` entry point.

### Pull and run a prebuilt image

Authenticate only if the package is private; public GHCR images can be pulled
without a login. The Makefile's default registry, owner, and image are GHCR and
ACIB's organization/repository:

```bash
make pull TAG=0.1.0
```

The pull target fetches
`ghcr.io/metascience-tu-wien/academic-inbreeding:0.1.0`. To run the pulled image,
explicitly set `IMAGE_REF`; existing Makefile commands otherwise continue to
use their locally built `academic-inbreeding:$(TAG)` image:

```bash
make matching TAG=0.1.0 IMAGE_REF=ghcr.io/metascience-tu-wien/academic-inbreeding:0.1.0
make disambiguation IMAGE_REF=ghcr.io/metascience-tu-wien/academic-inbreeding:0.1.0 ARGS=--overwrite
```

The five Docker stage targets invoke `match-author-names`,
`disambiguate-author-names`, `analyze-mobility`, `regression-analysis`, or
`statistical-matching-analysis`, each with
`--config /app/config/pipeline.toml`. They retain the config read-only and data
read/write mounts, and run noninteractively. `ARGS="..."` passes extra CLI
arguments. To use another registry namespace, set `REGISTRY`, `OWNER`, `IMAGE`,
and `TAG` for `make pull`; GHCR is the supported publication destination.

### Synthetic example

From the repository root, mount the project config and data directories:

```bash
docker run --rm \
  --volume "$PWD/config:/app/config:ro" \
  --volume "$PWD/data:/app/data" \
  ghcr.io/metascience-tu-wien/academic-inbreeding:0.1.0 \
  disambiguate-author-names --config /app/config/pipeline_synthetic.toml --overwrite
```

This runs matching if its generated artifact is absent, then disambiguation.
The synthetic fixtures do not provide the mobility GeoPackage; regression is
configured not to rebuild its input and therefore needs a pre-existing
`author_regression_input.csv`. This is not a complete five-stage synthetic
workflow. `--overwrite` applies to disambiguation outputs, not upstream matching
outputs.

### Shell, mounts, and outputs

`make shell` opens interactive Bash with the standard mounts. Set
`CONFIG_DIR=/host/config` and `DATA_DIR=/host/data` to mount alternate locations
at `/app/config` and `/app/data`. If using a direct `docker run`, supply those
same volume mounts and a registered ACIB CLI command. Do not rely on files
written to the container filesystem outside mounted paths: `--rm` removes the
container and its unmounted changes when it exits.

### Manual GHCR publishing

Official release publishing through GitHub Actions is preferred (see below).
Manual publishing requires GHCR authentication and an exact semantic-version
Git tag such as `v0.1.0` on `HEAD`:

```bash
uv sync
uv run pytest --import-mode=importlib -q
echo "$GHCR_TOKEN" | docker login ghcr.io -u GITHUB_USERNAME --password-stdin
make build TAG=0.1.0
make push TAG=0.1.0
```

Use a GitHub token with package-write permission. `make push` checks that the
versioned image is not already visible, pushes the normalized version tag (for
example, `0.1.0`), and for stable releases updates `0.1` and `latest`. A tag
containing a prerelease suffix (for example, `v0.1.0-rc.1`) pushes only that
version tag. It never creates a Git tag or pushes during `make build`. Do not
manually reuse a published version tag. The moving `0.1` and `latest` aliases
are intended to advance for later stable patch releases.

The manual target checks release-tag existence through Docker Buildx. A
registry/authentication/network failure can also make that lookup fail; ensure
you are logged into GHCR and can reach the registry before publishing. Docker
push itself will fail if credentials lack write permission.

## Apptainer

### Build from the local definition

```bash
make apptainer-build
```

This uses `Apptainer.def` and writes `academic-inbreeding.sif` by default;
override the path with `APPTAINER_IMAGE=/path/to/image.sif`. The definition
independently installs dependencies from `uv.lock` using the same Python 3.12
base and `uv` version as the Docker build. It retains a native Apptainer
runscript and test section used by existing HPC wrappers.

### Pull the prebuilt OCI image as a SIF

Apptainer can directly consume the GHCR Docker/OCI image. `make apptainer-pull`
uses `TAG` (default `latest`) and the configured `APPTAINER_IMAGE` destination:

```bash
make apptainer-pull TAG=0.1.0
# Equivalent direct command:
apptainer pull academic-inbreeding.sif \
  docker://ghcr.io/metascience-tu-wien/academic-inbreeding:0.1.0
```

The Make target refuses to overwrite an existing SIF. To refresh it, remove or
move that SIF explicitly, or set `APPTAINER_IMAGE` to a new output path before
pulling. Public images do not require login; a private package requires
Apptainer/ORAS-compatible registry credentials (`apptainer registry login`).

The locally built Apptainer image remains supported; no HPC wrapper or
`Apptainer.def` functionality is removed by the GHCR option. Existing
`ACIB_CONFIG_DIR` and `ACIB_APPTAINER_IMAGE` overrides remain in effect for HPC
scripts. The pulled GHCR image uses the Docker image's root filesystem, where
the same `/app`, `/opt/venv`, source, and scripts are present; use `apptainer
exec` with a CLI command for direct runs.

Example synthetic run:

```bash
apptainer exec --pwd /app \
  --bind "$PWD/config:/app/config:ro" \
  --bind "$PWD/data:/app/data" \
  academic-inbreeding.sif \
  disambiguate-author-names --config /app/config/pipeline_synthetic.toml --overwrite
```

`make apptainer-shell` opens an interactive shell with the standard binds. Stage
targets (`make apptainer-matching`, `apptainer-disambiguation`, `apptainer-mobility`,
`apptainer-regression`, and `apptainer-statistical-matching`) invoke the current
scripts under `scripts/hpc/` and pass optional arguments with `ARGS="..."`.

## HPC and Slurm

Apptainer is intended for systems where Docker is restricted. The wrappers can
be submitted to Slurm or invoked directly on a host with Apptainer. They require
the repository scripts/config and a shared data root containing `input/`. Set
the data root in the job environment or `~/.bashrc`:

```bash
export DATA_DIR_ACADEMIC_INBREEDING="/shared/path/academic_inbreeding/data"
```

`ACIB_DATA_DIR` overrides the data root directly. `ACIB_CONFIG_DIR` and
`ACIB_APPTAINER_IMAGE` override the defaults for config and SIF paths. The
wrappers mount config read-only and data read/write at `/app`, use `/app` as the
container working directory, and write Slurm logs under the repository's
`logs/`. `make apptainer-*` passes `CONFIG_DIR` and `APPTAINER_IMAGE` through to
the wrapper environment. Site-specific Slurm account, partition, time, and
memory directives in `scripts/hpc/*.sh` must be reviewed for the target cluster.

Submit all five stages in sequence:

```bash
make apptainer-submit
# or
sbatch --chdir="$PWD" scripts/hpc/submit_pipeline.sh
```

The submission wrapper uses `afterok` dependencies from matching through
statistical matching. Stage dependency checks also run within each CLI as a
fallback. The production config's `rebuild_input = false` requires the
regression input artifact to exist before regression; the batch sequence does
not change that configuration setting.

To use a prebuilt GHCR image in these scripts, pull it to a SIF and set
`ACIB_APPTAINER_IMAGE` to that SIF path. The Slurm scripts do not pull images
implicitly, avoiding network-dependent image conversion during scheduled jobs.

## Automated publishing and reproducibility

`.github/workflows/publish-container.yml` runs the test suite before building
or publishing. It triggers on `v*` tags and manual workflow dispatch, uses
Docker Buildx and `GITHUB_TOKEN`, and builds `linux/amd64` images with GitHub
Actions layer caching. Stable semantic-version tags (for example `v0.1.0`) are
published as `0.1.0`, `0.1`, and `latest`; prereleases publish only their
versioned tag. Manual workflow runs publish `dev-<short-sha>` only. Pull
requests do not publish. Existing version/development tags are checked to avoid
unintended replacement.

The workflow has `contents: read` and grants `packages: write` only to its
publishing job. The first publication may require a maintainer to grant the
repository's Actions workflow permission to create/link the GHCR package and
then set package visibility to public. Organization/repository policy settings
must be configured by a maintainer; this repository workflow does not change
them. Once associated, the OCI source label links the image to this source
repository.

For reproducibility, prefer a published immutable version tag rather than
`latest`, and record the image digest as well as the config revision and input
dataset versions. Inspect a digest with:

```bash
docker buildx imagetools inspect \
  ghcr.io/metascience-tu-wien/academic-inbreeding:0.1.0
```

The returned manifest digest identifies the exact image content more precisely
than a tag. Record it in analysis logs or cite it with the release tag. Keep
research data outside the image and persist generated artifacts through the
mounted data path.

## Troubleshooting

- **`denied` or authentication errors:** For private packages, authenticate to
  `ghcr.io` with a token authorized for packages. Manual publishing needs write
  permission; pulling needs read permission. Public-package access still depends
  on maintainer visibility settings.
- **Image not found:** Check the owner, repository, tag, and that the release
  workflow completed. `make pull` pulls `$(REMOTE_IMAGE):$(TAG)`; `make matching`
  defaults to the local image unless `IMAGE_REF` is set.
- **Make target cannot find a local image:** Run `make build` or specify the
  intended `IMAGE_REF`. `make pull` does not silently change the locally selected
  pipeline image.
- **Input/output path missing or not persisted:** Check that the host directory
  is mounted at the expected `/app` path and that the configuration points to
  the container path. Output paths must be on a writable mount.
- **SIF already exists:** `make apptainer-pull` refuses to replace it. Remove it
  intentionally or choose another `APPTAINER_IMAGE` path.
- **Slurm job rejects settings:** Review site-specific SBATCH directives and
  verify that the shared data directory, config, SIF, and `input/` are visible
  on compute nodes.
