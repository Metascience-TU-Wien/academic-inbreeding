REGISTRY ?= ghcr.io
OWNER ?= metascience-tu-wien
IMAGE ?= academic-inbreeding
TAG ?= latest
REMOTE_IMAGE = $(REGISTRY)/$(OWNER)/$(IMAGE)
IMAGE_REF ?= $(IMAGE):$(TAG)
CONFIG_DIR ?= $(CURDIR)/config
DATA_DIR ?= $(CURDIR)/data
APPTAINER_IMAGE ?= $(CURDIR)/academic-inbreeding.sif

DOCKER_RUN = docker run --rm \
	--volume "$(CONFIG_DIR):/app/config:ro" \
	--volume "$(DATA_DIR):/app/data"

.DEFAULT_GOAL := help
.PHONY: help build pull push shell matching disambiguation mobility regression statistical-matching \
	apptainer-build apptainer-shell apptainer-submit apptainer-matching \
	apptainer-pull apptainer-disambiguation apptainer-mobility apptainer-regression \
	apptainer-statistical-matching

help:
	@echo "Docker targets:"
	@echo "  make build                 Build the image"
	@echo "  make pull [TAG=version]    Pull the configured GHCR image"
	@echo "  make push                  Publish a Git-tagged image to GHCR"
	@echo "  make shell                 Open a shell in the container"
	@echo "  make matching              Run name matching (ARGS=... optional)"
	@echo "  make disambiguation        Run name disambiguation (ARGS=... optional)"
	@echo "  make mobility              Run mobility analysis (ARGS=... optional)"
	@echo "  make regression            Run regression analysis (ARGS=... optional)"
	@echo "  make statistical-matching  Run statistical matching (ARGS=... optional)"
	@echo "  make apptainer-build       Build the Apptainer image from Apptainer.def"
	@echo "  make apptainer-pull        Pull a GHCR image as a SIF (refuses overwrite)"
	@echo "  make apptainer-submit      Submit all five Slurm stages"
	@echo "  make apptainer-shell       Open a shell in the Apptainer image"

build:
	docker build --tag $(IMAGE):$(TAG) .

pull:
	docker pull $(REMOTE_IMAGE):$(TAG)

push:
	@set -e; \
	GIT_TAG=$$(git describe --tags --exact-match HEAD 2>/dev/null) || { echo "Not on a Git tag. Please create a semantic version tag (for example, v0.1.0)." >&2; exit 1; }; \
	VERSION=$${GIT_TAG#v}; \
	printf '%s\n' "$$VERSION" | grep -Eq '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?$$' || { echo "Expected a v-prefixed semantic version tag (for example, v0.1.0)." >&2; exit 1; }; \
	docker image inspect "$(IMAGE):$(TAG)" >/dev/null || { echo "Local image $(IMAGE):$(TAG) not found; run 'make build' first." >&2; exit 1; }; \
	if docker buildx imagetools inspect "$(REMOTE_IMAGE):$$VERSION" >/dev/null 2>&1; then echo "Refusing to overwrite existing release tag $(REMOTE_IMAGE):$$VERSION." >&2; exit 1; fi; \
	docker tag "$(IMAGE):$(TAG)" "$(REMOTE_IMAGE):$$VERSION"; \
	echo "Pushing $(REMOTE_IMAGE):$$VERSION..."; \
	docker push "$(REMOTE_IMAGE):$$VERSION"; \
	case "$$VERSION" in *-*) echo "Prerelease tag: leaving latest and major.minor aliases unchanged." ;; \
	*) MAJOR_MINOR=$${VERSION%.*}; \
	   docker tag "$(IMAGE):$(TAG)" "$(REMOTE_IMAGE):$$MAJOR_MINOR"; \
	   docker push "$(REMOTE_IMAGE):$$MAJOR_MINOR"; \
	   docker tag "$(IMAGE):$(TAG)" "$(REMOTE_IMAGE):latest"; \
	   docker push "$(REMOTE_IMAGE):latest" ;; esac

shell:
	docker run --rm -it --entrypoint /bin/bash \
		--volume "$(CONFIG_DIR):/app/config:ro" \
		--volume "$(DATA_DIR):/app/data" \
		$(IMAGE_REF)

matching:
	$(DOCKER_RUN) $(IMAGE_REF) match-author-names --config /app/config/pipeline.toml $(ARGS)

disambiguation:
	$(DOCKER_RUN) $(IMAGE_REF) disambiguate-author-names --config /app/config/pipeline.toml $(ARGS)

mobility:
	$(DOCKER_RUN) $(IMAGE_REF) analyze-mobility --config /app/config/pipeline.toml $(ARGS)

regression:
	$(DOCKER_RUN) $(IMAGE_REF) regression-analysis --config /app/config/pipeline.toml $(ARGS)

statistical-matching:
	$(DOCKER_RUN) $(IMAGE_REF) statistical-matching-analysis --config /app/config/pipeline.toml $(ARGS)

apptainer-build:
	apptainer build $(APPTAINER_IMAGE) Apptainer.def

apptainer-pull:
	@if [ -e "$(APPTAINER_IMAGE)" ]; then \
		echo "Refusing to overwrite $(APPTAINER_IMAGE); remove it or choose another APPTAINER_IMAGE." >&2; \
		exit 1; \
	fi
	apptainer pull $(APPTAINER_IMAGE) docker://$(REMOTE_IMAGE):$(TAG)

apptainer-submit:
	mkdir -p logs
	ACIB_CONFIG_DIR="$(CONFIG_DIR)" \
	ACIB_APPTAINER_IMAGE="$(APPTAINER_IMAGE)" \
	sbatch --chdir="$(CURDIR)" scripts/hpc/submit_pipeline.sh

apptainer-shell:
	apptainer exec --pwd /app \
		--bind "$(CONFIG_DIR):/app/config:ro" \
		--bind "$(DATA_DIR):/app/data" \
		$(APPTAINER_IMAGE) /bin/bash

apptainer-matching:
	ACIB_CONFIG_DIR="$(CONFIG_DIR)" \
	ACIB_APPTAINER_IMAGE="$(APPTAINER_IMAGE)" scripts/hpc/match_author_names.sh $(ARGS)

apptainer-disambiguation:
	ACIB_CONFIG_DIR="$(CONFIG_DIR)" \
	ACIB_APPTAINER_IMAGE="$(APPTAINER_IMAGE)" scripts/hpc/disambiguate_author_names.sh $(ARGS)

apptainer-mobility:
	ACIB_CONFIG_DIR="$(CONFIG_DIR)" \
	ACIB_APPTAINER_IMAGE="$(APPTAINER_IMAGE)" scripts/hpc/analyze_mobility.sh $(ARGS)

apptainer-regression:
	ACIB_CONFIG_DIR="$(CONFIG_DIR)" \
	ACIB_APPTAINER_IMAGE="$(APPTAINER_IMAGE)" scripts/hpc/regression_analysis.sh $(ARGS)

apptainer-statistical-matching:
	ACIB_CONFIG_DIR="$(CONFIG_DIR)" \
	ACIB_APPTAINER_IMAGE="$(APPTAINER_IMAGE)" scripts/hpc/statistical_matching_analysis.sh $(ARGS)
