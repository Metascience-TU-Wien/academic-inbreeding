IMAGE ?= academic-inbreeding
TAG ?= latest
UTAG ?=
CONFIG_DIR ?= $(CURDIR)/config
DATA_DIR ?= $(CURDIR)/data
APPTAINER_IMAGE ?= $(CURDIR)/academic-inbreeding.sif

DOCKER_RUN = docker run --rm \
	--volume "$(CONFIG_DIR):/app/config:ro" \
	--volume "$(DATA_DIR):/app/data"

.DEFAULT_GOAL := help
.PHONY: help build shell matching disambiguation mobility regression statistical-matching \
	push \
	apptainer-build apptainer-shell apptainer-submit apptainer-matching \
	apptainer-disambiguation apptainer-mobility apptainer-regression \
	apptainer-statistical-matching

help:
	@echo "Docker targets:"
	@echo "  make build                 Build the image"
	@echo "  make push UTAG=<registry>  Push version and latest tags (requires HEAD on a Git tag)"
	@echo "  make shell                 Open a shell in the container"
	@echo "  make matching              Run name matching (ARGS=... optional)"
	@echo "  make disambiguation        Run name disambiguation (ARGS=... optional)"
	@echo "  make mobility              Run mobility analysis (ARGS=... optional)"
	@echo "  make regression            Run regression analysis (ARGS=... optional)"
	@echo "  make statistical-matching  Run statistical matching (ARGS=... optional)"
	@echo "  make apptainer-build       Build the Apptainer image"
	@echo "  make apptainer-submit      Submit all five Slurm stages"
	@echo "  make apptainer-shell       Open a shell in the Apptainer image"

build:
	docker build --tag $(IMAGE):$(TAG) .

push:
	@set -e; \
	if [ -z "$(UTAG)" ]; then echo "Set UTAG to the registry namespace (for example, your Docker Hub username)." >&2; exit 1; fi; \
	GIT_TAG=$$(git describe --tags --exact-match HEAD 2>/dev/null) || { echo "Not on a Git tag. Please create a Git tag before pushing." >&2; exit 1; }; \
	docker image inspect "$(IMAGE):$(TAG)" >/dev/null || { echo "Local image $(IMAGE):$(TAG) not found; run 'make build' first." >&2; exit 1; }; \
	echo "Tagging and pushing $(UTAG)/$(IMAGE):$$GIT_TAG..."; \
	docker tag "$(IMAGE):$(TAG)" "$(UTAG)/$(IMAGE):$$GIT_TAG"; \
	docker push "$(UTAG)/$(IMAGE):$$GIT_TAG"; \
	echo "Tagging and pushing $(UTAG)/$(IMAGE):latest..."; \
	docker tag "$(IMAGE):$(TAG)" "$(UTAG)/$(IMAGE):latest"; \
	docker push "$(UTAG)/$(IMAGE):latest"

shell:
	docker run --rm -it --entrypoint /bin/bash \
		--volume "$(CONFIG_DIR):/app/config:ro" \
		--volume "$(DATA_DIR):/app/data" \
		$(IMAGE):$(TAG)

matching:
	$(DOCKER_RUN) $(IMAGE):$(TAG) match-author-names --config /app/config/pipeline.toml $(ARGS)

disambiguation:
	$(DOCKER_RUN) $(IMAGE):$(TAG) disambiguate-author-names --config /app/config/pipeline.toml $(ARGS)

mobility:
	$(DOCKER_RUN) $(IMAGE):$(TAG) analyze-mobility --config /app/config/pipeline.toml $(ARGS)

regression:
	$(DOCKER_RUN) $(IMAGE):$(TAG) regression-analysis --config /app/config/pipeline.toml $(ARGS)

statistical-matching:
	$(DOCKER_RUN) $(IMAGE):$(TAG) statistical-matching-analysis --config /app/config/pipeline.toml $(ARGS)

apptainer-build:
	apptainer build $(APPTAINER_IMAGE) Apptainer.def

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
