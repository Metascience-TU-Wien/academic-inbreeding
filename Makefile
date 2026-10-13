IMAGE ?= academic-inbreeding
TAG ?= latest
CONFIG_DIR ?= $(CURDIR)/config
DATA_DIR ?= $(CURDIR)/data

DOCKER_RUN = docker run --rm \
	--volume "$(CONFIG_DIR):/app/config:ro" \
	--volume "$(DATA_DIR):/app/data"

.DEFAULT_GOAL := help
.PHONY: help build shell matching disambiguation mobility regression statistical-matching

help:
	@echo "Docker targets:"
	@echo "  make build                 Build the image"
	@echo "  make shell                 Open a shell in the container"
	@echo "  make matching              Run name matching (ARGS=... optional)"
	@echo "  make disambiguation        Run name disambiguation (ARGS=... optional)"
	@echo "  make mobility              Run mobility analysis (ARGS=... optional)"
	@echo "  make regression            Run regression analysis (ARGS=... optional)"
	@echo "  make statistical-matching  Run statistical matching (ARGS=... optional)"

build:
	docker build --tag $(IMAGE):$(TAG) .

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
