#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = building-image-triplet-model
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
GCP_RAW_DIR ?= data/gcp/raw
GCP_EXTRACT_DIR ?= $(GCP_RAW_DIR)/extracted
GCP_CONFIG ?= config.gcp.local.yaml
GCP_MANIFEST ?= data/downloads/raw_archives.txt

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run flake8 building_image_triplet_model
	uv run isort --check --diff building_image_triplet_model
	uv run black --check building_image_triplet_model

## Format source code with black
.PHONY: format
format:
	uv run isort building_image_triplet_model
	uv run black building_image_triplet_model


## Download raw archives using aria2 (uses gcp/scripts/download_archives.sh)
.PHONY: gcp-download
gcp-download:
	mkdir -p $(GCP_RAW_DIR)
	cp $(GCP_MANIFEST) $(GCP_RAW_DIR)/raw_archives.txt
	RAW_MOUNT_DIR=$(GCP_RAW_DIR) MANIFEST_PATH=$(abspath $(GCP_RAW_DIR)/raw_archives.txt) GCS_MANIFEST_URI= bash gcp/scripts/download_archives.sh


## Extract downloaded archives into $(GCP_EXTRACT_DIR)
.PHONY: gcp-extract
gcp-extract:
	RAW_MOUNT_DIR=$(GCP_RAW_DIR) EXTRACT_DIR=$(GCP_EXTRACT_DIR) bash gcp/scripts/extract_archives.sh


## Run preprocessing locally using config $(GCP_CONFIG)
.PHONY: gcp-preprocess
gcp-preprocess:
	uv run python -m building_image_triplet_model.dataset_processor --config $(GCP_CONFIG)





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv
	uv sync
	
	@echo ">>> uv environment created. Activate with:\nsource .venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
