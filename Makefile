#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = building-image-triplet-model
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

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
