.PHONY: clean features lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = crm_validation
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
# run `conda activate $(PROJECT_NAME)` before running this command
requirements:
	conda env update --file environment.yml

## Save Environment
save-environment:
	conda env export > environment.yml

## Default commands
default: features models

## Make Features
features:
	$(PYTHON_INTERPRETER) -m src.features.build_features

## Run the model
models: models-train models-predict

## Run and train the model
models-train:
	$(PYTHON_INTERPRETER) -m src.models.train_model

## Run and train the model
models-predict:
	$(PYTHON_INTERPRETER) -m src.models.predict_model

## Run tests
test:
	pytest -q tests/

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Delete all the figures
clean-figures:
	rm reports/figures/*

## Lint using flake8
lint:
	flake8 src
