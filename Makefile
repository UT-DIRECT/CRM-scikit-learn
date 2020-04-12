.PHONY: clean data lint requirements

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

environment:
	conda create --name $(PROJECT_NAME) python=3

## Install Python Dependencies
# make a  `test_environment` command
# run `conda activate $(PROJECT_NAME)` before running this command
requirements:
	conda env update --file environment.yml

## Save Environment
save-environment:
	conda env export > environment.yml

## Make Dataset
data:
	# $(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed
	$(PYTHON_INTERPRETER) src/data/process_dataset.py

## Run the model
model:
	$(PYTHON_INTERPRETER) -m src.models.train_model

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src
