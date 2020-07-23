.PHONY: clean features lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = crm_validation
PYTHON_INTERPRETER = python3
PYTHON = $(PYTHON_INTERPRETER) -m

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Default commands
default: simulate-crmp plots

## Install Python Dependencies
# run `conda activate $(PROJECT_NAME)` before running this command
requirements:
	conda env create -f environment.yml

## Save Environment
save-environment:
	conda env export > environment.yml

## Make Features
features:
	$(PYTHON) src.features.build_features

## Run the models
simulate-crmp: train-crmp predict-crmp

## Run and train the models
train-crmp:
	$(PYTHON) src.simulations.train_crmp

## Run the models to make predictions
predict-crmp:
	$(PYTHON) src.simulations.predict_crmp

## Make all the plots
crmp-plots:
	$(PYTHON) src.visualization.visualize_crmp

## Run tests
test:
	pytest -q tests/

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Delete all the figures
# TODO: There has to be a better way to do this
clean-figures:
	find ./reports/figures/ ! -type d -delete && git checkout reports/figures/.gitkeep

## Lint using flake8
lint:
	flake8 src
