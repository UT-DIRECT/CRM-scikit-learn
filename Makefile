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
default: features models plots

## Install Python Dependencies
# run `conda activate $(PROJECT_NAME)` before running this command
requirements:
	conda env update --file environment.yml

## Save Environment
save-environment:
	conda env export > environment.yml

## Make Features
features:
	$(PYTHON) src.features.build_features

## Run the models
models: train predict

production-rate-models: train-production-rate predict-production-rate

injection-rate-models: train-injection-rate predict-injection-rate

## Run and train the models
train: train-production-rate train-injection-rate

train-production-rate:
	$(PYTHON) src.models.train_production_rate_models

train-injection-rate:
	$(PYTHON) src.models.train_injection_rate_models

## Run the models to make predictions
predict: predict-production-rate predict-injection-rate

predict-production-rate:
	$(PYTHON) src.models.predict_production_rate_models

predict-injection-rate:
	$(PYTHON) src.models.predict_injection_rate_models

## Make all the plots
plots: production-rate-plots injection-rate-plots

production-rate-plots:
	$(PYTHON) src.visualization.visualize_production_rate

injection-rate-plots:
	$(PYTHON) src.visualization.visualize_injection_rate

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

## Delete all the pickled models
clean-models:
	rm models/*

## Lint using flake8
lint:
	flake8 src
