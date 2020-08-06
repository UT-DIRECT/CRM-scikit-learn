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
default: crmp crmt koval

## Install Python Dependencies
# run `conda activate $(PROJECT_NAME)` before running this command
requirements:
	conda env create -f environment.yml

## Save Environment
save-environment:
	conda env export > environment.yml

## Make Features

features: crmp-features wfsim-features

crmp-features:
	$(PYTHON) src.features.build_crmp_features

wfsim-features:
	$(PYTHON) src.features.build_wfsim_features

## Rum all the models
simulate: simulate-crmp simulate-crmt simulate-koval

## Run all the plots
plots: crmp-plots crmt-plots koval-plots

## Entire CRMP workflow
crmp: simulate-crmp crmp-plots

## Run the CRMP models
simulate-crmp: train-crmp predict-crmp

## Run and train the models
train-crmp:
	$(PYTHON) src.simulations.train_crmp

## Run the models to make predictions
predict-crmp:
	$(PYTHON) src.simulations.predict_crmp

## Make the CRMP plots
crmp-plots:
	$(PYTHON) src.visualization.visualize_crmp

## Entire CRMT workflow
crmt: simulate-crmt crmt-plots

## Run the CRMT models
simulate-crmt: train-crmt predict-crmt

train-crmt:
	$(PYTHON) src.simulations.train_crmt

predict-crmt:
	$(PYTHON) src.simulations.predict_crmt

## Make the CRMT plots
crmt-plots:
	$(PYTHON) src.visualization.visualize_crmt

## Entire Koval workflow
koval: simulate-koval koval-plots

## Run the Koval models
simulate-koval: train-koval predict-koval

train-koval:
	$(PYTHON) src.simulations.train_koval

predict-koval:
	$(PYTHON) src.simulations.predict_koval

## Make the Koval plots
koval-plots:
	$(PYTHON) src.visualization.visualize_koval

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
