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
default: crmp koval

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

real-data-features:
	$(PYTHON) src.features.build_real_data_features

north-sea-data-features:
	$(PYTHON) src.features.build.north_sea_data_features

## Run all the models
simulate: simulate-crmp simulate-koval

## Run all the plots
plots: crmp-plots koval-plots crmp-sensitivity-analysis-plots

## Entire CRMP workflow
crmp: simulate-crmp crmp-plots

## Run the CRMP models
simulate-crmp: train-crmp predict-crmp

## Run and train the models
train-crmp:
	$(PYTHON) src.simulations.crmp.train

## Run the models to make predictions
predict-crmp:
	$(PYTHON) src.simulations.crmp.predict

## Make the CRMP plots
crmp-plots:
	$(PYTHON) src.visualization.crmp

## Run Sensitivity Analysis on CRMP
crmp-sensitivity-analysis:
	$(PYTHON) src.simulations.crmp.sensitivity_analysis

## Plot Sensitivity Analysis results for CRMP
crmp-sensitivity-analysis-plots:
	$(PYTHON) src.visualization.crmp_sensitivity_analysis

## Run script to find the characteristic parameter values
crmp-characteristic-param:
	$(PYTHON) src.simulations.crmp.characteristic_param

crmp-characteristic-param-plots:
	$(PYTHON) src.visualization.characteristic_param

crmp-north-sea-data:
	$(PYTHON) src.simulations.crmp.north_sea_data

crmpbhp-north-sea-data:
	$(PYTHON) src.simulations.crmpbhp.north_sea_data

north-sea-plots:
	$(PYTHON) src.visualization.north_sea_data

## Entire Koval workflow
koval: simulate-koval koval-plots

## Run the Koval models
simulate-koval: train-koval predict-koval

train-koval:
	$(PYTHON) src.simulations.koval.train

predict-koval:
	$(PYTHON) src.simulations.koval.predict

## Make the Koval plots
koval-plots:
	$(PYTHON) src.visualization.koval

## Run MBBaggingRegressor
simulate-mbbagging:
	$(PYTHON) src.simulations.mbbaggingregressor.simulate

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
