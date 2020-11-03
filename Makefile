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

## Rum all the models
simulate: simulate-crmp simulate-koval

## Run all the plots
plots: crmp-plots koval-plots

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

## Run Sensitivity Analysis on CRMP
crmp-sensitivity-analysis:
	$(PYTHON) src.simulations.crmp_sensitivity_analysis

## Plot Sensitivity Analysis results for CRMP
crmp-sensitivity-analysis-plot:
	$(PYTHON) src.visualization.crmp_sensitivity_analysis

## Make the CRMP plots
crmp-plots:
	$(PYTHON) src.visualization.crmp

## Entire ICRMP workflow
icrmp: simulate-icrmp icrmp-plots

## Run the ICRMP models
simulate-icrmp: train-icrmp predict-icrmp

## Run and train ICRMP
train-icrmp:
	$(PYTHON) src.simulations.train_icrmp

## Run ICRMP to make predictions
predict-icrmp:
	$(PYTHON) src.simulations.predict_icrmp

## Make the ICRMP plots
icrmp-plots:
	$(PYTHON) src.visualization.icrmp

## Run Sensitivity Analysis on CRMP
icrmp-sensitivity-analysis:
	$(PYTHON) src.simulations.icrmp_sensitivity_analysis

## Plot Sensitivity Analysis results for CRMP
icrmp-sensitivity-analysis-plot:
	$(PYTHON) src.visualization.icrmp_sensitivity_analysis

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
	$(PYTHON) src.visualization.koval

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
