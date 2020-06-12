crm_validation
==============================

This repository is going to evaluate how well the CRM model predicts reservoir
production.


Project Organization
------------

The Project is organized along the lines of [Cookie Cutter Data
Science](https://drivendata.github.io/cookiecutter-data-science/). There are
some modifications to the naming of certain files in the project, though
I believe that these will be self explanatory and will not hamper the intuitive
structure oft he project.

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank"
href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter
data science project template</a>. #cookiecutterdatascience</small></p>

## Setting up environment

In order to set up the environment for development, run the following commands.

```
$ conda env create -f environment.yml
$ conda activate crm_validation
```

If you do make changes to the dependencies of the project, and you would like
to save them run `make save-environment`.*

*Important note: When exporting the environment to a file with `conda env
export` (and in this case `make save-enironment`), the resulting
`environment.yml` will have elements that only work on the host file. You must
see the results of the unit tests on Travis and remove everything after the
first "=" on that dependency's line.

## Running the models

1. Add data to the `data/raw/` directory and modify the `inputs.yml` file.
2. Modify `src/models/__init__.py` to read in the data appropriately. (Might
   need to make changes in `src/features/build_features.py`).
3. Run `make`.

## Unit Tests

Unit tests are in the `tests/` directory, and the structure of the `tests/`
directory mirrors the structure of the `src/` directory. Not everything
is tested.

In order to run unit tests, run `make tests`.
