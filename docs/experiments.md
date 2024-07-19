---
hide:
  - navigation
---

# 🔬 Experimentation for `Auto-Sklong` (for the paper's reproduction)
# 🔬 Experimentation for `Auto-Sklong` (for the paper's reproduction)

!!! warning "Be aware we have reworked the experiments engine"
    During the research for the `Auto-Sklong` paper, we used a different engine 
    for the experiments. This engine was more hard-coded, with some tweaks 
    that were not publishable. Therefore, we reworked the engine to be more user-friendly 
    and to provide a better experience for the user, as well as for further experimenting 
    with `Auto-Sklong`, especially against other AutoML libraries or baseline algorithms/neural 
    networks – for us too!

!!! info "A better engine: `AutoML Benchmark`"
    [@PGijsbers'](https://github.com/PGijsbers) et al. have created a flexible 
    experimentation-based system for AutoML libraries called `AutoML Benchmark`. 
    This system is much more flexible and user-friendly than the engine we provide with 
    `Auto-Sklong`. However, at the time, we did not have both access and time to explore 
    this benchmark system. In the future, we may, but in the meantime, what we deliver below 
    is for the paper's reproduction. In other words, use the 
    [`AMLB: an AutoML Benchmark`](https://arxiv.org/abs/2207.12560) for a better experience 
    with your AutoML experiments against others, including `Auto-Sklong`.

## 📚 Experiments

!!! info "The paper is submitted to a conference"
    The paper is submitted to a conference, and we are awaiting the reviews. 
    This means that users cannot yet read the paper, but we will provide the link as 
    soon as it is accepted. Stay tuned!

_The documentation below is intended to help users understand how we achieved the results presented in our paper. 
Nonetheless, we urge readers to read all the above information and warning blocks before proceeding._

### 📽️ Introduction

First and foremost, we provide an engine that is flexible, but not as flexible as the 
`AutoML Benchmark` system mentioned above. Our engine allows for a comparison of any system 
to `Auto-Sklong`. The engine utilises a Nested Cross-Validation (NCV) approach to evaluate systems and algorithms (as per our paper).

We provide the engine with the required number of folds to assess the system/algorithm, 
which automatically partitions the original data if necessary, in order to then acquire the train 
and test sets and run only on the required fold number (i.e, you have to run this for each outer fold of your NCV).

The engine therefore uses the reporter method to report metrics that can be compared to other 
systems or algorithms afterwards. The following sections will cover how to use the engine 
for an existing system or algorithm, as well as how to add a new system or algorithm.

### 🌍 How to access the experiments engine

We have provided all the experiments-based information in a single branch 
called `experiments`. Therefore, if you would like to explore the code's engine and how we 
conducted the experiments, please visit the `experiments` branch.

To do this, you can clone the repository and check out the `experiments` branch:

```shell
git clone <repository-url>
cd <repository-name>
git checkout experiments
```

### ✅ How to use the experiments engine

Navigate through the experiments folder and you should find 
`experiments_engine.py` and another folder `experiments_launchers`.

The engine allows for any launchers, such as those to run experiments on 
`Auto-Sklong`, `Auto-Sklearn`, or any other systems, to be executed with a shared reporting
method to compare them all together afterwards. This is because the engine generates CSV results
for each NCV's outer-fold results, therefore, the shared reporting manners are needed to have
a similar CSV format for all systems or algorithms.

#### 1️⃣ Use an available launcher

At present, launchers for `Auto-Sklong`, `Auto-Sklearn`, `Random Forest`, and `Lexico Random Forest` 
as per the paper's experiments are available. This means that you can use these launchers to run 
the experiments.

To do this, you may create bash scripts that will set up the available parameters, 
refer to those launchers accordingly to understand what is available, and then conclude the 
bash script with a Python run of the launcher of interest. For example, you can refer to the 
folder `24_hours`, which contains the bash scripts used to run the experiments for 24 hours 
in the paper.

All launchers have default hyperparameters for their respective systems or algorithms used in the paper,
but you can change them by providing different values in the bash script at your convenience.

#### 2️⃣ Add a new launcher

!!! tip "Duplicate an existing launcher"
    If you would like to add a new launcher, we recommend doing so by duplicating an existing 
    launcher and modifying it according to your needs.

To add a new launcher, you can create a new Python file in the `experiments_launchers` folder. 
The new launcher should have the following available methods:

- A reporter function that could follow the convention `def _reporter_<your_system_name>(system: <your_system_type>, X_test: pd.DataFrame) -> dict[str, Any]:`.

In a nutshell, this reporter function acquires the fitted system and the test set, and then 
returns a dictionary of metrics that you would like to report. The expected outputs should look like this:

```
dict[str, Any]: A dictionary containing the following keys:
    - "predictions": Predictions made by the system.
    - "probability_predictions": Probability predictions made by the system.
    - "best_pipeline": A dictionary with the names of the techniques used in the best pipeline for data preparation, preprocessing, and classification.
    - "metric_optimised": The name of the metric that was optimized during training.
```

!!! tip "Sometimes you may not be able to fill out some of the above needed information"
    For exemple in `best_pipeline`, sometimes baselines algorithms such as `random forest` do not create
    a `best_pipeline` as it is not a pipeline-based algorithm. Therefore, you can create the dictionary with the keys 
    but values set to the information you would like to report. For example, for `best_pipeline` you can set the value to 
    `"Random Forest"` to `classification`. See the `Random Forest` launcher for an example.


- A launcher class that contains the following methods:
    - `__init__`: to acquire the arguments provided by the bash script.
    - `validate_parameters`: to validate the parameters provided by the bash script.
    - `launch_experiment`: to use the generic engine, and provide (1) your data and the Nested Cross-Validation parameters, (2) your custom system and its hyperparameters, and your reporter method previously created.
    - `default_parameters`: to provide the default parameters for your system or algorithm, which are not provided by the bash scripts.

- A main method that will be used to run the launcher. This method should start by acquiring the necessary arguments 
from the bash scripts, in order to then execute the Launcher class, validate the parameters, and launch the experiment.

!!! danger "Be aware that path modifications are needed"
    To use the current bash scripts available in the `24_hours` folder, you will need to modify the paths in a few lines. 
    We recommend you open one bash script to see how the paths are set up, and then modify them accordingly. 
    These bash scripts are made to run in a SLURM architecture, but you can modify them to run on your local machine or
    any other architecture (cloud-based, etc.).

!!! danger "Data availability"
    The data used in the paper is not available in the repository. This does not mean that it is not available at all. 
    Contact us if you would like to have access to the data used in the paper. You will need to pass some checks per 
    the data source: https://www.elsa-project.ac.uk/

    Therefore, this also means that all paths to data in the bash scripts will need to be modified to your own path
    where the data is stored on your machine/cluster.

!!! tip "For further information"
    If you would like to have further information on how to use the engine, or how to add a new launcher,
    please walk through the experiments folder's python files. They are docstring-based documented.

### ✅ How to gather all results from each NCV's outer-fold

After running the experiments, you will have a CSV file for each NCV's outer-fold. To gather all the results
you can use the last python file, called `experiments_gather_results.py`. Fill out the main's `root_folders` list variable
with the root folders to each experiments done (parent of each NCV's outer-fold CSV files). Then run the script.

It will navigate through each NCv's outer-fold CSV files, gather the results, and create a CSV file with all the results
Sorting by default by the `Fold` column numbers.







