---
hide:
  - navigation
---

# 💡 About The Project
# 💡 About The Project

Longitudinal datasets contain information about the same cohort of individuals (instances) over time, 
with the same set of features (variables) repeatedly measured across different time points 
(also called `waves`) [1,2].

[`Scikit-Longitudinal`, also called `Sklong`](https://simonprovost.github.io/scikit-longitudinal/)is a machine learning library designed to analyse
longitudinal data, also called _Panel data_ in certain fields. Today, [`Sklong`](https://simonprovost.github.io/scikit-longitudinal/) is focussed on the Longitudinal Machine Learning Classification task.
It offers tools and models for processing, analysing, 
and classify longitudinal data, with a user-friendly interface that 
integrates with the `Scikit-learn` ecosystem.

`Auto-Scikit-Longitudinal` (Auto-Sklong) is an Automated Machine Learning (AutoML) library, developed upon the
[`General Machine Learning Assistant (GAMA)`](https://openml-labs.github.io/gama/master/index.html#) framework, introduces a brand-new [`search space`](https://simonprovost.github.io/Auto-Sklong/search_space/) leveraging both
[`Sklong`](https://simonprovost.github.io/scikit-longitudinal/) and [`Scikit-learn`](https://scikit-learn.org/stable/) models to tackle the Longitudinal machine learning classification tasks.

`Auto-Sklong` comes with various search method to explore the [`search space`](https://simonprovost.github.io/Auto-Sklong/search_space/) introduced. `Bayesian Optimisation`
via [`SMAC3`](https://github.com/automl/SMAC3), `Random Search`, `Successive Halving`, and `Evolutionary Algorithms`, via [`GAMA`](https://openml-labs.github.io/gama/master/index.html#).

## 🛠️ Installation

1. ✅ **Install the latest version of `Auto-Sklong`**:

```shell
pip install Auto-Sklong
```
!!! info "Different Versions?"
    You can also install different versions of the library by specifying the version number, e.g., `pip install Auto-Sklong==0.0.1`. 
    Refer to the [Release Notes](https://github.com/simonprovost/scikit-longitudinal/releases).

2. 📦 **[MANDATORY] Update the required dependencies (Why? See [here](https://github.com/pdm-project/pdm/issues/1316#issuecomment-2106457708))**

`Auto-Sklong` incorporates via `Sklong` a modified version of `Scikit-Learn` called `Scikit-Lexicographical-Trees`, 
which can be found at [this Pypi link](https://pypi.org/project/scikit-lexicographical-trees/).

This revised version guarantees compatibility with the unique features of `Scikit-longitudinal`. 
Nevertheless, conflicts may occur with other dependencies in `Auto-Sklong` that also require `Scikit-Learn`. 
Follow these steps to prevent any issues when running your project.

<details>
<summary><strong>🫵 Simple Setup: Command Line Installation</strong></summary>

Say you want to try `Auto-Sklong` in a very simple environment. Such as without a proper `project.toml` file (`Poetry`, `PDM`, etc).
Run the following command:

```shell
pip uninstall scikit-learn scikit-lexicographical-trees && pip install scikit-lexicographical-trees
```
</details>

<details>
<summary><strong>🫵 Project Setup: Using `PDM` (or any other such as `Poetry`, etc.)</strong></summary>

Imagine you have a project being managed by `PDM`, or any other package manager. The example below demonstrates `PDM`. 
Nevertheless, the process is similar for `Poetry` and others. Consult their documentation for instructions on excluding a 
package.

Therefore, to prevent dependency conflicts, you can exclude `Scikit-Learn` by adding the provided configuration 
to your `pyproject.toml` file.

```toml
[tool.pdm.resolution]
excludes = ["scikit-learn"]
```

*This exclusion ensures Scikit-Lexicographical-Trees (used as `Scikit-learn`) is used seamlessly within your project.*
</details>

### 💻 Developer Notes

For developers looking to contribute, please refer to the `Contributing` section of `GAMA` [here](https://openml-labs.github.io/gama/master/contributing/index.html)
and `Scikit-Longitudinal` [here](https://simonprovost.github.io/scikit-longitudinal/contribution/).

## 🛠️ Supported Operating Systems

`Auto-Sklong` is compatible with the following operating systems:

- MacOS  
- Linux 🐧
- On Windows 🪟, you are recommended to run the library within a Docker container under a Linux distribution.

!!! warning
    We haven't tested it on Windows without Docker.

## 🚀 Getting Started

To perform an AutoML search for your longitudinal machine learning classification task using `Auto-Sklong`, start by employing the
`LongitudinalDataset` class to prepare your dataset (i.e, data itself, temporal vector, etc.). Next, instantiate
a `GamaLongitudinalClassifier` object, which will set up the necessary configuration to run a search on your data,
with the parameters you would have entered in the `GamaLongitudinalClassifier` constructor.

``` py
from sklearn.metrics import classification_report
from scikit_longitudinal.data_preparation import LongitudinalDataset
from gama.GamaLongitudinalClassifier import GamaLongitudinalClassifier

# Load your longitudinal dataset
dataset = LongitudinalDataset('./stroke.csv')
dataset.load_data_target_train_test_split(
  target_column="class_stroke_wave_4",
)

# Pre-set or manually set your temporal dependencies 
dataset.setup_features_group(input_data="elsa") # (1)

# Instantiate the AutoML system
automl = GamaLongitudinalClassifier(
    features_group=dataset.features_group(),
    non_longitudinal_features=dataset.non_longitudinal_features(), # (2)
    feature_list_names=dataset.data.columns,
    # (3)
)

# Run the AutoML system to find the best model and hyperparameters
model.fit(dataset.X_train, dataset.y_train)

# Predictions and prediction probabilities
label_predictions = automl.predict(X_test)
probability_predictions = automl.predict_proba(X_test)

print(classification_report(y_test, label_predictions))
automl.export_script()  # (4)
```

1. Define the features_group manually or use a pre-set from the LongitudinalDataset class. If the data was from the ELSA database, you could have used the pre-sets such as `.setup_features_group('elsa')`. Read further in [here](https://simonprovost.github.io/scikit-longitudinal/API/data_preparation/longitudinal_dataset/)
2. Define the non-longitudinal features manually or use a pre-set from the LongitudinalDataset class. If the data was from the ELSA database, you could have used the pre-sets such as `.setup_features_group('elsa')`, then the non-longitudinal features would have been automatically set. Read further in [here](https://simonprovost.github.io/scikit-longitudinal/API/data_preparation/longitudinal_dataset/).
3. The `GamaLongtudinalClassifier` comes with a variety of hyperparameters that can be set. Refer to the [API](https://simonprovost.github.io/Auto-Sklong/API) for more information.
4. The `export_script` method allows you to export the best model found by the AutoML system as a Python script. This script can be used to reproduce the model without the need for the AutoML system. Refer to the [API](https://simonprovost.github.io/Auto-Sklong/API) for more information.

!!! question "Wants to understand what's the `feature_groups`? How your temporal dependencies are set via `pre-set` or `manually`?"
    To understand how to set your temporal dependencies, please refer to the [`Temporal Dependency`](https://simonprovost.github.io/scikit-longitudinal/temporal_dependency/) tab of the documentation.

!!! question "Wants more control on `hyper-parameters`?"
    To see the full API reference, please refer to the [`API`](https://simonprovost.github.io/Auto-Sklong/API/) tab.

!!! question "Wants more information on the Search Space `Auto-Sklong` comes with?"
    To see the full Search Space, please refer to the [`Search Space`](https://simonprovost.github.io/Auto-Sklong/search_space/) tab.

!!! question "Wants more to grasp the idea?"
    To see more examples, please refer to the [`Examples`](https://simonprovost.github.io/Auto-Sklong/examples/) tab of the documentation.

# 📚 References

> [1] Kelloway, E.K. and Francis, L., 2012. Longitudinal research and data analysis. In Research methods in occupational health psychology (pp. 374-394). Routledge.

> [2] Ribeiro, C. and Freitas, A.A., 2019. A mini-survey of supervised machine learning approaches for coping with ageing-related longitudinal datasets. In 3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (ARIAL), held as part of IJCAI-2019 (num. of pages: 5).
