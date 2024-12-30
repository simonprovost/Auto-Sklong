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
longitudinal data, also called _Panel data_ in certain fields. Today, [`Sklong`](https://simonprovost.github.io/scikit-longitudinal/) 
is focussed on the Longitudinal Machine Learning Classification task.
It offers tools and models for processing, analysing, 
and classify longitudinal data, with a user-friendly interface that 
integrates with the `Scikit-learn` ecosystem.

`Auto-Scikit-Longitudinal` (Auto-Sklong) is an Automated Machine Learning (AutoML) library, developed upon the
[`General Machine Learning Assistant (GAMA)`](https://openml-labs.github.io/gama/master/index.html#) framework, 
introduces a brand-new [`search space`](https://simonprovost.github.io/Auto-Sklong/search_space/) leveraging both
[`Sklong`](https://simonprovost.github.io/scikit-longitudinal/) and [`Scikit-learn`](https://scikit-learn.org/stable/) 
models to tackle the Longitudinal machine learning classification tasks.

`Auto-Sklong` comes with various search method to explore the [`search space`](https://simonprovost.github.io/Auto-Sklong/search_space/) introduced. `Bayesian Optimisation`


## <a id="installation"></a>🛠️ Installation

To install `Auto-Sklong`, take these two easy steps:

1. ✅ **Install the latest version of `Auto-Sklong`**:

```shell
pip install Auto-Sklong
```
You could also install different versions of the library by specifying the version number, 
e.g. `pip install Auto-Sklong==0.0.1`. 
Refer to [Release Notes](https://github.com/simonprovost/auto-sklong/releases)

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
<summary><strong>🫵 Project Setup: Using `UV`</strong></summary>

Imagine you are managing your project with **UV**, a powerful and flexible project management tool. Below is an example configuration for integrating **UV** in your `pyproject.toml` file.

To ensure smooth operation and avoid dependency conflicts, you can override specific dependencies like `Scikit-Learn`. Add the following configuration to your `pyproject.toml`:

```toml
[tool.uv]
package = true
override-dependencies = [
    "scikit-learn ; sys_platform == 'never'",
]
```

This setup ensures that UV will manage your project’s dependencies efficiently, while avoiding conflicts with Scikit-Learn.

</details>

<details>
<summary><strong>🫵 Project Setup: Using `PDM`</strong></summary>

Imagine you have a project being managed by `PDM`, or any other package manager. The example below demonstrates `PDM`. 
Nevertheless, the process is similar for `Poetry`.

Therefore, to prevent dependency conflicts, you can exclude `Scikit-Learn` by adding the provided configuration 
to your `pyproject.toml` file.

```toml
[tool.pdm.resolution]
excludes = ["scikit-learn"]
```

*This exclusion ensures Scikit-Lexicographical-Trees (used as `Scikit-learn`) is used seamlessly within your project.*
</details>

### 🐾 Installing `Auto-Sklong` on Apple Silicon Macs

Apple Silicon-based Macs require running under an `x86_64` architecture to ensure proper installation and functioning 
of `Auto-Sklong`. Below is a step-by-step guide using **UV** as the state-of-the-art package manager to address this:

Note that this is mainly due to Deep-Forest not being compatible with Apple Silicon.

---

#### Step 1: Start a Terminal Session Under `x86_64` Architecture
Run the following command to instantiate a `zsh` shell under the `x86_64` architecture:

```bash
arch -x86_64 zsh
```
Further reading: [Switching Terminal Between x86_64 and ARM64](https://vineethbharadwaj.medium.com/m1-mac-switching-terminal-between-x86-64-and-arm64-e45f324184d9).

---

#### Step 2: Install an `x86_64` Python Version
Install an `x86_64` compatible Python version using **UV**:

```bash
uv python install cpython-3.9.21-macos-x86_64-none # Can be using another 3.9.x version. Run `uv python list` to see available versions.
```
*Reference: [UV Python Install Documentation](https://docs.astral.sh/uv/reference/cli/#uv-python-install).*

---

#### Step 3: Set Up an Isolated Environment
To avoid conflicts, set up a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

---

#### Step 4: Pin the Installed Python Version
Ensure that the installed `x86_64` Python version is the one used by UV:

```bash
uv python pin cpython-3.9.21-macos-x86_64-none
```
*Reference: [UV Python Pin Documentation](https://docs.astral.sh/uv/reference/cli/#uv-python-pin).*

---

#### Step 5: Install `Auto-Sklong`
Finally, install `Auto-Sklong` under the correct architecture:

```bash
uv pip install auto-sklong
# Alternatively
uv add auto-sklong
```

---

#### Post-Installation
Once installed, follow the [UV documentation](https://docs.astral.sh/uv/) or use Python as usual.

If you prefer a package manager other than UV, you can use it as long as you're operating under the `x86_64` architecture.

### 💻 Developer Notes

For developers looking to contribute, please refer to the `Contributing` section of `GAMA` [here](https://openml-labs.github.io/gama/master/contributing/index.html)
and `Scikit-Longitudinal` [here](https://simonprovost.github.io/scikit-longitudinal/contribution/).

## 🛠️ Supported Operating Systems

`Auto-Sklong` is compatible with the following operating systems:

- MacOS  _(Careful, you may need to force your settings to be under intel x86_64 and not apple silicon if you hold an M-based chip)_
- Linux 🐧
- On Windows 🪟, you are recommended to run the library within a Docker container under a Linux distribution.

!!! warning
    We haven't tested it on Windows without Docker.

## <a id="how-to-use"></a></a>🚀 Getting Started

To perform AutoML on your longitudinal analysis with `Auto-Sklong`, use the following two-easy-steps.

- First, load and prepare  your dataset using the `LongitudinalDataset` class of 
[`Sklong`](https://simonprovost.github.io/scikit-longitudinal/).

- Second, use the `GamaLongitudinalClassifier` class of `Auto-Sklong`. 
Following instantiating it set up its `hyperparameters` or let default, you can apply the popular 
_fit_, _predict_, _prodict_proba_, methods in the same way that `Scikit-learn` 
does, as shown in the example below. It will then automatically search for the best model and hyperparameters for your dataset.

_Refer to the documentation for more information on the `GamaLongitudinalClassifier` class._

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
dataset.setup_features_group(input_data="elsa")

# Instantiate the AutoML system
automl = GamaLongitudinalClassifier(
    features_group=dataset.features_group(),
    non_longitudinal_features=dataset.non_longitudinal_features(),
    feature_list_names=dataset.data.columns.tolist(),
)

# Run the AutoML system to find the best model and hyperparameters
model.fit(dataset.X_train, dataset.y_train)

# Predictions and prediction probabilities
label_predictions = automl.predict(X_test)
probability_predictions = automl.predict_proba(X_test)

# Classification report
print(classification_report(y_test, label_predictions))

# Export a reproducible script of the champion model
automl.export_script() 
```

1. Define the `features_group` manually or use a pre-set from the `LongitudinalDataset` class. If your dataset comes from the ELSA database or similar, you can use the pre-sets like `.setup_features_group('elsa')`. These pre-sets simplify the process of assigning temporal dependencies. Learn more [here](https://simonprovost.github.io/scikit-longitudinal/API/data_preparation/longitudinal_dataset/).
2. Define the `non_longitudinal_features` manually or use a pre-set from the `LongitudinalDataset` class. For example, if using the ELSA database, setting up features using `.setup_features_group('elsa')` automatically determines the non-longitudinal features. See details [here](https://simonprovost.github.io/scikit-longitudinal/API/data_preparation/longitudinal_dataset/).
3. Customise the `GamaLongitudinalClassifier` with various hyperparameters or use default settings. This flexibility allows you to tailor the search process to your specific longitudinal dataset. For a full list of configurable options, refer to the [API documentation](https://simonprovost.github.io/Auto-Sklong/API).
4. Use the `export_script` method to generate a Python script of the best model found during the AutoML search. This script enables you to reproduce the champion model independently of the `Auto-Sklong` system. More information can be found in the [API documentation](https://simonprovost.github.io/Auto-Sklong/API).

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
