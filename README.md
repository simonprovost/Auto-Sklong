<!--suppress HtmlDeprecatedAttribute -->
<div align="center">
   <p align="center">
   <h1 align="center">
      <br>
      <a href="https://i.imgur.com/Qu8fIfA.png">
         <img src="https://i.imgur.com/Qu8fIfA.png" alt="Auto-Sklong" width="200">
      </a>
      <br>
      Auto-Sklong
      <br>
   </h1>
   <h4 align="center">
      A specialised Python library for Automated Machine Learning (AutoML) 
      of Longitudinal machine learning classification tasks built upon 
      <a href="https://github.com/openml-labs/gama">GAMA</a>
   </h4>
</div>

<div align="center">

<!-- All badges in a row -->
<a href="https://github.com/openml-labs/gama">
   <img src="https://img.shields.io/badge/Fork-GAMA-green?labelColor=Purple&style=for-the-badge"
        alt="Fork GAMA" />
</a>
<a href="https://pytest.org/">
   <img alt="pytest" src="https://img.shields.io/badge/pytest-passing-green?style=for-the-badge&logo=pytest">
</a>
<a href="https://codecov.io/gh/Scikit-Longitudinal/Scikit-Longitudinal">
   <img alt="Codecov" src="https://img.shields.io/badge/coverage-76%25-brightgreen.svg?style=for-the-badge&logo=appveyor">
</a>
<a href="https://www.pylint.org/">
   <img alt="pylint" src="https://img.shields.io/badge/pylint-checked-blue?style=for-the-badge&logo=python">
</a>
<a href="https://pre-commit.com/">
   <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-checked-blue?style=for-the-badge&logo=python">
</a>
<a href="https://github.com/psf/black">
   <img alt="black" src="https://img.shields.io/badge/black-formatted-black?style=for-the-badge&logo=python">
</a>
<a href="https://github.com/astral-sh/ruff">
   <img alt="Ruff" src="https://img.shields.io/badge/Linter-Ruff-brightgreen?style=for-the-badge">
</a>
<a href="https://github.com/astral-sh/uv">
   <img alt="UV Managed" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json">
</a>

[simonprovostdev.vercel.app](https://simonprovostdev.vercel.app/)

</div>

# 📰 Latest News 

- **Bye Bye PDM!**: We are now leveraging 
  [UV](https://docs.astral.sh/uv/) from **Astral** (alongside Ruff)!

- **Documentation**: For a deep dive into `Auto-Sklong`, check out our 
  [official docs](https://simonprovost.github.io/Auto-Sklong/).

- **PyPi**: The library's latest version is published on [PyPi here](https://pypi.org/project/Auto-Sklong/).


## <a id="about-the-project"></a>💡 About The Project

`Auto-Scikit-Longitudinal`, also called `Auto-Sklong` is an automated machine learning (AutoML) library designed to analyse
longitudinal data (Classification tasks focussed as of today) using various search methods. Namely,
`Bayesian Optimisation` via [SMAC3](https://github.com/automl/SMAC3), `Asynchronous Successive Halving`, 
`Evolutionary Algorithms`, and `Random Search` 
via [the General Automated Machine Learning Assistant (GAMA)](https://github.com/openml-labs/gama).

`Auto-Sklong` built upon `GAMA`, offers a brand-new search space to tackle the Longitudinal Machine Learning classification problems,
with a user-friendly interface, similar to the popular `Scikit` paradigm.

Please for further information, visit the [official documentation](https://simonprovost.github.io/scikit-longitudinal/).

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

### 🚀 **What's New Compared to GAMA?**

We enhanced [@PGijsbers'](https://github.com/PGijsbers) open-source `GAMA` initiative by introducing a brand-new search space designed specifically for tackling longitudinal classification problems. This search space is powered by our custom library, [`Scikit-Longitudinal` (Sklong)](https://github.com/simonprovost/scikit-longitudinal), enabling Combined Algorithm Selection and Hyperparameter Optimization (CASH Optimization).

Unlike `GAMA` or other existing AutoML libraries, `Auto-Sklong` offers out-of-the-box support for 
longitudinal classification tasks—a capability not previously available. 

#### Search Space Viz.:
To better understand our proposed search space, refer to the visualisation below (read from left to right, each step being one new component to a final pipeline candidate configuration):

[![Search Space Visualization](https://i.imgur.com/advUOnU.png)](https://i.imgur.com/advUOnU.png)

While `GAMA` offers some configurability for search spaces, we improved its functionality to better suit our needs. You can find the details of our contributions in the following pull requests:
- [ConfigSpace Technology Integration for Enhanced GAMA Configuration and Management 🥇](https://github.com/openml-labs/gama/pull/210)
- [Search Methods Enhancements to Avoid Duplicate Evaluated Pipelines 🥈](https://github.com/openml-labs/gama/pull/211)
- [SMAC3 Bayesian Optimisation Integration 🆕](https://github.com/openml-labs/gama/pull/212)

### 💻 Developer Notes

For developers looking to contribute, please refer to the `Contributing` section of `GAMA` [here](https://openml-labs.github.io/gama/master/contributing/index.html)
and `Scikit-Longitudinal` [here](https://simonprovost.github.io/scikit-longitudinal/contribution/).

## <a id="Supported-Operating-Systems"></a>🛠️ Supported Operating Systems

`Auto-Sklong` is compatible with the following operating systems:

- MacOS  _(Careful, you may need to force your settings to be under intel x86_64 and not apple silicon if you hold an M-based chip)_
- Linux 🐧
- On Windows 🪟, you are recommended to run the library within a Docker container under a Linux distribution.

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

## <a id="citation"></a>📝 How to Cite?

`Auto-Sklong` paper has been accepted to the International Conference on Bioinformatics and Biomedicine (BIBM) 2024 edition. Awaiting for the proceeding to be released.
In the meantime, for the repository, utilise the button top right corner of the
repository "How to cite?", or open the following citation file: [CITATION.cff](./CITATION.cff).

## <a id="license"></a>🔐 License

[MIT License](./LICENSE)