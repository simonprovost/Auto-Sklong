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
   <h4 align="center">An Automated Machine Learning library for longitudinal classification built on GAMA and Scikit-longitudinal</h4>
</div>

<div align="center">

<!-- All badges in a row -->

<a href="https://pytest.org/">
   <img alt="pytest" src="https://img.shields.io/badge/pytest-passing-green?style=for-the-badge&logo=pytest">
</a>
<a href="https://www.pylint.org/">
   <img alt="pylint" src="https://img.shields.io/badge/pylint-checked-blue?style=for-the-badge&logo=python">
</a>
<a href="https://pre-commit.com/">
   <img alt="pre--commit" src="https://img.shields.io/badge/pre--commit-checked-blue?style=for-the-badge&logo=python">
</a>
<a href="https://github.com/psf/black">
   <img alt="black" src="https://img.shields.io/badge/black-formatted-black?style=for-the-badge&logo=python">
</a>

<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
<img src="https://img.shields.io/static/v1?label=RUFF&message=compliant&color=9C27B0&style=for-the-badge&logo=RUFF&logoColor=white" alt="RUFF compliant">
<img src="https://img.shields.io/static/v1?label=UV&message=compliant&color=2196F3&style=for-the-badge&logo=UV&logoColor=white" alt="UV compliant">
<a href="https://codecov.io/gh/simonprovost/Auto-Sklong">
   <img alt="Codecov" src="https://img.shields.io/badge/coverage-76%25-brightgreen.svg?style=for-the-badge&logo=appveyor">
</a>
<a href="https://github.com/openml-labs/gama">
   <img src="https://img.shields.io/badge/Fork-GAMA-green?labelColor=Purple&style=for-the-badge"
        alt="Fork GAMA" />
</a>
<img src="https://img.shields.io/static/v1?label=Python&message=3.9%2B%3C3.10&color=3776AB&style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+ < 3.10">

</div>

---

## <a id="about-the-project"></a>💡 About The Project

`Auto-Scikit-Longitudinal` (Auto-Sklong) is an Automated Machine Learning (AutoML) library, developed upon the
[`General Machine Learning Assistant (GAMA)`](https://openml-labs.github.io/gama/master/index.html#) framework, 
introducing a brand-new [`search space`](https://auto-sklong.readthedocs.io/en/latest/tutorials/search_space/) leveraging both
[`Scikit-Longitudinal`](https://scikit-longitudinal.readthedocs.io/latest/) and [`Scikit-learn`](https://scikit-learn.org/stable/) 
models to tackle the Longitudinal machine learning classification tasks.

For more scientific details, you can refer to our [paper](https://doi.org/10.1109/BIBM62325.2024.10821737) published by `IEEE` in the [IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2024 Edition](https://ieeexplore.ieee.org/xpl/conhome/10821710/proceeding).

`Auto-Sklong` comes with various search methods to explore the [`search space`](https://auto-sklong.readthedocs.io/en/latest/tutorials/search_space/) introduced, such as `Bayesian Optimisation`.  For more details, visit the [official documentation](https://auto-sklong.readthedocs.io/en/latest/).

---

## <a id="installation"></a>🛠️ Installation

> [!NOTE]
> Want to use `Jupyter Notebook`, `Marimo`, `Google Colab`, or `JupyterLab`?
> Head to the `Getting Started` section of the documentation for full instructions! 🎉

To install Auto-Sklong:

1. ✅ Install the latest version:
   ```bash
   pip install auto-sklong
   ```

   To install a specific version:
   ```bash
   pip install auto-sklong==0.0.1
   ```

> [!CAUTION]
> `Auto-Sklong` is currently compatible with Python versions `3.9` only. 
> Ensure you have this version installed before proceeding. 
> 
> This limitation stems from the `Deep Forest` dependency. 
> Follow updates on [this GitHub issue](https://github.com/LAMDA-NJU/Deep-Forest/issues/124).
> 
> If you encounter errors, explore the `installation` section in the `Getting Started` of the documentation.
> If issues persist, open a GitHub issue.

---

## <a id="getting-started"></a>🚀 Getting Started

Here's how to run AutoML on longitudinal data with Auto-Sklong:

```python
from sklearn.metrics import classification_report
from scikit_longitudinal.data_preparation import LongitudinalDataset
from gama.GamaLongitudinalClassifier import GamaLongitudinalClassifier

# Load your dataset (replace 'stroke.csv' with your actual dataset path)
dataset = LongitudinalDataset('./stroke.csv')

# Set up the target column and split the data (replace 'class_stroke_wave_4' with your target)
dataset.load_data_target_train_test_split(
    target_column="class_stroke_wave_4",
)

# Set up feature groups (temporal dependencies)
# Use a pre-set for ELSA data or define manually (See docs for details)
dataset.setup_features_group(input_data="elsa")

# Initialise the AutoML system
automl = GamaLongitudinalClassifier(
    features_group=dataset.feature_groups(),
    non_longitudinal_features=dataset.non_longitudinal_features(),
    feature_list_names=dataset.data.columns.tolist(),
    max_total_time=3600  # Adjust time as needed (in seconds)
)

# Fit the AutoML system
automl.fit(dataset.X_train, dataset.y_train)

# Make predictions
y_pred = automl.predict(dataset.X_test)

# Print the classification report
print(classification_report(dataset.y_test, y_pred))
```

More detailed examples and tutorials can be found in the [documentation](https://auto-sklong.readthedocs.io/en/latest/tutorials/overview/)!

---

## <a id="citation"></a>📝 How to Cite

If you use Auto-Sklong in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{10821737,
  author={Provost, Simon and Freitas, Alex A.},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
  title={Auto-Sklong: A New AutoML System for Longitudinal Classification}, 
  year={2024},
  volume={},
  number={},
  pages={2021-2028},
  keywords={Pipelines;Optimization;Predictive models;Classification algorithms;Conferences;Bioinformatics;Biomedical computing;Automated Machine Learning;AutoML;Longitudinal Classification;Scikit-Longitudinal;GAMA},
  doi={10.1109/BIBM62325.2024.10821737}}
```

## 🚀 **What's New Compared to GAMA?**

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

## <a id="license"></a>🔐 License

Auto-Sklong is licensed under the [MIT License](./LICENSE).