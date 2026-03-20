<!--suppress HtmlDeprecatedAttribute -->
<div align="center">
   <p align="center">
      <br>
      <a href="docs/assets/images/AutoSklong_banner.avif">
         <img src="docs/assets/images/AutoSklong_banner.avif" alt="Auto-Sklong banner">
      </a>
   </p>
   <h4 align="center">
      An Automated Machine Learning system for longitudinal classification built on GAMA and Scikit-Longitudinal —
      <a href="https://doi.org/10.1109/BIBM62325.2024.10821737">Paper</a> ·
      <a href="https://auto-sklong.readthedocs.io/en/latest/">Documentation</a> ·
      <a href="https://pypi.org/project/Auto-Sklong/">PyPi Index</a>
   </h4>
</div>

<div align="center">

<a href="https://pytest.org/">
   <img alt="pytest" src="https://img.shields.io/static/v1?label=Pytest&message=passing&color=0E5DF0&labelColor=FA9817&style=for-the-badge&logo=pytest&logoColor=white">
</a>

<a href="https://pre-commit.com/">
   <img alt="pre-commit" src="https://img.shields.io/static/v1?label=Pre-commit&message=checked&color=0E5DF0&labelColor=FA9817&style=for-the-badge&logo=pre-commit&logoColor=white">
</a>

<a href="https://github.com/psf/black">
   <img alt="black" src="https://img.shields.io/static/v1?label=Black&message=formatted&color=0E5DF0&labelColor=FA9817&style=for-the-badge&logo=black&logoColor=white">
</a>

<img src="https://img.shields.io/static/v1?label=Ruff&message=compliant&color=0E5DF0&labelColor=FA9817&style=for-the-badge&logo=ruff&logoColor=white" alt="Ruff compliant">

<img src="https://img.shields.io/static/v1?label=UV&message=managed&color=0E5DF0&labelColor=FA9817&style=for-the-badge&logo=uv&logoColor=white" alt="UV managed">

<a href="https://codecov.io/gh/simonprovost/Auto-Sklong">
   <img alt="Coverage" src="https://img.shields.io/static/v1?label=Coverage&message=76%25&color=0E5DF0&labelColor=FA9817&style=for-the-badge&logo=appveyor&logoColor=white">
</a>

<a href="https://github.com/openml-labs/gama">
   <img alt="GAMA fork" src="https://img.shields.io/static/v1?label=Fork&message=GAMA&color=0E5DF0&labelColor=FA9817&style=for-the-badge&logo=github&logoColor=white">
</a>

<img src="https://img.shields.io/static/v1?label=Python&message=3.9&color=0E5DF0&labelColor=FA9817&style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9">

</div>

## <a id="about-the-project"></a><img src="docs/assets/icons/lucide/github.svg" width="32" alt="" /> About The Project

`Auto-Sklong` is an Automated Machine Learning (AutoML) library developed on top of the
[`General Machine Learning Assistant (GAMA)`](https://openml-labs.github.io/gama/master/index.html#) framework.
It introduces a dedicated [`search space`](https://auto-sklong.readthedocs.io/en/latest/tutorials/search_space/)
that combines [`Scikit-Longitudinal`](https://scikit-longitudinal.readthedocs.io/latest/) and
[`Scikit-learn`](https://scikit-learn.org/stable/) models to tackle longitudinal classification tasks.
More specifically, `Auto-Sklong` is designed to perform combinatorial optimisation of both algorithm selection
and the tuning of their associated hyperparameters in the context of longitudinal machine learning.

**Wait, what is Longitudinal Data - In layman's terms?**

Longitudinal data is a "time-lapse" snapshot of the same subject, entity, or group tracked over time periods,
similar to checking in on patients to see how they change. For instance, doctors may monitor a patient's blood pressure,
weight, and cholesterol every year for a decade to identify health trends or risk factors. This data is more useful for
predicting future results than a one-time survey because it captures evolution, patterns, and cause-effect throughout
time.

> [!IMPORTANT]
> We are currently revamping the whole repository to support:
> - Longitudinal-data-aware post-hoc ensembling
> - Python 3.10+ via [Scikit-Longitudinal 0.1.8+](https://scikit-longitudinal.readthedocs.io/latest/)
> - New documentation
>
> Please bear with us while we work on this. The focus has primarily been on bringing
> [Scikit-Longitudinal](https://scikit-longitudinal.readthedocs.io/latest/) support to Python 3.10+,
> integrating new algorithms from the community, and finalising multi-class support, all of which are now
> very close to completion.


## <a id="installation"></a><img src="docs/assets/icons/lucide/terminal.svg" width="32" alt="" /> Installation

To install Auto-Sklong:

```bash
pip install auto-sklong
```

To install a specific version:

```bash
pip install auto-sklong==0.0.1
```

> [!TIP]
> Want to use `Jupyter Notebook`, `Marimo`, `Google Colab`, or `JupyterLab`?
> Head to the [Getting Started](https://auto-sklong.readthedocs.io/en/latest/getting-started/) section of the documentation.

> [!CAUTION]
> `Auto-Sklong` is currently compatible with Python `3.9` only.
> This limitation stems from the `Deep Forest` dependency.
> Follow updates on [this GitHub issue](https://github.com/LAMDA-NJU/Deep-Forest/issues/124).

## <a id="getting-started"></a><img src="docs/assets/icons/lucide/square-code.svg" width="32" alt="" /> Getting Started

Here's how to run AutoML on longitudinal data with Auto-Sklong:

```python
from sklearn.metrics import classification_report
from scikit_longitudinal.data_preparation import LongitudinalDataset
from gama.GamaLongitudinalClassifier import GamaLongitudinalClassifier

# Load your dataset (replace "stroke.csv" with your actual dataset path)
dataset = LongitudinalDataset("./stroke.csv")

# Set up the target column and split the data
dataset.load_data_target_train_test_split(
    target_column="class_stroke_wave_4",
)

# Set up feature groups (temporal dependencies)
dataset.setup_features_group(input_data="elsa")

# Initialise the AutoML system
automl = GamaLongitudinalClassifier(
    features_group=dataset.feature_groups(),
    non_longitudinal_features=dataset.non_longitudinal_features(),
    feature_list_names=dataset.data.columns.tolist(),
    max_total_time=3600,  # Time budget in seconds
)

# Fit the AutoML system
automl.fit(dataset.X_train, dataset.y_train)

# Make predictions
y_pred = automl.predict(dataset.X_test)

# Print the classification report
print(classification_report(dataset.y_test, y_pred))
```

More detailed examples and tutorials can be found in the
[documentation](https://auto-sklong.readthedocs.io/en/latest/tutorials/overview/).

## <a id="citation"></a><img src="docs/assets/icons/lucide/graduation-cap.svg" width="32" alt="" /> How to Cite

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

## <a id="license"></a><img src="docs/assets/icons/lucide/fingerprint-pattern.svg" width="32" alt="" /> License

Auto-Sklong is licensed under the [MIT License](./LICENSE).
