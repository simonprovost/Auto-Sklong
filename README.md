<!--suppress HtmlDeprecatedAttribute -->
<div align="center">
   <p align="center">
   <h1 align="center">
      <br>
      <a href="https://i.imgur.com/Qu8fIfA.png"><img src="https://i.imgur.com/Qu8fIfA.png" alt="Auto-Sklong" width="200"></a>
      <br>
      Auto-Sklong
      <br>
   </h1>
   <h4 align="center">A specialised Python library for Automated Machine Learning (AutoML) of Longitudinal machine learning classification tasks built upon <a href="https://github.com/openml-labs/gama">GAMA</a> </h4>
   <table align="center">
      <tr>
         <td align="center">
            <h3>⚙️ Project Status</h3>
         </td>
         <td align="center">
            <h3>☎️ Contacts</h3>
         </td>
      </tr>
      <tr>
         <td valign="top">
            <!-- Python-related badges table -->
            <table>
               <tr>
                  <table>
                     <tr>
                        <td>
                           <a href="https://github.com/openml-labs/gama"><img src="https://img.shields.io/badge/Fork-GAMA-green?labelColor=Purple&style=for-the-badge&link=https://github.com/openml-labs/gama" alt="Fork" /></a>
                           <br />
                           <a href="https://pdm.fming.dev">
                           <img alt="pdm" src="https://img.shields.io/badge/pdm-managed-blue?style=for-the-badge&logo=python">
                           </a>
                        </td>
                        <td>
                           <a href="https://pytest.org/">
                           <img alt="pytest" src="https://img.shields.io/badge/pytest-passing-green?style=for-the-badge&logo=pytest">
                           </a><br />
                           <a href="https://codecov.io/gh/Scikit-Longitudinal/Scikit-Longitudinal">
                           <img alt="Codecov" src="https://img.shields.io/badge/coverage-76%25-brightgreen.svg?style=for-the-badge&logo=appveyor">
                           </a>
                        </td>
                     </tr>
                     <tr>
                        <td>
                           <a href="https://www.pylint.org/">
                           <img alt="pylint" src="https://img.shields.io/badge/pylint-checked-blue?style=for-the-badge&logo=python">
                           </a><br />
                           <a href="https://pre-commit.com/">
                           <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-checked-blue?style=for-the-badge&logo=python">
                           </a>
                        </td>
                        <td>
                           <a href="https://github.com/psf/black">
                           <img alt="black" src="https://img.shields.io/badge/black-formatted-black?style=for-the-badge&logo=python">
                           </a><br />
                           <a href="https://github.com/astral-sh/ruff">
                           <img alt="Ruff" src="https://img.shields.io/badge/Linter-Ruff-brightgreen?style=for-the-badge">
                           </a><br />
                        </td>
                     </tr>
                  </table>
                  <td valign="center">
                     <table>
                        <tr>
                           <td>
                                <a href="mailto:s.g.provost@kent.ac.uk">
                                    <img alt="Microsoft Outlook" src="https://upload.wikimedia.org/wikipedia/commons/d/df/Microsoft_Office_Outlook_%282018%E2%80%93present%29.svg" width="40" height="40">
                                </a><br />
                                <a href="https://linkedin.com/in/simonprovostdev/">
                                    <img alt="LinkedIn" src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="40" height="40">
                                </a><br />
                                <a href="https://stackoverflow.com/users/9814037/simon-provost">
                                    <img alt="Stack Overflow" src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" width="40" height="40">
                                </a><br />
                                <a href="https://scholar.google.com/citations?user=Lv_LddYAAAAJ">
                                    <img alt="Google Scholar" src="https://upload.wikimedia.org/wikipedia/commons/c/c7/Google_Scholar_logo.svg" width="40" height="40">
                                </a>
                            </td>
                        </tr>
                     </table>
                  </td>
               </tr>
            </table>
         </td>
      </tr>
   </table>
</div>


> 🌟 **Exciting Update**: We're delighted to introduce the brand new v0.1 documentation for `Auto-Sklong`! For a
> deep dive into the library's capabilities and features,
> please [visit here](https://simonprovost.github.io/Auto-Sklong/).

> 🎉 **PyPi is available!**: We published `Auto-Sklong`, [here](https://pypi.org/project/Auto-Sklong/)!

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

```shell
pip install Auto-Sklong
```
You could also install different versions of the library by specifying the version number, 
e.g. `pip install Auto-Sklong==0.0.1`. 
Refer to [Release Notes](https://github.com/simonprovost/auto-sklong/releases)

### 🚀 **What's new compared to GAMA?**

We improved [@PGijsbers'](https://github.com/PGijsbers) open-source `GAMA` initiative to propose a new search space that 
leverages our other newly-designed library
[`Scikit-Longitudinal` (Sklong)](https://github.com/simonprovost/scikit-longitudinal) in order to tackle the longitudinal
classification problems via Combined Algorithm Selection and Hyperparameter Optimization (CASH Optimization).

Worth noting that it previously was not possible with `GAMA` or any other AutoML libraries to the best of our knowledge
_(refer to the Related Projects in the
[official documentation](https://simonprovost.github.io/scikit-longitudinal/) nonetheless)._

While `GAMA` is offering a way to update the search space, we had to improve `GAMA` to support a couple of new features as follow.
Nonetheless, it is worth-noting that in the coming months, the current version of `Auto-Sklong` might speedy increase due 
to the following pull requests ongoing on `GAMA`:

- [ ] [ConfigSpace Technology Integration for Enhanced GAMA Configuration and Management 🥇](https://github.com/openml-labs/gama/pull/210)
- [ ] [Search Methods Enhancements to Avoid Duplicate Evaluated Pipelines 🥈 #211](https://github.com/openml-labs/gama/pull/211)
- [ ] [SMAC3 Bayesian Optimisation Integration [🆕 Search Method] 🥉 #212](https://github.com/openml-labs/gama/pull/212)

As soon as we are able to publish those on `GAMA`, there will be a compatibility refactoring to align 
`Auto-Sklong` with the most recent version of `GAMA`. As a result, this section will be removed appropriately.

### 💻 Developer Notes

For developers looking to contribute, please refer to the `Contributing` section of `GAMA` [here](https://openml-labs.github.io/gama/master/contributing/index.html)
and `Scikit-Longitudinal` [here](https://simonprovost.github.io/scikit-longitudinal/contribution/).

## <a id="Supported-Operating-Systems"></a>🛠️ Supported Operating Systems

`Auto-Sklong` is compatible with the following operating systems:

- MacOS  
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
    feature_list_names=dataset.data.columns,
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

Paper has been submitted to a conference. In the meantime, for the repository, utilise the button top right corner of the
repository "How to cite?", or open the following citation file: [CITATION.cff](./CITATION.cff).

## <a id="license"></a>🔐 License

[MIT License](./LICENSE)