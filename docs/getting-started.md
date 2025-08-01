---
hide:
  - navigation
---

# 💡 `Auto-Sklong`, in a nutshell!

# 💡 `Auto-Sklong`, in a nutshell!

Longitudinal datasets contain information about the same cohort of individuals (instances) over time, 
with the same set of features (variables) repeatedly measured across different time points 
(also called `waves`) [^1][^2].

`Auto-Scikit-Longitudinal` (abbreviated `Auto-Sklong`, pronounced /ˌɔːtoʊ ˌɛs keɪ ˈlɒŋ/ or "Aw-toh Ess-kay-long") 
is an Automated Machine Learning (AutoML) library built on the [`General Machine Learning Assistant (GAMA)`](https://openml-labs.github.io/gama/master/index.html#) 
framework. It introduces a novel [`search space`](./tutorials/search_space.md) leveraging both 
[`Scikit-Longitudinal (Sklong)`](https://scikit-longitudinal.readthedocs.io/latest/) and [`Scikit-learn`](https://scikit-learn.org/stable/) 
models to tackle Longitudinal Machine Learning _Classification_ tasks. 

Auto-Sklong automates the design of predictive modeling pipelines tailored for longitudinal data, 
handling temporal dependencies seamlessly while integrating with the Scikit-learn ecosystem.

Note that while Longitudinal datasets have a temporal component, other types of datasets, such as time series,
also have a temporal component but are not considered longitudinal datasets. Time series data typically involves
a single variable measured at regular intervals over time, while longitudinal datasets involve multiple variables
measured across the same cohort of individuals at different time points. More is discussed in the [FAQ](faq.md).
However, I would like to highlight that time points are therefore considered as `waves` in `Auto-Sklong` [^1][^2][^3].

To start your Longitudinal AutoML journey with `Auto-Sklong`, you first will have to install the library.

---

## 🛠️ Installation

!!! warning "Operating System Support"
    `Auto-Sklong` is currently supported on `OSX` (MacOS) and `Linux`. 
    `Windows` users should use `notebooks` or `Docker` with a `Linux` (
    Ubuntu) distribution due to limitations with a dependency library. For more details, open an issue, I would be 
    happy to discuss this out further.

!!! important "Python Version Compatibility"
    `Auto-Sklong` is currently compatible with Python versions `3.9` only.
    Ensure you have one of these versions installed before proceeding with the installation.

    Now, while we understand that this is a limitation, we are tied for the time being because of `Deep Forest`.
    `Deep Forest` is a dependency of `Auto-Sklong` that is not compatible with Python versions greater than `3.9`.
    `Deep Forest` helps us with the `Deep Forest` algorithm, to which we have made some modifications to
    welcome `Lexicographical Deep Forest`.

    To follow up on this discussion, please refer to [this github issue](https://github.com/LAMDA-NJU/Deep-Forest/issues/124).

Please, start by choosing the installation method that best suits your needs:

=== ":simple-pypi: PyPi (Classic Install)"

    To install `Auto-Sklong`, you can use `pip`:

    ```bash
    pip install auto-sklong
    ```

    This will install the latest version of `Auto-Sklong` from the Python Package Index (PyPI).

    If you want to install a specific version, you can specify it like this:

    ```bash
    pip install auto-sklong==0.0.1  # Replace with the desired version
    ```

    Please note that here we assume you have a compatible Python version installed (3.9) and a working environment (e.g Conda).

=== ":simple-python: Conda (CondaForge)"

    To install `Auto-Sklong` using `Conda`, follow these steps:

    1. Open your terminal or Anaconda Prompt.
    2. Create a new Conda environment with Python 3.9:

       ```bash
       conda create --name autosklong -c conda-forge python=3.9 
       ```

    3. Activate the environment:

       ```bash
       conda activate autosklong
       ```

    4. Install `Auto-Sklong`:

       ```bash
       pip install auto-sklong
       ```

    This will install `Auto-Sklong` in your newly created Conda environment.

=== ":simple-jupyter: Jupyter Notebook (~ 1 line) via `UV`"

    To run `Jupyter lab` with `Auto-Sklong`, we recommend using `UV`, a fast and efficient Python package 
    manager that simplifies environment and dependency management.

    Here's how to set it up:

    1. Install `UV` if you haven't already. See [UV installation instructions](https://docs.astral.sh/uv/).
    2. Run the following command to launch `Jupyter lab` with `Auto-Sklong`:

       ```bash
       uv run --python /usr/bin/python3 --with auto-sklong jupyter lab
       ```

       Replace `/usr/bin/python3` with the path to your desired Python version, as long as it is `3.9` and less than `3.10`, it has been tested.
       For more options, refer to the [UV CLI documentation](https://docs.astral.sh/uv/reference/cli/#uv-python).

    You are ready to play with `Auto-Sklong` in `Jupyter lab`! 🎉

    ??? question "How to install different version if we do not have `3.9`?"
        You can install a different version of Python using `uv` by running:

        ```bash
        uv python install 3.9
        uv python pin 3.9
        ```
        This command will install Python 3.9 and set it as the default version for your environment.

    ??? question "How do I get the path to my just installed Python version?"
        You can find the path to your installed Python version by running:

        ```bash
        uv python list --all-versions
        ```
        This command will list all installed Python versions along with their paths. If a path is present, it means
        that the version is installed. You can then use the path in the `uv run` command.

    This command creates a temporary environment with `Auto-Sklong` installed and starts `Jupyter lab`.

    ––––––––––––––

    Some shoutouts to the `UV` team for their amazing work! 🙌

    ![UV Proof](https://github.com/astral-sh/uv/assets/1309177/03aa9163-1c79-4a87-a31d-7a9311ed9310#only-dark)

    !!! tip "UV's readings recommendations:"
        - [Python Packaging in Rust](https://astral.sh/blog/uv)
        - [A Year of UV](https://www.bitecode.dev/p/a-year-of-uv-pros-cons-and-should)
        - [UV Is All You Need](https://dev.to/astrojuanlu/python-packaging-is-great-now-uv-is-all-you-need-4i2d)
        - [State of the Art Python 2024](https://4zm.org/2024/10/28/state-of-the-art-python-in-2024.html)
        - [Data Scientist, From School to Work](https://towardsdatascience.com/data-scientist-from-school-to-work-part-i/)

=== ":simple-googlecolab: Google Colab (~5 lines)"

    To use `Auto-Sklong` in `Google Colab`, follow these steps due to compatibility requirements:
    
    You also can follow the follwing gist as we reproduce the below's steps: 
    [gist](https://gist.github.com/simonprovost/356030bd8f1ea077bdbc120fdc116c16#file-support_39_scikit_longitudinal_in_google_colab-ipynb) 
    –– or –– [Open in Google Colab :simple-googlecolab:](https://colab.research.google.com/drive/1Q0wW9Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z) { .md-button }
    
    Preliminary steps:
    
    1. Open a new `Google Colab` notebook.
    2. Open in a code / text editor the notebook you want to use `Auto-Sklong` in, and proceed with the following modifications.
       
        ```
        "metadata": {
            ...
            "kernelspec": {
              "name": "python3.9",
              "display_name": "Python 3.9"
            },
            ...
        },
        ```
    3. Save the notebook.

    You are ready to add stuff in your notebook!

    1. Downgrade the Python version to 3.9, as `Auto-Sklong` supports this version.
         You can do this by running the following command in a code cell:

         Shoutout to @J3soon for [this solution](https://github.com/j3soon/colab-python-version)!
    
         ```bash
         !wget -O py39.sh https://raw.githubusercontent.com/j3soon/colab-python-version/main/scripts/py39.sh
         !bash py39.sh
         ```
    
         This command installs Python 3.9 on your Colab instance.

    2. It'll automatically refresh the kernel to ensure it uses Python 3.9, no worries!
    3. Install `Auto-Sklong`:

       ```bash
       !pip install auto-sklong
       ```

    4. Remove `Scikit-learn` if installed, as `Auto-Sklong` is a fork and may conflict:

       ```bash
       !pip uninstall scikit-learn -y
       ```

    5. Remove & Re-Install `Scikit-lexicographical-trees`, which is the modified version of `Scikit-learn` used by `Auto-Sklong`:

       ```bash
       !pip uninstall scikit-lexicographical-trees -y
       !pip install scikit-lexicographical-trees
       ```

    After these steps, you can use `Auto-Sklong` in your Colab notebook 🎉

=== ":light_blue_heart: Marimo"

    Support for Marimo is incoming. If you're interested in contributing to this feature, please submit a pull request!

=== ":simple-codingninjas: Within A Project"

    For developing your own scripts with `Auto-Sklong`, install it via pip:

    ```bash
    pip install auto-sklong
    ```

    Following the pip install, let's explore various scenarios. As follows:

    #### 🫵 Project Setup: Using PDM
    
    If you’re managing your project dependencies with `PDM`, note that `Auto-Sklong` is a fork of `Scikit-Learn` 
    and is incompatible with the original `Scikit-Learn` package. To ensure compatibility, exclude `Scikit-Learn` from 
    your project dependencies by adding the following configuration to your `pyproject.toml` file:
    
    ````toml
    [tool.pdm.resolution]
    excludes = ["scikit-learn"]
    ````
    
    This ensures that the modified version of `Scikit-Learn`—`Scikit-Lexicographical-Trees`—is used seamlessly within your project.
    
    To install dependencies:
    ```shell
    pdm install
    ```
    
    To install only production dependencies:
    ```shell
    pdm install --prod
    ```
    
    For additional configurations, refer to the [PDM documentation](https://pdm.fming.dev/).
    
    ---
    
    #### 🫵 Project Setup: Using UV
    
    If you prefer **UV** for dependency management, configure your `pyproject.toml` file to override conflicting 
    packages. Add the following configuration:
    
    ````toml
    [tool.uv]
    package = true
    override-dependencies = [
        "scikit-learn ; sys_platform == 'never'",
    ]
    ````
    
    Steps to set up your environment:
    1. **Create a Virtual Environment:**

       ```bash
       uv venv
       ```
    
    2. **Pin the Required Python Version:**
       ```bash
       uv python pin cpython-3.9.21 # Or any other version you want as long as it fits Auto-Sklong requirements.
       ```
    
    3. **Lock Dependencies:**
       ```bash
       uv lock
       ```
    
    4. **Install All Dependencies:**
       ```bash
       uv sync --all-groups
       ```
    
    5. **Run Tests:**
       ```bash
       uv run pytest -sv auto_sklong -p no:warnings
       ```
    
    For more information, refer to the [UV documentation](https://docs.astral.sh/uv/).
    
    ---
    
    ### 💻 Developer Notes
    
    For developers looking to contribute, please refer to the `Contributing` section of the [documentation](https://auto-sklong.readthedocs.io/latest/).

---

## 🚀 Quick Start (Code)

`Auto-Sklong` has numerous primitives to deal with longitudinal machine learning classification tasks. To begin, use the
`LongitudinalDataset` class from [`Sklong`](https://scikit-longitudinal.readthedocs.io/latest/) to prepare your dataset, including `data` and `temporal vectors`. To train an AutoML pipeline on your data, use the `GamaLongitudinalClassifier` class.

> "The `GamaLongitudinalClassifier` in a nutshell: a wrapper around GAMA's AutoML system, tailored for longitudinal data, automating the search for optimal pipelines using a novel search space."

!!! tip "Where's the data at?"
    The `Auto-Sklong` library does not include datasets by default, mainly due to privacy reasons.
    You can use your own longitudinal datasets
    or download publicly available ones, such as the [ELSA dataset](https://www.elsa-project.ac.uk/). 
    If synthetic datasets are of interest, open an issue; I would be happy to discuss this further.

Here's our basic workflow example:

```python
from sklearn.metrics import classification_report
from scikit_longitudinal.data_preparation import LongitudinalDataset
from gama.GamaLongitudinalClassifier import GamaLongitudinalClassifier

# Load your dataset (replace 'stroke.csv' with your actual dataset path)
dataset = LongitudinalDataset('./stroke.csv')

# Set up the target column and split the data into training and testing sets (replace 'stroke_wave_4' with your target column)
dataset.load_data_target_train_test_split(
    target_column="class_stroke_wave_4",
)

# Set up feature groups (temporal dependencies)
# Use a pre-set for ELSA data or define manually (See further in the API reference)
dataset.setup_features_group(input_data="elsa")

# Initialise the AutoML system with feature groups
automl = GamaLongitudinalClassifier(
    features_group=dataset.feature_groups(),
    non_longitudinal_features=dataset.non_longitudinal_features(),
    feature_list_names=dataset.data.columns.tolist(),
    max_total_time=3600  # Adjust time as needed (in seconds)
)

# Fit the AutoML system to the training data
automl.fit(dataset.X_train, dataset.y_train)

# Make predictions on the test data
y_pred = automl.predict(dataset.X_test)

# Print the classification report
print(classification_report(dataset.y_test, y_pred))
```

---

## ❓ Questions

!!! question "What are feature groups?"
    Feature groups define the temporal dependencies in your longitudinal data. They are lists of feature indices
    corresponding to different waves. See
    the [Temporal Dependency](https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/) section for more
    details.

!!! question "How do I set temporal dependencies?"
    Use pre-sets for known datasets like ELSA or define them manually based on your data structure. Refer to
    the [Temporal Dependency](https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/) section.

!!! question "How do I tune hyperparameters?"
    Check the [API Reference](API/index.md) for a complete list of
    hyperparameters and their meanings.

!!! question "Neural Network Models?"
    `Auto-Sklong` currently does not support neural network-based models. For similar projects that do, see
    the [FAQ](faq.md) section.

---

## What's Next?

Next, we highly recommend that you explore the [`Search Space` section](./tutorials/search_space.md), which provides a comprehensive
understanding of Auto-Sklong's novel search space for longitudinal pipelines.

Following that? Dive into the `API Reference` section, which provides detailed
information on the various search methods, hyperparameters, and classes available in `Auto-Sklong`.
___

[^1]: Kelloway, E.K. and Francis, L., 2012. Longitudinal research and data analysis. In Research methods in occupational
health psychology (pp. 374-394). Routledge.

[^2]: Ribeiro, C. and Freitas, A.A., 2019. A mini-survey of supervised machine learning approaches for coping with
ageing-related longitudinal datasets. In 3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (
ARIAL), held as part of IJCAI-2019 (num. of pages: 5).

[^3]: Ribeiro, C. and Freitas, A.A., 2024. A lexicographic optimisation approach to promote more recent features on
longitudinal decision-tree-based classifiers: applications to the English Longitudinal Study of Ageing. Artificial
Intelligence Review, 57(4), p.84.


## 🚨 Troubleshooting

=== ":simple-apple: Install On Apple Silicon Chips"
    Apple Silicon-based Macs require running under an `x86_64` architecture to ensure proper installation and
    functioning of `Auto-Sklong`. This is primarily due to the `Deep-Forest` dependency being incompatible
    with Apple Silicon (ref Github Issue with `Deep Forest`
    authors [here](https://github.com/LAMDA-NJU/Deep-Forest/issues/133)).

    The following steps are somehow extracted & adapted
    from [https://apple.stackexchange.com/a/408379](https://apple.stackexchange.com/a/408379).

    1. **Install Rosetta 2**
       Rosetta 2 allows your Apple Silicon Mac (M1, M2, etc.) to run apps built for the `x86_64` architecture,
       which is needed for `Auto-Sklong` due to its `Deep-Forest` dependency.
        ```bash
        softwareupdate --install-rosetta
        ```
       Note: When prompted, press `'A'` and `Enter` to agree to the license terms.
    
    2. **Launch Terminal in `x86_64` Mode**
       Restart your terminal (close and reopen it), then start a new shell session under the `x86_64` architecture.
       ```bash
       arch -x86_64 zsh
       ```
       Note: To verify so you can run `uname -m` and it should output `x86_64`.
    
    3. **Install `Auto-Sklong` via `Conda` with `Pip`**
       ```bash
       conda create --name autosklong python=3.9
       conda activate autosklong
       pip install auto-sklong
       ```
    4. **Verify Installation**
       ```bash
       python -c "import auto_sklong"
       ```
    
    And voila! You should now have `Auto-Sklong` installed and running on your Apple Silicon Mac.