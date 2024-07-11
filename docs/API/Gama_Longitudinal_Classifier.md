# Gama Longitudinal Classifier

## GamaLongitudinalClassifier

[source](https://github.com/simonprovost/auto-sklong/blob/main/gama/GamaLongitudinalClassifier.py/#L51)

```python
GamaLongitudinalClassifier(
   features_group: List[List[Union[int, str]]],
   non_longitudinal_features: List[Union[int, str]],
   feature_list_names: List[str], update_feature_groups_callback: Union[Callable, str] = 'default',
   search_space: Optional[cs.ConfigurationSpace] = None, scoring: Metric = 'roc_auc',
   max_pipeline_length: Optional[int] = None, search: Optional[BaseSearch] = None,
   *args, **kwargs
)
```
The `GamaLongitudinalClassifier` is the principal class for running `Auto-Sklong`, specifically designed to 
handle longitudinal datasets. It integrates the proposed [`search space`](https://simonprovost.github.io/Auto-Sklong/search_space/) and employs various search methods to effectively 
explore the longitudinal-based [`search space`](https://simonprovost.github.io/Auto-Sklong/search_space/) for your dataset.

!!! note "Naming Convention: `GamaLongitudinalClassifier`"
    The name `GamaLongitudinalClassifier` reflects its foundation on the [`GAMA`](https://openml-labs.github.io/gama/master/index.html#) framework, ensuring consistency with 
    the framework's naming conventions, even though it is a specialised fork tailored for longitudinal data.

!!! tip "Understanding `features_group` and `non_longitudinal_features`"
    To leverage the full potential of `GamaLongitudinalClassifier`, it is crucial to understand the mandatory parameters 
    `features_group` and `non_longitudinal_features`. These parameters enable the classifiers' abilities to manage the
    temporal dependencies in your dataset. Detailed information can be found in the [Temporal Dependencies documentation page](https://simonprovost.github.io/scikit-longitudinal/temporal_dependency/)

!!! warning "Refer to `GAMA base` Documentation"
    The `GamaLongitudinalClassifier` inherits many parameters from the [`GAMA`](https://openml-labs.github.io/gama/master/index.html#) base class (e.g the max total time the AutoML search goes on).
    A core class in the [`GAMA`](https://openml-labs.github.io/gama/master/index.html#) AutoML framework to all sub-class such as the standard `GamaClassifier` or `GamaRegressor` classes.
    For comprehensive details on these inherited parameters, consult the [GAMA Base API documentation](https://openml-labs.github.io/gama/master/api/index.html#gama)


## Parameters

- **features_group** (`List[List[Union[int, str]]]`): A temporal matrix representing the temporal dependency of a longitudinal dataset. Each tuple/list of integers in the outer list represents the indices of a longitudinal attribute's waves, with each longitudinal attribute having its own sublist in that outer list. For more details, see the documentation's [`Temporal Dependency`](https://simonprovost.github.io/scikit-longitudinal/temporal_dependency/) page.
- **non_longitudinal_features** (`List[Union[int, str]]`): A list of indices or names of features that are not longitudinal.
- **feature_list_names** (`List[str]`): A list of feature names in the dataset.
- **update_feature_groups_callback** (`Union[Callable, str]`, optional): Callback function to update feature groups. This function is invoked to update the structure of longitudinal features during pipeline transformations. By default, if you did not change the [`search space`](https://simonprovost.github.io/Auto-Sklong/search_space/), you should be all right not to set it it up. Read further in the [`Sklong`](https://simonprovost.github.io/scikit-longitudinal/) documentation [here](https://simonprovost.github.io/scikit-longitudinal/API/pipeline/#detailed-explanation-of-update_feature_groups_callback)
- **search_space** (`Optional[cs.ConfigurationSpace]`): The [`search space`](https://simonprovost.github.io/Auto-Sklong/search_space/) being explored by the system. The original is by default, another one could be passed. Further explore the `Search_Space` documentation page.
- **scoring** (`Metric`, default='roc_auc'): The metric used for evaluating the performance of the model. Similar to `Scikit-Learn` / `Auto-Sklearn`, you can pass a custom `Metric` scoring function that the AutoML system will have to optimise.
- **max_pipeline_length** (`Optional[int]`, default=None): The maximum length of the pipeline to be searched. Whether you would like to have a `pre-processing` technique in the middle or not. If you put `2`, it will discard the `pre-processing` step of the [`search space`](https://simonprovost.github.io/Auto-Sklong/search_space/).
- **search** (`Optional[BaseSearch]`, default=None): The search strategy used for pipeline optimization. Either, `BayesianOptimization`, `RandomSearch`, `Asynchronous Evolutionary Algorithm`, or `Asynchronous Successive Halving`. Further explore the [Search Methods](https://simonprovost.github.io/Auto-Sklong/API/#search-methods) documentation section.
- **args**: Additional arguments.
- **kwargs**: Additional keyword arguments.

_More are available, here in the [GAMA base](https://openml-labs.github.io/gama/master/api/index.html#gama) documentation. Such as, the `max_total_time`, `max_eval_time`, etc._

## Attributes

- **classes_** (`ndarray` of shape (n_classes,)): The class labels.
- **n_classes_** (`int`): The number of classes.
- **feature_importances_** (`ndarray` of shape (n_features,)): The feature importances.
- **best_pipeline_** (`Pipeline` object): The best pipeline found during the search.

## Methods

### Fit
[source](https://github.com/simonprovost/auto-sklong/blob/main/gama/GamaLongitudinalClassifier.py/#L368)

```python
.fit(
   x: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
   *args, **kwargs
)
```

Fit the `Gama Longitudinal Classifier`. Or in another word, start the AutoML experiment on your input data.
This takes the amount of time you set in the `max_total_time` parameter of the [GAMA Base class](https://openml-labs.github.io/gama/master/api/index.html#gama).

#### Parameters

- **x** (`Union[pd.DataFrame, np.ndarray]`): The training input samples of shape `(n_samples, n_features)`.
- **y** (`Union[pd.Series, np.ndarray]`): The target values of shape `(n_samples,)`.
- **args**: Additional arguments.
- **kwargs**: Additional keyword arguments.

#### Returns

- **GamaLongitudinalClassifier**: The fitted AutoML classifier. Capable of predicting the target for input X using the best found pipeline.

### Predict
[source](https://github.com/simonprovost/Auto-Sklong/blob/07e65ba697e767a2bb7de8e29fc231f0025290fe/gama/GamaLongitudinalClassifier.py#L301)

```python
.predict(
   x: Union[pd.DataFrame, np.ndarray]
)
```

Predict the target for input X using the best found pipeline.

This method computes the class labels for the input samples.

#### Parameters

- **x** (`Union[pd.DataFrame, np.ndarray]`): The input samples of shape `(n_samples, n_features)`.

#### Returns

- **np.ndarray**: The predicted class labels of shape `(n_samples,)`.

### Predict Proba
[source](https://github.com/simonprovost/auto-sklong/blob/main/gama/GamaLongitudinalClassifier.py/#L319)

```python
.predict_proba(
   x: Union[pd.DataFrame, np.ndarray]
)
```

Predict class probabilities for X using the best found pipeline.

This method computes the class probabilities for the input samples.

#### Parameters

- **x** (`Union[pd.DataFrame, np.ndarray]`): The input samples of shape `(n_samples, n_features)`.

#### Returns

- **np.ndarray**: The predicted class probabilities of shape `(n_samples, n_classes)`.

### Predict Proba from File
[source](https://github.com/simonprovost/auto-sklong/blob/main/gama/GamaLongitudinalClassifier.py/#L338)

```python
.predict_proba_from_file(
   arff_file_path: str, target_column: Optional[str] = None,
   encoding: Optional[str] = None
)
```

Predict class probabilities from an ARFF file.

This method computes the class probabilities for the input samples in the ARFF file.

#### Parameters

- **arff_file_path** (`str`): The path to the ARFF file containing the input samples.
- **target_column** (`Optional[str]`, default=None): The name of the target column in the ARFF file. If None, the last column is assumed to be the target.
- **encoding** (`Optional[str]`, default=None): The encoding of the ARFF file.

#### Returns

- **np.ndarray**: The predicted class probabilities of shape `(n_samples, n_classes)`.

## Examples

!!! tip "Make sure to open the little `+` icon next to any line having them"
    The examples below are just a starting point. Make sure to open the little `+` icon next to any line having them to see further explanation / guidance.

### Dummy Longitudinal Dataset

!!! example "Consider the following dataset: `stroke.csv`"
    Features:

    - `smoke` (longitudinal) with two waves/time-points
    - `cholesterol` (longitudinal) with two waves/time-points
    - `age` (non-longitudinal)
    - `gender` (non-longitudinal)

    Target:

    - `stroke` (binary classification) at wave/time-point 2 only for the sake of the example

    The dataset is shown below:

    | smoke_w1 | smoke_w2 | cholesterol_w1 | cholesterol_w2 | age | gender | stroke_w2 |
    |----------|----------|----------------|----------------|-----|--------|-----------|
    | 0        | 1        | 0              | 1              | 45  | 1      | 0         |
    | 1        | 1        | 1              | 1              | 50  | 0      | 1         |
    | 0        | 0        | 0              | 0              | 55  | 1      | 0         |
    | 1        | 1        | 1              | 1              | 60  | 0      | 1         |
    | 0        | 1        | 0              | 1              | 65  | 1      | 0         |

### Example 1: Basic Usage

```python
from sklearn.metrics import classification_report
from scikit_longitudinal.data_preparation import LongitudinalDataset
from gama.GamaLongitudinalClassifier import GamaLongitudinalClassifier

# Load your longitudinal dataset
dataset = LongitudinalDataset('./stroke.csv') # (1)
dataset.load_data_target_train_test_split(
  target_column="stroke_w2",
)

# Pre-set or manually set your temporal dependencies 
dataset.setup_features_group(input_data="elsa") # (2)

# Instantiate the AutoML system
automl = GamaLongitudinalClassifier( # (3)
    features_group=dataset.features_group(),
    non_longitudinal_features=dataset.non_longitudinal_features(), # (4)
    feature_list_names=dataset.data.columns,
)

# Run the AutoML system to find the best model and hyperparameters
automl.fit(dataset.X_train, dataset.y_train)

# Predictions and prediction probabilities
label_predictions = automl.predict(X_test)
probability_predictions = automl.predict_proba(X_test)

# Classification report
print(classification_report(y_test, label_predictions))
```

1. To further explore the documentation about `LongitudinalDataset` available via [`Sklong`](https://simonprovost.github.io/scikit-longitudinal/), read [here](https://simonprovost.github.io/scikit-longitudinal/API/data_preparation/longitudinal_dataset/).
2. Define the features_group manually or use a pre-set from the `LongitudinalDataset` class. If the data was from the ELSA database, you could use as per the example the pre-sets such as `.setup_features_group('elsa')`.
3. Instantiate the `GamaLongitudinalClassifier` class with the features_group and non-longitudinal features and the rest by default.
4. Define the non-longitudinal features manually or use a pre-set from the `LongitudinalDataset` class. If the data was from the ELSA database, you could use as per the example the pre-sets such as `.setup_features_group('elsa')`, then the non-longitudinal features would have been automatically set.


### Example 2: How to set my own scoring function

```python
from sklearn.metrics import f1_score
from scikit_longitudinal.data_preparation import LongitudinalDataset
from gama.GamaLongitudinalClassifier import GamaLongitudinalClassifier


# Load your longitudinal dataset
dataset = LongitudinalDataset('./stroke.csv') # (1)
dataset.load_data_target_train_test_split(
  target_column="stroke_w2",
)

# Pre-set or manually set your temporal dependencies 
dataset.setup_features_group(input_data="elsa") # (2)

# Instantiate the AutoML system
automl = GamaLongitudinalClassifier( # (3)
    features_group=dataset.features_group(),
    non_longitudinal_features=dataset.non_longitudinal_features(), # (4)
    feature_list_names=dataset.data.columns,
    scoring=f1_score # (5)
)

# Run the AutoML system to find the best model and hyperparameters
automl.fit(dataset.X_train, dataset.y_train)

# Predictions and prediction probabilities
label_predictions = automl.predict(X_test)
probability_predictions = automl.predict_proba(X_test)

# Classification report
print(classification_report(y_test, label_predictions))
```

1. To further explore the documentation about `LongitudinalDataset` available via [`Sklong`](https://simonprovost.github.io/scikit-longitudinal/), read [here](https://simonprovost.github.io/scikit-longitudinal/API/data_preparation/longitudinal_dataset/).
2. Define the features_group manually or use a pre-set from the `LongitudinalDataset` class. If the data was from the ELSA database, you could use as per the example the pre-sets such as `.setup_features_group('elsa')`.
3. Instantiate the `GamaLongitudinalClassifier` class with the features_group and non-longitudinal features and the rest by default.
4. Define the non-longitudinal features manually or use a pre-set from the `LongitudinalDataset` class. If the data was from the ELSA database, you could use as per the example the pre-sets such as `.setup_features_group('elsa')`, then the non-longitudinal features would have been automatically set.
5. Set the scoring function to `f1_score` instead of the default `roc_auc`. You could create your own scorer with `make_scorer` from `sklearn.metrics` if you want to use a custom metric. See furthere [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).


### Example 3: How to export the best pipeline

```python
from sklearn.metrics import classification_report
from scikit_longitudinal.data_preparation import LongitudinalDataset
from gama.GamaLongitudinalClassifier import GamaLongitudinalClassifier

# Load your longitudinal dataset
dataset = LongitudinalDataset('./stroke.csv') # (1)
dataset.load_data_target_train_test_split(
  target_column="stroke_w2",
)

# Pre-set or manually set your temporal dependencies 
dataset.setup_features_group(input_data="elsa") # (2)

# Instantiate the AutoML system
automl = GamaLongitudinalClassifier( # (3)
    features_group=dataset.features_group(),
    non_longitudinal_features=dataset.non_longitudinal_features(), # (4)
    feature_list_names=dataset.data.columns,
)

# Run the AutoML system to find the best model and hyperparameters
automl.fit(dataset.X_train, dataset.y_train)

# Export the best pipeline
automl.export_script('my_pipeline_script.py') # (5)
```

1. To further explore the documentation about `LongitudinalDataset` available via [`Sklong`](https://simonprovost.github.io/scikit-longitudinal/), read [here](https://simonprovost.github.io/scikit-longitudinal/API/data_preparation/longitudinal_dataset/).
2. Define the features_group manually or use a pre-set from the `LongitudinalDataset` class. If the data was from the ELSA database, you could use as per the example the pre-sets such as `.setup_features_group('elsa')`.
3. Instantiate the `GamaLongitudinalClassifier` class with the features_group and non-longitudinal features and the rest by default.
4. Define the non-longitudinal features manually or use a pre-set from the `LongitudinalDataset` class. If the data was from the ELSA database, you could use as per the example the pre-sets such as `.setup_features_group('elsa')`, then the non-longitudinal features would have been automatically set.
5. Export the best pipeline to a Python file named `my_pipeline_script.py`. The resulting script will define a variable `pipeline` or `ensemble`, depending on the post-processing method that was used after search. Reader further about the post-ensembling methods in the [`GAMA`](https://openml-labs.github.io/gama/master/index.html#) class documentation.

### Example 4: How to play with the [`GAMA`](https://openml-labs.github.io/gama/master/index.html#) base parameters

```python
from sklearn.metrics import classification_report
from scikit_longitudinal.data_preparation import LongitudinalDataset
from gama.GamaLongitudinalClassifier import GamaLongitudinalClassifier
from gama.search_methods import RandomSearch

# Load your longitudinal dataset
dataset = LongitudinalDataset('./stroke.csv') # (1)
dataset.load_data_target_train_test_split(
  target_column="stroke_w2",
)

# Pre-set or manually set your temporal dependencies 
dataset.setup_features_group(input_data="elsa") # (2)

# Instantiate the AutoML system
automl = GamaLongitudinalClassifier( # (3)
    features_group=dataset.features_group(),
    non_longitudinal_features=dataset.non_longitudinal_features(), # (4)
    feature_list_names=dataset.data.columns,
    max_total_time=86400, # (5)
    max_eval_time=1000, # (6)
    n_jobs=4, # (7)
    max_memory_mb=2000, # (8)
    post_processing=EnsemblePostProcessing(), # (9)
    output_directory="my_output", # (10)
    store="all", # (11)
    search=RandomSearch(), # (12)
)

# Run the AutoML system to find the best model and hyperparameters
automl.fit(dataset.X_train, dataset.y_train)

# Predictions and prediction probabilities
label_predictions = automl.predict(X_test)
probability_predictions = automl.predict_proba(X_test)

# Classification report
print(classification_report(y_test, label_predictions))
```

1. To further explore the documentation about `LongitudinalDataset` available via [`Sklong`](https://simonprovost.github.io/scikit-longitudinal/), read [here](https://simonprovost.github.io/scikit-longitudinal/API/data_preparation/longitudinal_dataset/).
2. Define the features_group manually or use a pre-set from the `LongitudinalDataset` class. If the data was from the ELSA database, you could use as per the example the pre-sets such as `.setup_features_group('elsa')`.
3. Instantiate the `GamaLongitudinalClassifier` class with the features_group and non-longitudinal features and the rest by default.
4. Define the non-longitudinal features manually or use a pre-set from the `LongitudinalDataset` class. If the data was from the ELSA database, you could use as per the example the pre-sets such as `.setup_features_group('elsa')`, then the non-longitudinal features would have been automatically set.
5. Set the maximum total time – in seconds – for the AutoML system to 24 hours (86400 seconds). Read further in the `Gama base` class documentation.
6. Set the maximum evaluation time – in seconds – for each pipeline evaluation to 1000 seconds. Read further in the `Gama base` class documentation.
7. Turn on the parallel processing with 4 jobs. This means that 4 candidates at the same time will be able to be evaluated, if the number of CPUs availale permits-so. Read further in the `Gama base` class documentation.
8. Set the maximum memory usage – in megabytes – for the AutoML system to 2000 MB. This means that above the current candidate's evaluation will crash. Read further in the `Gama base` class documentation.
9. Set the post-processing method to `EnsemblePostProcessing`. This will ensemble the best pipelines found during the search. Read further in the official [`GAMA`](https://openml-labs.github.io/gama/master/index.html#) documentation [here](https://openml-labs.github.io/gama/master/api/index.html#module-gama.postprocessing).
10. Set the output directory to `my_output`. This will save the results of the search in the `my_output` directory. Read further in the `Gama base` class documentation.
11. Set the store level to `all`, which keep logs and cache with models and predictions. Read further in the `Gama base` class documentation.
12. Run the AutoML's search under `RandomSearch` strategy. Others are available, read further in the [Search Methods section](https://simonprovost.github.io/Auto-Sklong/API/#search-methods).
