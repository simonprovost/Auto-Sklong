""" Contains full system tests for GamaClassifier """

import warnings

import ConfigSpace as cs
import numpy as np
import pytest
from scikit_longitudinal.data_preparation import LongitudinalDataset
from sklearn.metrics import accuracy_score

from gama.GamaLongitudinalClassifier import GamaLongitudinalClassifier
from gama.configuration.longitudinal_classification_task import (
    LongitudinalClassifierConfig,
    LongitudinalPreprocessorConfig,
)
from gama.configuration.longitudinal_classification_task.data_preparation import (
    LongitudinalDataPreparationConfig,
)
from gama.configuration.testconfiguration import config_space
from gama.postprocessing import EnsemblePostProcessing
from gama.search_methods import AsynchronousSuccessiveHalving, AsyncEA, RandomSearch
from gama.search_methods.base_search import BaseSearch
from gama.search_methods.bayesian_optimisation import BayesianOptimisation
from gama.utilities.generic.stopwatch import Stopwatch

warnings.filterwarnings("error")

FIT_TIME_MARGIN = 1.1

# While we could derive statistics dynamically,
# we want to know if any changes ever happen, so we save them statically.
arthritis_disease = dict(
    name="arthritis",
    reduce_data_fraction=0.02,
    n_classes=2,
    base_accuracy=0.40,
    random_state=42,
)


def _get_data_elsa(name: str, reduce_data_fraction: float, random_state: int):
    """
    This function is for the ELSA database.
    Contact authors for access. On the other hand, if you have access to the
    ELSA database, you can use this function to load the data and test
    the GamaLongitudinalClassifier but replace the "_get_data(.)" with this one.
    """
    longitudinal_data = LongitudinalDataset(
        # your path here
    )
    longitudinal_data.load_data()
    longitudinal_data.set_data(
        longitudinal_data._data.sample(
            frac=reduce_data_fraction, random_state=random_state
        )
    )
    target_column = f"class_{name}_w8"
    longitudinal_data.load_target(target_column=target_column, remove_target_waves=True)
    longitudinal_data.load_train_test_split(random_state=random_state)
    longitudinal_data.setup_features_group(input_data="elsa")
    return longitudinal_data


def _get_data(name: str, reduce_data_fraction: float, random_state: int):
    """
    Default Get Data. See Get Data Elsa for further use of name,
    reduce_data_fraction and random_state.

    Currently, is being loaded a dummy dataset.
    Could be replaced with a real dataset of your choice.
    Nonetheless, either public via e.g OpenML or local but if local,
    do not forget to change the path prior to pushing.
    """
    _, _, _ = name, reduce_data_fraction, random_state
    data_file = "tests/data/stroke_longitudinal.csv"
    target_column = "stroke_w2"

    longitudinal_data = LongitudinalDataset(data_file)
    longitudinal_data.load_data_target_train_test_split(
        target_column, remove_target_waves=False, random_state=42
    )
    longitudinal_data.setup_features_group(input_data="elsa")
    return longitudinal_data


def _test_dataset_problem(
    data,
    metric: str,
    search: BaseSearch = AsyncEA(),
    max_time: int = 60,
):
    longitudinal_data = _get_data(
        data["name"], data["reduce_data_fraction"], data["random_state"]
    )
    features_group = longitudinal_data.feature_groups()
    non_longitudinal_features = longitudinal_data.non_longitudinal_features()
    feature_list_names = longitudinal_data.data.columns.tolist()

    X_train = longitudinal_data.X_train
    y_train = longitudinal_data.y_train
    X_test = longitudinal_data.X_test
    y_test = longitudinal_data.y_test

    test_size = X_test.shape[0]

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    gama = GamaLongitudinalClassifier(
        features_group=features_group,
        non_longitudinal_features=non_longitudinal_features,
        feature_list_names=feature_list_names,
        random_state=data["random_state"],
        max_total_time=max_time,
        scoring=metric,
        search=search,
        n_jobs=1,
        post_processing=EnsemblePostProcessing(ensemble_size=5),
        store="nothing",
    )

    with Stopwatch() as sw:
        gama.fit(X_train, y_train)
    class_predictions = gama.predict(X_test)
    class_probabilities = gama.predict_proba(X_test)

    assert (
        60 * FIT_TIME_MARGIN > sw.elapsed_time
    ), "fit must stay within 110% of allotted time."

    assert isinstance(
        class_predictions, np.ndarray
    ), "predictions should be numpy arrays."
    assert (
        test_size,
    ) == class_predictions.shape, "predict should return (N,) shaped array."

    accuracy = accuracy_score(y_test, class_predictions)
    # Majority classifier on this split achieves 0.6293706293706294
    print(data["name"], metric, "accuracy:", accuracy)
    assert (
        data["base_accuracy"] <= accuracy
    ), "predictions should be at least as good as majority class."

    assert isinstance(
        class_probabilities, np.ndarray
    ), "probability predictions should be numpy arrays."
    assert (test_size, data["n_classes"]) == class_probabilities.shape, (
        "predict_proba should return" " (N,K) shaped array."
    )

    gama.cleanup("all")
    return gama


def test_binary_longitudinal_classification_accuracy_asha():
    """Binary classification, accuracy, numpy data, ASHA search."""
    _test_dataset_problem(
        arthritis_disease,
        "accuracy",
        search=AsynchronousSuccessiveHalving(),
        max_time=60,
    )


def test_binary_longitudinal_classification_accuracy_random_search():
    """Binary classification, accuracy, numpy data, random search."""
    _test_dataset_problem(arthritis_disease, "accuracy", search=RandomSearch())


def test_binary_longitudinal_classification_accuracy_bayesian_optimisation():
    """Binary classification, accuracy, numpy data, bayesian optimisation."""
    _test_dataset_problem(arthritis_disease, "accuracy", search=BayesianOptimisation())


def test_binary_longitudinal_classification_auprc():
    """Binary classification, auprc (probabilities), numpy data, ASYNCEA search."""
    _test_dataset_problem(arthritis_disease, "auprc")


def test_wrong_longitudinal_meta_estimators_config_space_gc():
    """Meta with wrong estimators"""
    with pytest.raises(ValueError):
        config_space.meta = {
            # "gama_system_name": "current_configuration_name",
            "dummy": "dummy",
        }
        GamaLongitudinalClassifier(
            search_space=config_space,
            features_group=[[0, 1], [2, 3]],
            non_longitudinal_features=[4, 5],
            feature_list_names=["a", "b", "c", "d", "e", "f"],
        )


def test_wrong_longitudinal_meta_data_preprocessors_config_space_gc():
    """Meta with wrong preprocessors"""
    with pytest.raises(ValueError):
        dummy_config_space = cs.ConfigurationSpace(
            meta={
                # "gama_system_name": "current_configuration_name",
                "estimators": "classifiers",
                "data_preparation": "data_preparation",
                "preprocessors": "preprocessors",
            }
        )

        dummy_classifier_config = LongitudinalClassifierConfig(dummy_config_space)
        dummy_classifier_config.setup_classifiers()

        dummy_preprocessor_config = LongitudinalDataPreparationConfig(
            dummy_config_space
        )
        dummy_preprocessor_config.setup_data_preparation()

        GamaLongitudinalClassifier(
            search_space=dummy_config_space,
            features_group=[[0, 1], [2, 3]],
            non_longitudinal_features=[4, 5],
            feature_list_names=["a", "b", "c", "d", "e", "f"],
        )


def test_wrong_longitudinal_meta_data_preparation_config_space_gc():
    """Meta with wrong preprocessors"""
    with pytest.raises(ValueError):
        dummy_config_space = cs.ConfigurationSpace(
            meta={
                # "gama_system_name": "current_configuration_name",
                "estimators": "classifiers",
                "data_preparation": "dummy",
                "preprocessors": "preprocessors",
            }
        )

        dummy_classifier_config = LongitudinalClassifierConfig(dummy_config_space)
        dummy_classifier_config.setup_classifiers()

        dummy_preprocessor_config = LongitudinalPreprocessorConfig(dummy_config_space)
        dummy_preprocessor_config.setup_preprocessors()

        GamaLongitudinalClassifier(
            search_space=dummy_config_space,
            features_group=[[0, 1], [2, 3]],
            non_longitudinal_features=[4, 5],
            feature_list_names=["a", "b", "c", "d", "e", "f"],
        )
