from math import sqrt, log2
from typing import List, Union

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh

from gama.configuration.longitudinal_classification_task.classifiers import (
    LongitudinalClassifierConfig,
)
from gama.configuration.longitudinal_classification_task.data_preparation import (
    LongitudinalDataPreparationConfig,
)
from gama.configuration.longitudinal_classification_task.preprocessors import (
    LongitudinalPreprocessorConfig,
)
from gama.genetic_programming.longitudinal.constraints_longitudinal import (
    LongitudinalSearchSpaceConstraint,
)

# Classifiers & Preprocessors 🚀

LongitudinalConstraints = LongitudinalSearchSpaceConstraint(
    constraints={
        "MerWavTimePlus": {
            "data_preparation": ["MerWavTimePlus"],
            "preprocessors": ["CorrelationBasedFeatureSelection", "None"],
            "estimators": [
                "DecisionTreeClassifier",
                "RandomForestClassifier",
                "ExtraTreesClassifier",
                "KNeighborsClassifier",
                "LinearSVC",
                "DeepForestClassifier",
                "GradientBoostingClassifier",
            ],
        },
        "Default": {
            "data_preparation": ["MerWavTimePlus"],
            "preprocessors": ["CorrelationBasedFeatureSelectionPerGroup", "None"],
            "estimators": [
                "NestedTreesClassifier",
                "LexicoRandomForestClassifier",
                "LexicoDecisionTreeClassifier",
                "LexicoDeepForestClassifier",
                "LexicoGradientBoostingClassifier",
            ],
        },
    }
)

config_space = cs.ConfigurationSpace(
    meta={
        # "gama_system_name": "current_configuration_name",
        "data_preparation": "data_preparation",
        "preprocessors": "preprocessors",
        "estimators": "classifiers",
    }
)

data_preparation_config = LongitudinalDataPreparationConfig(config_space)
data_preparation_config.setup_data_preparation()

classifier_config = LongitudinalClassifierConfig(config_space)
classifier_config.setup_classifiers()

preprocessor_config = LongitudinalPreprocessorConfig(config_space)
preprocessor_config.setup_preprocessors()


# Extra Dynamic Hyperparameters 🚀


def max_features_based_on_dataset_features_length(
    search_space: cs.ConfigurationSpace, features: List[str], algorithm_name: str
) -> cs.ConfigurationSpace:
    def compute_max_features_float(value: Union[int, float]) -> Union[int, float]:
        return value / len(features)

    # Check if the CategoryHyperparameter already exists
    # if so do not add any new hyperparameters
    if f"max_features__{algorithm_name}" in search_space.get_hyperparameters_dict():
        return search_space

    max_features = csh.CategoricalHyperparameter(
        name=f"max_features__{algorithm_name}",
        choices=[
            compute_max_features_float(sqrt(len(features))),
            compute_max_features_float(sqrt(len(features)) * 2),
            compute_max_features_float(sqrt(len(features)) / 2),
            compute_max_features_float(log2(len(features)) + 1),
        ],
    )
    search_space.add_hyperparameter(max_features)
    search_space.add_condition(
        cs.EqualsCondition(
            max_features,
            search_space.get_hyperparameter("classifiers"),
            algorithm_name,
        )
    )
    return search_space


extra_hyperparameters_based_on_dataset_features_length = [
    lambda search_space, features, algorithm_name=algorithm_name: max_features_based_on_dataset_features_length(  # noqa: E501
        search_space, features, algorithm_name
    )
    for algorithm_name in [
        "LexicoRandomForestClassifier",
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "LexicoGradientBoostingClassifier",
    ]
]
