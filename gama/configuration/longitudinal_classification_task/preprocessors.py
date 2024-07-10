from typing import List, Dict

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh


class LongitudinalPreprocessorConfig:
    """Manages the configuration space for preprocessors

    Focus: Longitudinal supervised learning contexts

    Parameters
    ----------
    config_space : cs.ConfigurationSpace
        The ConfigSpace object that will be used to add the preprocessors and their
        respective hyperparameters.

    """

    def __init__(
        self,
        config_space: cs.ConfigurationSpace,
    ):
        if any(
            meta_key not in config_space.meta
            for meta_key in ["data_preparation", "preprocessors"]
        ):
            raise ValueError(
                f"Expected 'data_preparation', 'preprocessors', "
                f"keys in meta of config_space. "
                f"Got {config_space.meta.keys()}"
            )
        if (
            config_space.get_hyperparameter(config_space.meta["data_preparation"])
            is None
        ):
            raise ValueError(
                f"Expected '{config_space.meta['data_preparation']}' hyperparameter "
                f"in config_space"
            )
        self.config_space = config_space
        self.preprocessors_setup_map = {
            "Dummy_To_Ignore": self.setup_dummy_to_ignore,
            "CorrelationBasedFeatureSelectionPerGroup": self.setup_cfs_per_group,
            "CorrelationBasedFeatureSelection": self.setup_cfs,
            "None": self.setup_none,
        }
        self.forbidden_clauses: List[Dict[str, List[str]]] = [
            {
                "data_preparations": ["MerWavTimePlus"],
                "preprocessors_to_forbid": ["CorrelationBasedFeatureSelection"],
            },
            {
                "data_preparations": [
                    "MerWavTimeMinus",
                    "AggrFuncMean",
                    "AggrFuncMedian",
                    # "AggrFuncMode",
                    "SepWavMajorityVoting",
                    "SepWavDecayLinearWeighting",
                    "SepWavDecayExponentialWeighting",
                    "SepWavDecayCrossValidation",
                    "SepWavStackingLogisticRegression",
                    "SepWavStackingDecisionTree",
                    "SepWavStackingRandomForest",
                ],
                "preprocessors_to_forbid": ["CorrelationBasedFeatureSelectionPerGroup"],
            },
        ]
        self.cs_preprocessors_name = config_space.meta["preprocessors"]
        self.cs_data_preparation_name = config_space.meta["data_preparation"]

    @property
    def shared_hyperparameters(self):
        return {}

    def setup_preprocessors(self):
        preprocessors_choices = list(self.preprocessors_setup_map.keys())

        if not preprocessors_choices:
            raise ValueError("No preprocessors to add to config space")

        preprocessors = csh.CategoricalHyperparameter(
            name=self.cs_preprocessors_name,
            choices=preprocessors_choices,
            default_value=preprocessors_choices[0],
        )
        self.config_space.add_hyperparameter(preprocessors)

        if self.forbidden_clauses is not None:
            if (
                self.cs_data_preparation_name
                not in self.config_space.get_hyperparameter_names()
            ):
                raise ValueError(
                    f"Data preparation hyperparameter "
                    f"{self.cs_data_preparation_name} not found in config space"
                )
            data_preparation = self.config_space.get_hyperparameter(
                self.cs_data_preparation_name
            )
            data_preparation_choices = list(data_preparation.choices)
            forbidden_clauses = [
                cs.ForbiddenAndConjunction(
                    cs.ForbiddenInClause(
                        preprocessors, clause["preprocessors_to_forbid"]
                    ),
                    cs.ForbiddenInClause(data_preparation, clause["data_preparations"]),
                )
                for clause in self.forbidden_clauses
                if all(
                    data_preparation_name in data_preparation_choices
                    for data_preparation_name in clause["data_preparations"]
                )
            ]
            self.config_space.add_forbidden_clauses(forbidden_clauses)

        for preprocessor_name in preprocessors_choices:
            if setup_func := self.preprocessors_setup_map.get(preprocessor_name):
                setup_func(preprocessors)

    def _add_hyperparameters_and_equals_conditions(
        self, local_vars: dict, preprocessor_name: str
    ):
        if "preprocessors" not in local_vars or not isinstance(
            local_vars["preprocessors"], csh.CategoricalHyperparameter
        ):
            raise ValueError(
                "Expected 'preprocessors' key with a CategoricalHyperparameter in local"
                "vars"
            )

        hyperparameters_to_add = [
            hyperparameter
            for hyperparameter in local_vars.values()
            if isinstance(hyperparameter, csh.Hyperparameter)
            and hyperparameter != local_vars["preprocessors"]
        ]

        conditions_to_add = [
            cs.EqualsCondition(
                hyperparameter, local_vars["preprocessors"], preprocessor_name
            )
            for hyperparameter in hyperparameters_to_add
        ]

        self.config_space.add_hyperparameters(hyperparameters_to_add)
        self.config_space.add_conditions(conditions_to_add)

    def setup_dummy_to_ignore(self, classifiers: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_cfs(self, preprocessors: csh.CategoricalHyperparameter):
        pass

    def setup_cfs_per_group(self, preprocessors: csh.CategoricalHyperparameter):
        version = csh.CategoricalHyperparameter(
            name="version__CorrelationBasedFeatureSelectionPerGroup",
            choices=[1, 2],
            default_value=1,
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "CorrelationBasedFeatureSelectionPerGroup"
        )

    def setup_none(self, preprocessors: csh.CategoricalHyperparameter):
        pass
