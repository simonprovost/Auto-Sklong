import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh


class LongitudinalDataPreparationConfig:
    """Manages the configuration space for Data Preparation

    Focus: longitudinal supervised learning contexts

    Parameters
    ----------
    config_space : cs.ConfigurationSpace
        The ConfigSpace object that defines the hyperparameters and their ranges for
        the classifiers.

    """

    def __init__(
        self,
        config_space: cs.ConfigurationSpace,
    ):
        if "data_preparation" not in config_space.meta:
            raise ValueError(
                f"Expected 'data_preparation', "
                f"key in meta of config_space. "
                f"Got {config_space.meta.keys()}"
            )
        self.config_space = config_space
        self.data_preparation_setup_map = {
            "Dummy_To_Ignore": self.setup_dummy_to_ignore,
            "MerWavTimePlus": self.setup_merwav_time_plus,
            "MerWavTimeMinus": self.setup_merwav_time_minus,
            "AggrFuncMean": self.setup_aggr_func_mean,
            "AggrFuncMedian": self.setup_aggr_func_median,
            # "AggrFuncMode": self.setup_aggr_func_mode,
            "SepWavMajorityVoting": self.setup_sep_wav_majority_voting,
            "SepWavDecayLinearWeighting": self.setup_sep_wav_decay_linear_weighting,
            "SepWavDecayExponentialWeighting": self.setup_sep_wav_decay_exponential_weighting,  # noqa: E501
            "SepWavDecayCrossValidation": self.setup_sep_wav_decay_cross_validation,
            "SepWavStackingLogisticRegression": self.setup_sep_wav_stacking_logistic_regression,  # noqa: E501
            "SepWavStackingDecisionTree": self.setup_sep_wav_stacking_decision_tree,
            "SepWavStackingRandomForest": self.setup_sep_wav_stacking_random_forest,
        }
        self.cs_data_preparation_name = self.config_space.meta["data_preparation"]

    def setup_data_preparation(self):
        data_preparation_choices = list(self.data_preparation_setup_map.keys())

        if not data_preparation_choices:
            raise ValueError("No data preparation to add to config space")

        weights = [
            (
                0.5
                if choice == "MerWavTimePlus"
                else 0.5 / (len(data_preparation_choices) - 1)
            )
            for choice in data_preparation_choices
        ]

        data_preparation = csh.CategoricalHyperparameter(
            name=self.cs_data_preparation_name,
            choices=data_preparation_choices,
            weights=weights,
            default_value=data_preparation_choices[0],
        )

        self.config_space.add_hyperparameter(data_preparation)

        for data_preparation_name in data_preparation_choices:
            if setup_func := self.data_preparation_setup_map.get(data_preparation_name):
                setup_func(data_preparation)

    def setup_dummy_to_ignore(self, classifiers: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_merwav_time_plus(self, data_preparation: csh.CategoricalHyperparameter):
        pass

    def setup_merwav_time_minus(self, data_preparation: csh.CategoricalHyperparameter):
        pass

    def setup_aggr_func_mean(self, data_preparation: csh.CategoricalHyperparameter):
        pass

    def setup_aggr_func_median(self, data_preparation: csh.CategoricalHyperparameter):
        pass

    def setup_aggr_func_mode(self, data_preparation: csh.CategoricalHyperparameter):
        pass

    def setup_sep_wav_majority_voting(
        self, data_preparation: csh.CategoricalHyperparameter
    ):
        pass

    def setup_sep_wav_decay_linear_weighting(
        self, data_preparation: csh.CategoricalHyperparameter
    ):
        pass

    def setup_sep_wav_decay_exponential_weighting(
        self, data_preparation: csh.CategoricalHyperparameter
    ):
        pass

    def setup_sep_wav_decay_cross_validation(
        self, data_preparation: csh.CategoricalHyperparameter
    ):
        pass

    def setup_sep_wav_stacking_logistic_regression(
        self, data_preparation: csh.CategoricalHyperparameter
    ):
        pass

    def setup_sep_wav_stacking_decision_tree(
        self, data_preparation: csh.CategoricalHyperparameter
    ):
        pass

    def setup_sep_wav_stacking_random_forest(
        self, data_preparation: csh.CategoricalHyperparameter
    ):
        pass
