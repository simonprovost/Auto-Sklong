import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh

class LongitudinalResamplingConfig:
    def __init__(
        self,
        config_space: cs.ConfigurationSpace,
    ):
        if "resampling" not in config_space.meta:
            raise ValueError(
                "Expected 'resampling' key in meta of config_space."
            )
        self.config_space = config_space
        self.resampling_setup_map = {
            "NoResampling": self.setup_no_resampling,
            "SampleWeight": self.setup_sample_weight,
            # "RandomUnderSampler": self.setup_random_under_sampler, # Not of interest in current exploration, maybe later.,
        }
        self.cs_resampling_name = config_space.meta["resampling"]

    def setup_resampling(self):
        resampling_choices = list(self.resampling_setup_map.keys())

        if not resampling_choices:
            raise ValueError("No resampling methods to add to config space")

        resampling = csh.CategoricalHyperparameter(
            name=self.cs_resampling_name,
            choices=resampling_choices,
            default_value=resampling_choices[0],
        )
        self.config_space.add_hyperparameter(resampling)

        for resampling_name in resampling_choices:
            if setup_func := self.resampling_setup_map.get(resampling_name):
                setup_func(resampling)

    def _add_hyperparameters_and_equals_conditions(
        self, local_vars: dict, resampling_name: str
    ):
        if "resampling" not in local_vars or not isinstance(
            local_vars["resampling"], csh.CategoricalHyperparameter
        ):
            raise ValueError(
                "Expected 'resampling' key with a CategoricalHyperparameter in local vars"
            )

        hyperparameters_to_add = [
            hyperparameter
            for hyperparameter in local_vars.values()
            if isinstance(hyperparameter, csh.Hyperparameter)
            and hyperparameter != local_vars["resampling"]
        ]

        conditions_to_add = [
            cs.EqualsCondition(
                hyperparameter, local_vars["resampling"], resampling_name
            )
            for hyperparameter in hyperparameters_to_add
        ]

        self.config_space.add_hyperparameters(hyperparameters_to_add)
        self.config_space.add_conditions(conditions_to_add)

    def setup_no_resampling(self, resampling: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_sample_weight(self, resampling: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_random_under_sampler(self, resampling: csh.CategoricalHyperparameter):
        sampling_strategy = csh.CategoricalHyperparameter(
            "sampling_strategy__RandomUnderSampler",
            ["majority", "not minority", "not majority", "all", "auto"],
            default_value="auto"
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "RandomUnderSampler")