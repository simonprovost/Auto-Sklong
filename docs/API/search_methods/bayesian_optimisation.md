# Bayesian Optimisation

## BayesianOptimisation
[source](https://github.com/simonprovost/auto-sklong/blob/main/gama/search_methods/bayesian_optimisation.py/#L28)

```python
BayesianOptimisation(
    scenario_params: Optional[dict] = None,
    initial_design_params: Optional[dict] = None, 
    facade_params: Optional[dict] = None,
    config_to_individual_fun: Callable = config_to_individual, 
    **kwargs
)
```

Bayesian Optimisation is an advanced technique for hyperparameter tuning and pipeline optimisation. 
It efficiently searches through the hyperparameter space to find the best model configurations by using a 
probabilistic model to guide the search. The Bayesian Optimisation framework being leveraged via [`GAMA`](https://openml-labs.github.io/gama/master/index.html#) and `Auto-Sklong`,
is `SMAC3` (Sequential Model-based Algorithm Configuration), which is a state-of-the-art implementation of Bayesian Optimisation.
Available at [`SMAC3`](https://github.com/automl/SMAC3).

Next is what parameters can be passed to the `BayesianOptimisation` when being 
instantiated in the `GamaLongitudinalClassifier`'s `search` parameter.

### Parameters
- **scenario_params** (`Optional[dict]`): The Scenario is used to provide environment variables. For example, if you want to limit the optimization process by a time limit or want to specify where to save the results. By default is provided the `seed` and the `output_directory` of the `GAMA base` class. Therefore, this is an optional parameter. To modify at your own risk.
- **initial_design_params** (`Optional[dict]`): Parameters for the initial design, which dictates how the initial set of configurations is generated. By default, the initial design is random. This is an optional parameter, to modify at your own risk. How to set it up and with what can be found in the following [Python script](https://github.com/simonprovost/Auto-Sklong/blob/main/gama/utilities/smac.py).
- **facade_params** (`Optional[dict]`): Parameters for the SMAC facade, which manages the overall optimization process. Similar to `initial_design_params`, this is an optional parameter, to modify at your own risk. How to set it up and with what can be found in the following [Python script](https://github.com/simonprovost/Auto-Sklong/blob/main/gama/utilities/smac.py).
- **config_to_individual_fun** (`Callable`): Function that converts a configuration into an individual pipeline. By default, no need to modify it. However, if you introduce a brand-new [`search space`](https://simonprovost.github.io/Auto-Sklong/search_space/), you might need to modify it. Explore the code and open a new issue if you need help.
- **kwargs**: Additional parameters for custom configurations.

!!! note "SMAC3 is being used, look their documentation for more information"
    The `SMAC3` documentation can be found [here](https://automl.github.io/SMAC3/stable/). 
    It is recommended to check it out to understand the full potential of the Bayesian Optimisation framework being 
    used in `Auto-Sklong`. We simply implemented a wrapper around it to make it easier to use in [`GAMA`](https://openml-labs.github.io/gama/master/index.html#).