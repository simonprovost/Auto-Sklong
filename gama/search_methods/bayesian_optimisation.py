import logging
from asyncio import Future
from typing import List, Optional, Tuple, Callable, Any

import numpy as np
import pandas as pd
from smac.facade import AbstractFacade
from smac.runhistory import StatusType
from smac.runhistory.dataclasses import TrialValue, TrialInfo

from gama.genetic_programming.components import Individual
from gama.genetic_programming.operator_set import OperatorSet
from gama.search_methods.base_search import (
    BaseSearch,
    _check_base_search_hyperparameters,
)
from gama.utilities.generic.async_evaluator import AsyncEvaluator
from gama.utilities.smac import (
    get_smac,
    validate_future_valid,
    config_to_individual,
)
import ConfigSpace as cs

log = logging.getLogger(__name__)


class BayesianOptimisation(BaseSearch):
    """Perform Bayesian Optimisation over all possible pipelines."""

    def __init__(
        self,
        scenario_params: Optional[dict] = None,
        initial_design_params: Optional[dict] = None,
        facade_params: Optional[dict] = None,
        config_to_individual_fun: Callable = config_to_individual,
        **kwargs,
    ):
        super().__init__()
        self._scenario_params = scenario_params
        self._initial_design_params = initial_design_params
        self._facade_params = facade_params
        self._smac: Optional[AbstractFacade] = None
        self._config_to_individual_fun = config_to_individual_fun
        self._output_directory = None
        self.random_state = None

    def dynamic_defaults(
        self, x: pd.DataFrame, y: pd.DataFrame, time_limit: float
    ) -> None:
        def set_gama_smac_default_scenario(gama_param: str, smac_param: Any) -> None:
            if (
                self._scenario_params
                and gama_param not in self._scenario_params
                and smac_param
            ):
                self._scenario_params[gama_param] = smac_param
            elif not self._scenario_params and smac_param:
                self._scenario_params = {gama_param: smac_param}

        for gama_param, smac_param in {
            "output_directory": self._output_directory,
            "seed": self.random_state,
        }.items():
            set_gama_smac_default_scenario(gama_param, smac_param)

    def search(
        self, operations: OperatorSet, start_candidates: List[Individual]
    ) -> None:
        if (
            operations.get_search_space is None
            or (config_space := operations.get_search_space()) is None
        ):
            raise ValueError("BayesianOptimisation: Config space is None in 'search'.")
        self.smac = get_smac(
            configSpace=config_space,
            scenario_params=self._scenario_params,
            initial_design_params=self._initial_design_params,
            facade_params=self._facade_params,
        )

        self.output, self.smac = bayesian_optimisation(
            config_to_individual_fun=self._config_to_individual_fun,
            operations=operations,
            output=self.output,
            start_candidates=start_candidates,
            smac=self.smac,
        )


def bayesian_optimisation(
    config_to_individual_fun: Callable,
    operations: OperatorSet,
    output: List[Individual],
    start_candidates: List[Individual],
    smac: AbstractFacade,
    max_evaluations: Optional[int] = None,
    max_attempts: int = 100000,
) -> Tuple[List[Individual], AbstractFacade]:
    """Perform Bayesian Optimisation over all possible pipelines."""

    _check_base_search_hyperparameters(operations, output, start_candidates)
    with AsyncEvaluator() as async_:

        def smac_ask_and_submit() -> TrialInfo:
            """Ask SMAC for a configuration to turn into an individual for evaluation."""

            def _ask() -> TrialInfo:
                """Ask SMAC for a configuration"""
                if (
                    smac is None
                    or not (info_cand := smac.ask())
                    or info_cand.seed is None
                ):
                    raise ValueError(
                        "BayesianOptimisation: SMAC ask failed in 'smac_ask_and_submit'. "
                        "SMAC object or smac.ask().seed should not be None."
                    )
                return info_cand

            def ask_and_ignore_dummy_techniques(
                technique_name_to_ignore: str = "Dummy_To_Ignore",
            ) -> cs.Configuration:
                """ConfigSpace do not allow adding forbidden clauses on the default choice of a hyperparameter.

                Dummy_To_Ignore becomes default and is ignored / found an alternative when picked by sample_configuration.
                Nonetheless, for better maintainability, "Dummy_To_Ignore" can be changed, for example in the future it could be
                in the meta of the config so that it can be changed in one place by the user, i.e in /configuratins.
                """
                new_candidate = _ask()
                config = new_candidate.config

                is_preprocessor_or_estimator_to_ignore = (
                    technique_name_to_ignore in config.values()
                )
                while is_preprocessor_or_estimator_to_ignore:
                    temp_candidate = _ask()
                    temp_config = temp_candidate.config

                    is_preprocessor_or_estimator_to_ignore = (
                        technique_name_to_ignore in temp_config.values()
                    )
                    if not is_preprocessor_or_estimator_to_ignore:
                        return temp_candidate
                return new_candidate

            candidate = ask_and_ignore_dummy_techniques()
            attempts = 0
            while True:
                if not (
                    individual := config_to_individual_fun(candidate.config, operations)
                ):
                    raise ValueError(
                        "BayesianOptimisation: Conversion of SMAC config to GAMA individual"
                        "failed in 'smac_ask_and_submit'."
                    )
                if operations.is_evaluated is None:
                    raise ValueError(
                        "BayesianOptimisation: Operations.is_evaluated is None in "
                        "'smac_ask_and_submit'."
                    )
                if not operations.is_evaluated(individual):
                    async_.submit(operations.evaluate, individual)
                    break

                attempts += 1
                if attempts >= max_attempts:
                    raise ValueError(
                        "Maximum attempts reached while trying to generate a"
                        "unique individual."
                    )
                candidate = ask_and_ignore_dummy_techniques()

            return candidate

        @validate_future_valid
        def smac_handle_and_tell(future: Future, info: TrialInfo) -> None:
            """Handle the result of an evaluation and update SMAC with it."""
            if smac is None:
                raise ValueError("BayesianOptimisation: SMAC object is None.")

            individual = future.result.individual  # type: ignore

            if (
                fitness_values := individual.fitness.values
            ) and np.inf in fitness_values:
                log.warning(
                    f"BayesianOptimisation: The pipeline crashed during evaluation. "
                    f"The cost is set to -1. Individual: {individual}"
                    f"Fitness Values: {fitness_values}"
                )
                cost = -1
            else:
                cost = 1 - fitness_values[0]

            start_time = individual.fitness.start_time.timestamp()
            trial_value = TrialValue(
                cost=cost,
                time=individual.fitness.wallclock_time,
                status=(
                    StatusType.CRASHED
                    if hasattr(individual.fitness, "error")
                    else StatusType.SUCCESS
                ),
                starttime=start_time,
                endtime=start_time + individual.fitness.process_time,
            )

            output.append(individual)
            smac.tell(info, trial_value)

        if start_candidates:
            log.warning(
                "BayesianOptimisation: No start candidates are evaluated as of the "
                "current version. Further will be implemented to "
                "support both start candidates and SMAC warms-tart using GAMA meta"
                "learning, in future versions."
            )

        info = smac_ask_and_submit()

        while (max_evaluations is None) or (len(output) < max_evaluations):
            if (future := operations.wait_next(async_)).result is not None:
                smac_handle_and_tell(future, info)
            info = smac_ask_and_submit()

    return output, smac
