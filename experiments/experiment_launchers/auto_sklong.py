import argparse
import logging
import os
from typing import Any

import pandas as pd
from sklearn.utils import Bunch

from experiments.experiment_engine import ExperimentEngine
from gama.GamaLongitudinalClassifier import GamaLongitudinalClassifier
from gama.search_methods import RandomSearch, AsyncEA, AsynchronousSuccessiveHalving
from gama.search_methods.bayesian_optimisation import BayesianOptimisation


def _reporter_auto_sklong(system: GamaLongitudinalClassifier, X_test: pd.DataFrame) -> dict[str, Any]:
    """
    Reports the results of the fitted AutoML system Auto-Sklong (default system) on the test data.

    Args:
        system (GamaLongitudinalClassifier): The fitted AutoML system.
        X_test (pd.DataFrame): The test dataset.

    Returns:
        dict[str, Any]: A dictionary containing the following keys:
            - "predictions": Predictions made by the system.
            - "probability_predictions": Probability predictions made by the system.
            - "best_pipeline": A dictionary with the names of the techniques used in the best pipeline for data preparation, preprocessing, and classification.
            - "metric_optimised": The name of the metric that was optimized during training.

    Raises:
        ValueError: If the system is not fitted or if required pipeline steps are not found.
    """

    def _get_technique(named_steps: Bunch, step_name: str) -> str:
        if step_name == "data_preparation":
            if len(named_steps) == 3 and "2" in named_steps:
                if hasattr(named_steps["2"], "__class__"):
                    return named_steps["2"].__class__.__name__
                return named_steps["2"]
            elif len(named_steps) == 2 and "1" in named_steps:
                if hasattr(named_steps["1"], "__class__"):
                    return named_steps["1"].__class__.__name__
                return named_steps["1"]
            raise ValueError("Data preparation step not found.")
        if step_name == "preprocessor":
            if len(named_steps) == 3 and "1" in named_steps:
                if hasattr(named_steps["1"], "__class__"):
                    return named_steps["1"].__class__.__name__
                return named_steps["1"]
            else:
                return "None"
        if step_name == "classifier":
            if "0" in named_steps:
                if hasattr(named_steps["0"], "__class__"):
                    return named_steps["0"].__class__.__name__
                return named_steps["0"]
            raise ValueError("Classifier step not found.")
        raise ValueError("Step name not found.")

    if not system:
        raise ValueError("System's not fitted yet.")
    return {
        "predictions": system.predict(X_test),
        "probability_predictions": system.predict_proba(X_test),
        "best_pipeline": {
            "data_preparation": _get_technique(system.model.named_steps, "data_preparation"),
            "preprocessor": _get_technique(system.model.named_steps, "preprocessor"),
            "classifier": _get_technique(system.model.named_steps, "classifier")
        },
        "metric_optimised": system._metrics[0].name,
    }


class Launcher:
    def __init__(self, args):
        self.args = args

    def validate_parameters(self):
        if not os.path.exists(self.args.dataset_path):
            raise ValueError("Dataset path does not exist.")
        if not isinstance(self.args.export_name, str):
            raise ValueError("Export name must be a string.")
        if not isinstance(self.args.fold_number, int) or self.args.fold_number <= 0:
            raise ValueError("Fold number must be a positive integer.")
        if not isinstance(self.args.max_eval_time, int) or self.args.max_eval_time <= 0:
            raise ValueError("Max eval time must be a positive integer.")
        if not isinstance(self.args.max_memory_mb, int) or self.args.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be a positive integer.")
        if not isinstance(self.args.max_total_time, int) or self.args.max_total_time <= 0:
            raise ValueError("Max total time must be a positive integer.")
        if not isinstance(self.args.n_inner_jobs, int) or self.args.n_inner_jobs <= 0:
            raise ValueError("n_inner_jobs must be a positive integer.")
        if not isinstance(self.args.n_outer_splits, int) or self.args.n_outer_splits <= 0:
            raise ValueError("n_outer_splits must be a positive integer.")
        if not isinstance(self.args.random_state, int) or self.args.random_state <= 0:
            raise ValueError("Random state must be a positive integer.")
        if not isinstance(self.args.scoring, str):
            raise ValueError("Scoring must be a string.")
        if not isinstance(self.args.shuffling, bool):
            raise ValueError("Shuffling must be a boolean.")
        if not isinstance(self.args.store, str):
            raise ValueError("Store must be a string.")
        if not isinstance(self.args.verbosity, int):
            raise ValueError("Verbosity must be an integer.")
        if not isinstance(self.args.search_algorithm, str) or self.args.search_algorithm not in [
            "bayesian_optimisation",
            "random_search",
            "evolutionary_algorithm",
            "ASHA"
        ]:
            raise ValueError(
                "Search algorithm must be either "
                "'bayesian_optimisation', 'random_search', 'evolutionary_algorithm', 'ASHA'. ")
        else:
            if self.args.search_algorithm == "random_search":
                self.args.search_algorithm = RandomSearch()
            elif self.args.search_algorithm == "bayesian_optimisation":
                self.args.search_algorithm = BayesianOptimisation()
            elif self.args.search_algorithm == "evolutionary_algorithm":
                self.args.search_algorithm = AsyncEA()
            elif self.args.search_algorithm == "ASHA":
                self.args.search_algorithm = AsynchronousSuccessiveHalving()

    def launch_experiment(self):
        experiment = ExperimentEngine(
            output_path=self.args.export_name,
            fold_number=self.args.fold_number,
            setup_data_parameters={
                "dataset_file_path": self.args.dataset_path,
                "random_state": self.args.random_state,
                "target_column": self.args.target_column,
                "shuffling": self.args.shuffling,
                "n_outer_splits": self.args.n_outer_splits,
            },
            system_hyperparameters={
                "max_total_time": self.args.max_total_time,
                "max_eval_time": self.args.max_eval_time,
                "n_inner_jobs": self.args.n_inner_jobs,
                "scoring": self.args.scoring,
                "search": self.args.search_algorithm,
                "max_memory_mb": self.args.max_memory_mb,
                "store": self.args.store,
                "verbosity": self.args.verbosity,
            },
            system_reporter=_reporter_auto_sklong,
        )
        experiment.run_experiment()
        experiment.report_experiment()

    def default_parameters(self):
        default_parameters = {
            "max_eval_time": 10,
            "max_memory_mb": 4096,
            "max_total_time": 60,
            "n_inner_jobs": 1,
            "n_outer_splits": 5,
            # "random_state": 64,
            "random_state": 42,
            "scoring": "roc_auc",
            "search_algorithm": "random_search",
            "shuffling": True,
            "store": "all",
            "verbosity": logging.DEBUG,
        }
        for key, value in default_parameters.items():
            if getattr(self.args, key) is None:
                setattr(self.args, key, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--export_name", type=str)
    parser.add_argument("--fold_number", type=int)
    parser.add_argument("--max_eval_time", nargs='?', type=int)
    parser.add_argument("--max_memory_mb", nargs='?', type=int)
    parser.add_argument("--max_total_time", nargs='?', type=int)
    parser.add_argument("--n_inner_jobs", nargs='?', type=int)
    parser.add_argument("--n_outer_splits", nargs='?', type=int)
    parser.add_argument("--random_state", nargs='?', type=int)
    parser.add_argument("--scoring", nargs='?', type=str)
    parser.add_argument("--search_algorithm", nargs='?', type=str)
    parser.add_argument("--shuffling", nargs='?', type=bool)
    parser.add_argument("--store", nargs='?', type=str)
    parser.add_argument("--target_column", type=str)
    parser.add_argument("--verbosity", nargs='?', type=int)

    args = parser.parse_args()

    runner = Launcher(args)
    runner.default_parameters()
    runner.validate_parameters()
    runner.launch_experiment()


if __name__ == "__main__":
    main()
