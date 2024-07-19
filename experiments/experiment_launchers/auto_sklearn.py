import argparse
import os
from typing import Any, Dict
from typing import List, Union

import autosklearn.classification
import numpy as np
import pandas as pd
from autosklearn.classification import AutoSklearnClassifier
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics._base import _average_binary_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target

from experiments.experiment_engine import ExperimentEngine

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "[%(levelname)s] [%(asctime)s:%(name)s] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "autosklearn.log"
        },
        "distributed_logfile": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "distributed.log"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file_handler"]
    },
    "loggers": {
        "autosklearn.metalearning": {
            "level": "DEBUG",
            "handlers": ["file_handler"]
        },
        "autosklearn.automl_common.utils.backend": {
            "level": "DEBUG",
            "handlers": ["file_handler"],
            "propagate": False
        },
        "smac.intensification.intensification.Intensifier": {
            "level": "DEBUG",
            "handlers": ["file_handler", "console"]
        },
        "smac.optimizer.local_search.LocalSearch": {
            "level": "DEBUG",
            "handlers": ["file_handler", "console"]
        },
        "smac.optimizer.smbo.SMBO": {
            "level": "DEBUG",
            "handlers": ["file_handler", "console"]
        },
        "EnsembleBuilder": {
            "level": "DEBUG",
            "handlers": ["file_handler", "console"]
        },
        "distributed": {
            "level": "DEBUG",
            "handlers": ["distributed_logfile"]
        }
    }
}


def _binary_uninterpolated_average_precision(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def auprc_score(
    y_true, y_score, average = "macro"
) -> Union[float, List[float]]:
    """Calculate the Area Under the Precision-Recall Curve (AUPRC).

    Args:
        y_true (Union[List[int], pd.Series]): Ground truth (correct) target values.
        y_score (np.ndarray): Estimated probabilities or decision function.
        average (Optional[str]): {'micro', 'macro', 'samples', 'weighted'} or None

    Returns:
        Union[float, List[float]]: Area under the precision-recall curve.

    Raises:
        ValueError: If `y_true` and `y_score` have different lengths.
        ValueError: If `y_score` contains non-numerical values.
        ValueError: If `y_true` contains non-integer values.
        ValueError: If `y_true` contains non-binary values (other than 0 and 1).

    """
    y_type = type_of_target(y_true)
    if y_score.ndim == 2:
        if y_score.shape[1] == 2:
            y_score = y_score[:, 1]  # select the scores for the positive class
        elif average == "micro":
            if y_type == "multiclass":
                y_true = label_binarize(y_true, classes=np.unique(y_true))
            return _binary_uninterpolated_average_precision(np.ravel(y_true), np.ravel(y_score))
        else:
            return _average_binary_score(_binary_uninterpolated_average_precision, y_true, y_score, average)
    return _binary_uninterpolated_average_precision(y_true, y_score)



def _reporter_auto_sklearn(system: AutoSklearnClassifier, X_test: pd.DataFrame) -> dict[str, Any]:
    """
    Reports the results of the fitted AutoSklearnClassifier AutoML system on the test data.

    Args:
        system (AutoSklearnClassifier): The fitted AutoSklearnClassifier AutoML syste,.
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

    if not system:
        raise ValueError("System's (RandomForestClassifier) not fitted yet.")
    return {
        "predictions": system.predict(X_test),
        "probability_predictions": system.predict_proba(X_test),
        "best_pipeline": {
            "data_preparation": "None (No Need To Report For Auto-Sklearn)",
            "preprocessor": "None (No Need To Report For Auto-Sklearn)",
            "classifier": "None (No Need To Report For Auto-Sklearn)"
        },
        "metric_optimised": "None (No Need To Report For Auto-Sklearn)"
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
        if not isinstance(self.args.auto_sklearn_default, bool):
            raise ValueError("Auto-Sklearn Default must be a boolean.")

    def get_auto_sklearn_hps(self) -> Dict[str, Any]:
        if self.args.scoring == "auprc":
            scoring = autosklearn.metrics.make_scorer(
                name="auprc",
                score_func=auprc_score,
                optimum=1,
                greater_is_better=True,
                needs_proba=True,
                needs_threshold=False,
            )
        elif self.args.scoring == "roc_auc":
            scoring = autosklearn.metrics.roc_auc
        else:
            raise ValueError("Scoring metric not supported. Please add it to the list.")

        if self.args.auto_sklearn_default:
            auto_sklearn_hps = {
                "ensemble_size": 50,
                "ensemble_nbest": 50,
                "max_models_on_disc": 50,
                "initial_configurations_via_metalearning": 25,
                "ensemble_class": "default",
            }
        else:
            auto_sklearn_hps = {
                "ensemble_size": 1,
                "ensemble_nbest": 1,
                "max_models_on_disc": 1,
                "initial_configurations_via_metalearning": 0,
                "ensemble_class": None,
            }

        auto_sklearn_hps["metric"] = scoring
        auto_sklearn_hps["time_left_for_this_task"] = self.args.max_total_time
        auto_sklearn_hps["per_run_time_limit"] = self.args.max_eval_time
        auto_sklearn_hps["seed"] = self.args.random_state
        auto_sklearn_hps["memory_limit"] = self.args.max_memory_mb
        auto_sklearn_hps["resampling_strategy"] = "cv"
        auto_sklearn_hps["resampling_strategy_arguments"] = {"folds": 5}
        auto_sklearn_hps["delete_tmp_folder_after_terminate"] = True
        auto_sklearn_hps["n_jobs"] = self.args.n_inner_jobs
        auto_sklearn_hps["allow_string_features"] = False
        auto_sklearn_hps["output_folder"] = "tmp_folder"
        auto_sklearn_hps["logging_config"] = logging_config

        return auto_sklearn_hps

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
                "custom_system": AutoSklearnClassifier,
                **self.get_auto_sklearn_hps(),
            },
            system_reporter=_reporter_auto_sklearn,
        )
        experiment.run_experiment()
        experiment.report_experiment()

    def default_parameters(self):
        default_parameters = {
            "max_eval_time": 30,
            "max_memory_mb": 4096,
            "max_total_time": 300,
            "n_inner_jobs": 1,
            "n_outer_splits": 5,
            # "random_state": 64,
            "random_state": 42,
            "scoring": "roc_auc",
            "shuffling": True,
            "auto_sklearn_default": True,
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
    parser.add_argument("--target_column", type=str)
    parser.add_argument("--auto_sklearn_default", nargs='?', type=bool)

    args = parser.parse_args()

    runner = Launcher(args)
    runner.default_parameters()
    runner.validate_parameters()
    runner.launch_experiment()


if __name__ == "__main__":
    main()
