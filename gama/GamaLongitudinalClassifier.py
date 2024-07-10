import inspect
from functools import partial
from typing import Union, Optional, List, Callable
import logging

import numpy as np
import pandas as pd
from ConfigSpace import ForbiddenEqualsClause
from gama.search_methods.base_search import BaseSearch

from gama.genetic_programming.compilers.scikitlearn import compile_individual


from gama.genetic_programming.operator_set import OperatorSet
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import ConfigSpace as cs

from gama.configuration.longitudinal_classification import (
    config_space as clf_config,
    LongitudinalConstraints,
    extra_hyperparameters_based_on_dataset_features_length,
)
from gama.data_loading import X_y_from_file
from gama.utilities.metrics import scoring_to_metric, Metric
from .GamaPipeline import GamaPipelineType
from .gama import Gama
from .genetic_programming.longitudinal.crossover_longitudinal import (
    random_longitudinal_crossover,
)
from .genetic_programming.longitudinal.mutation_longitudinal import (
    random_longitudinal_valid_mutation_in_place,
)
from .genetic_programming.longitudinal.operations_longitudinal import (
    create_longitudinal_random_expression,
)
from .genetic_programming.selection import eliminate_from_pareto
from .genetic_programming.longitudinal.selection_longitudinal import (
    create_longitudinal_from_population,
)
from .search_methods import AsyncEA
from .search_methods.bayesian_optimisation import BayesianOptimisation
from .utilities.config_space import get_longitudinal_estimator_by_name
from .utilities.smac import config_to_longitudinal_individual

# Avoid stopit from logging warnings every time a pipeline evaluation times out
logging.getLogger("stopit").setLevel(logging.ERROR)
log = logging.getLogger(__name__)


class GamaLongitudinalClassifier(Gama):
    """Gama with adaptations for Longitudinal classification."""

    def __init__(
        self,
        features_group: List[List[Union[int, str]]],
        non_longitudinal_features: List[Union[int, str]],
        feature_list_names: List[str],
        update_feature_groups_callback: Union[Callable, str] = "default",
        search_space: Optional[cs.ConfigurationSpace] = None,
        scoring: Metric = "roc_auc",  # type: ignore
        max_pipeline_length: Optional[int] = None,
        search: Optional[BaseSearch] = None,
        *args,
        **kwargs,
    ):
        if not search_space:
            # Do this to avoid the whole dictionary being included in the documentation.
            search_space = clf_config

        self._metrics = scoring_to_metric(scoring)

        search_space = self._search_space_check(search_space)

        if extra_hyperparameters_based_on_dataset_features_length is not None:
            for (
                extra_hyperparameter
            ) in extra_hyperparameters_based_on_dataset_features_length:
                search_space = extra_hyperparameter(search_space, feature_list_names)

        self._label_encoder = None

        if isinstance(search, BayesianOptimisation):
            search._config_to_individual_fun = config_to_longitudinal_individual
        elif isinstance(search, AsyncEA) and LongitudinalConstraints is None:
            raise ValueError(
                "LongitudinalConstraints is not available. For Longitudinal tasks, "
                "please make sure LongitudinalConstraints is available."
            )

        super().__init__(
            *args,
            search_space=search_space,
            scoring=scoring,
            search=search,
            gama_pipeline_type=GamaPipelineType.ScikitLongitudinal,
            **kwargs,
        )  # type: ignore

        self._features_group = features_group
        self._non_longitudinal_features = non_longitudinal_features
        self._update_feature_groups_callback = update_feature_groups_callback
        self._feature_list_names = feature_list_names

        if (
            "preprocessors" in self.search_space.meta
            and self.search_space.meta["preprocessors"]
            not in self.search_space.get_hyperparameter_names()
        ) or ("preprocessors" not in self.search_space.meta):
            if max_pipeline_length is None:
                log.info(
                    "Setting `max_pipeline_length` to 2 "
                    "because there are no preprocessing steps in the search space."
                )
                max_pipeline_length = 2
            elif max_pipeline_length > 2:
                raise ValueError(
                    f"`max_pipeline_length` can't be {max_pipeline_length} "
                    "because there are no preprocessing steps in the search space."
                )
        else:
            if max_pipeline_length is None:
                max_pipeline_length = 3
            elif max_pipeline_length < 3:
                log.warning(
                    f"[CAUTIOUS] Setting `max_pipeline_length` to "
                    f"{max_pipeline_length} can be problematic because there most "
                    f"probably are preprocessing steps available in the search space."
                )
            elif max_pipeline_length > 3:
                log.warning(
                    f"[CAUTIOUS] Setting `max_pipeline_length` to "
                    f"{max_pipeline_length} can be problematic because there most "
                    f"probably are only estimators/preprocessing steps available in "
                    f"the search space."
                )
        max_start_length = 3 if max_pipeline_length is None else max_pipeline_length
        self._operator_set = OperatorSet(
            mutate=partial(  # type: ignore #https://github.com/python/mypy/issues/1484
                random_longitudinal_valid_mutation_in_place,
                config_space=self.search_space,
                max_length=max_pipeline_length,
            ),
            mate=partial(random_longitudinal_crossover, max_length=max_pipeline_length),
            create_from_population=partial(
                create_longitudinal_from_population, cxpb=0.2, mutpb=0.8
            ),
            create_new=partial(
                create_longitudinal_random_expression,
                config_space=self.search_space,
                max_length=max_start_length,
            ),
            compile_=partial(
                compile_individual,
                gama_pipeline_type=self._gama_pipeline_type,
                features_group=self._features_group,
                non_longitudinal_features=self._non_longitudinal_features,
                feature_list_names=self._feature_list_names,
                update_feature_groups_callback=self._update_feature_groups_callback,
            ),
            eliminate=eliminate_from_pareto,
            evaluate_callback=self._on_evaluation_completed,
            completed_evaluations=self._evaluation_library.lookup,
            is_evaluated=self._evaluation_library.is_evaluated,
            get_search_space=lambda: self.search_space,
        )

    def _search_space_check(
        self,
        search_space: cs.ConfigurationSpace,
    ) -> cs.ConfigurationSpace:
        """Check if the search space is valid for classification."""

        # Check if the search space contains a classifier hyperparameter.
        if (
            "estimators" not in search_space.meta
            or (
                search_space.meta["estimators"]
                not in search_space.get_hyperparameters_dict()
            )
            or not isinstance(
                search_space.get_hyperparameter(search_space.meta["estimators"]),
                cs.CategoricalHyperparameter,
            )
        ):
            raise ValueError(
                "The search space must include a hyperparameter for the classifiers "
                "that is a CategoricalHyperparameter with choices for all desired "
                "classifiers. Please double-check the spelling of the name, and review "
                "the `meta` object in the search space configuration located at "
                "`configurations/classification.py`. The `meta` object should contain "
                "a key `estimators` with a value that is the name of the hyperparameter"
                " that contains the classifier choices."
            )

        # Check if the search space contains a preprocessor hyperparameter
        # if it is specified in the meta.
        if (
            "preprocessors" in search_space.meta
            and (
                search_space.meta["preprocessors"]
                not in search_space.get_hyperparameters_dict()
            )
            or "preprocessors" in search_space.meta
            and not isinstance(
                search_space.get_hyperparameter(search_space.meta["preprocessors"]),
                cs.CategoricalHyperparameter,
            )
        ):
            raise ValueError(
                "The search space must include a hyperparameter for the preprocessors "
                "that is a CategoricalHyperparameter with choices for all desired "
                "preprocessors. Please double-check the spelling of the name, and "
                "review the `meta` object in the search space configuration located at "
                "`configurations/classification.py`. The `meta` object should contain "
                "a key `preprocessors` with a value that is the name of the "
                "hyperparameter that contains the preprocessor choices. "
            )

        # Check if the search space contains a data_preparation hyperparameter
        # if it is specified in the meta.
        if (
            "data_preparation" in search_space.meta
            and (
                search_space.meta["data_preparation"]
                not in search_space.get_hyperparameters_dict()
            )
            or "data_preparation" in search_space.meta
            and not isinstance(
                search_space.get_hyperparameter(search_space.meta["data_preparation"]),
                cs.CategoricalHyperparameter,
            )
        ):
            raise ValueError(
                "The search space must include a hyperparameter for the data "
                "preparation that is a CategoricalHyperparameter with choices for all "
                "desired data preparations. Please double-check the spelling of the "
                "name, and review the `meta` object in the search space configuration "
                "located at `configurations/classification.py`. The `meta` object "
                "should contain a key `data_preparation` with a value that is the name"
                " of the hyperparameter that contains the data preparation choices. "
            )

        # Check if the search space contains only classifiers that have predict_proba
        # if the scoring requires probabilities.
        if any(metric.requires_probabilities for metric in self._metrics):
            # we don't want classifiers that do not have `predict_proba`,
            # because then we have to start doing one hot encodings of predictions etc.
            no_proba_clfs = []
            for classifier in search_space.get_hyperparameter(
                search_space.meta["estimators"]
            ).choices:
                if classifier == "Dummy_To_Ignore":
                    continue
                estimator = get_longitudinal_estimator_by_name(classifier)
                if (
                    estimator is not None
                    and issubclass(estimator, ClassifierMixin)
                    and not hasattr(estimator(), "predict_proba")
                ):
                    no_proba_clfs.append(classifier)

            log.info(
                f"The following classifiers do not have a predict_proba method "
                f"and will be excluded from the search space: {no_proba_clfs}"
            )
            search_space.add_forbidden_clauses(
                [
                    ForbiddenEqualsClause(
                        search_space.get_hyperparameter(
                            search_space.meta["estimators"]
                        ),
                        classifier,
                    )
                    for classifier in no_proba_clfs
                    if classifier
                ]
            )

        return search_space

    def _predict(self, x: pd.DataFrame):
        """Predict the target for input X.

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe with the same number of columns as the input to `fit`.

        Returns
        -------
        numpy.ndarray
            Array with predictions of shape (N,) where N is len(X).
        """
        y = self.model.predict(x)  # type: ignore
        # Decode the predicted labels - necessary only if ensemble is not used.
        if y[0] not in list(self._label_encoder.classes_):  # type: ignore
            y = self._label_encoder.inverse_transform(y)  # type: ignore
        return y

    def _predict_proba(self, x: pd.DataFrame):
        """Predict the class probabilities for input x.

        Predict target for x, using the best found pipeline(s) during the `fit` call.

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe with the same number of columns as the input to `fit`.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, K) with class probabilities where N is len(x),
             and K is the number of class labels found in `y` of `fit`.
        """
        return self.model.predict_proba(x)  # type: ignore

    def predict_proba(self, x: Union[pd.DataFrame, np.ndarray]):
        """Predict the class probabilities for input x.

        Predict target for x, using the best found pipeline(s) during the `fit` call.

        Parameters
        ----------
        x: pandas.DataFrame or numpy.ndarray
            Data with the same number of columns as the input to `fit`.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, K) with class probabilities where N is len(x),
             and K is the number of class labels found in `y` of `fit`.
        """
        x = self._prepare_for_prediction(x)
        return self._predict_proba(x)

    def predict_proba_from_file(
        self,
        arff_file_path: str,
        target_column: Optional[str] = None,
        encoding: Optional[str] = None,
    ):
        """Predict the class probabilities for input in the arff_file.

        Parameters
        ----------
        arff_file_path: str
            An ARFF file with the same columns as the one that used in fit.
            Target column must be present in file, but its values are ignored.
        target_column: str, optional (default=None)
            Specifies which column the model should predict.
            If left None, the last column is taken to be the target.
        encoding: str, optional
            Encoding of the ARFF file.

        Returns
        -------
        numpy.ndarray
            Numpy array with class probabilities.
            The array is of shape (N, K) where N is len(X),
            and K is the number of class labels found in `y` of `fit`.
        """
        x, _ = X_y_from_file(arff_file_path, target_column, encoding)
        x = self._prepare_for_prediction(x)
        return self._predict_proba(x)

    def fit(self, x, y, *args, **kwargs):
        """Should use base class documentation."""
        y_ = y.squeeze() if isinstance(y, pd.DataFrame) else y
        self._label_encoder = LabelEncoder().fit(y_)
        if any(isinstance(yi, str) for yi in y_):
            # If target values are `str` we encode them or scikit-learn will complain.
            y = self._label_encoder.transform(y_)
        self._evaluation_library.determine_sample_indices(stratify=y)

        # Add label information for classification to the scorer such that
        # the cross validator does not encounter unseen labels in smaller
        # data sets during pipeline evaluation.
        for m in self._metrics:
            if "labels" in inspect.signature(m.scorer._score_func).parameters:
                m.scorer._kwargs.update({"labels": y})

        super().fit(
            x=x,
            y=y,
            basic_encoding_step=False,
            fixed_encoding_step=False,
            *args,
            **kwargs,
        )

    def _encode_labels(self, y):
        self._label_encoder = LabelEncoder().fit(y)
        return self._label_encoder.transform(y)
