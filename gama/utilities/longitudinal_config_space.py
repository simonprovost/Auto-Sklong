# type: ignore
# flake8: noqa
from deepforest import CascadeForestClassifier
from scikit_longitudinal.data_preparation import AggrFunc, SepWav
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
    LongitudinalEnsemblingStrategy,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels


class AggrFuncMean(AggrFunc):
    def __init__(
        self,
        features_group=None,
        non_longitudinal_features=None,
        feature_list_names=None,
        aggregation_func="mean",
        parallel=False,
        num_cpus=-1,
    ):
        super().__init__(
            features_group=features_group,
            non_longitudinal_features=non_longitudinal_features,
            feature_list_names=feature_list_names,
            aggregation_func=aggregation_func,
            parallel=parallel,
            num_cpus=num_cpus,
        )


class AggrFuncMedian(AggrFunc):
    def __init__(
        self,
        features_group=None,
        non_longitudinal_features=None,
        feature_list_names=None,
        aggregation_func="median",
        parallel=False,
        num_cpus=-1,
    ):
        super().__init__(
            features_group=features_group,
            non_longitudinal_features=non_longitudinal_features,
            feature_list_names=feature_list_names,
            aggregation_func=aggregation_func,
            parallel=parallel,
            num_cpus=num_cpus,
        )


class AggrFuncMode(AggrFunc):
    def __init__(
        self,
        features_group=None,
        non_longitudinal_features=None,
        feature_list_names=None,
        aggregation_func="mode",
        parallel=False,
        num_cpus=-1,
    ):
        super().__init__(
            features_group=features_group,
            non_longitudinal_features=non_longitudinal_features,
            feature_list_names=feature_list_names,
            aggregation_func=aggregation_func,
            parallel=parallel,
            num_cpus=num_cpus,
        )


class SepWavMajorityVoting(SepWav):
    def __init__(
        self,
        estimator=None,
        features_group=None,
        non_longitudinal_features=None,
        feature_list_names=None,
        voting=LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
        stacking_meta_learner=LogisticRegression(),
        n_jobs: int = None,
        parallel: bool = False,
        num_cpus: int = -1,
    ):
        super().__init__(
            estimator=estimator,
            features_group=features_group,
            non_longitudinal_features=non_longitudinal_features,
            feature_list_names=feature_list_names,
            voting=voting,
            stacking_meta_learner=stacking_meta_learner,
            n_jobs=n_jobs,
            parallel=parallel,
            num_cpus=num_cpus,
        )


class SepWavStackingLogisticRegression(SepWav):
    def __init__(
        self,
        estimator=None,
        features_group=None,
        non_longitudinal_features=None,
        feature_list_names=None,
        voting=LongitudinalEnsemblingStrategy.STACKING,
        stacking_meta_learner=LogisticRegression(),
        n_jobs: int = None,
        parallel: bool = False,
        num_cpus: int = -1,
    ):
        super().__init__(
            estimator=estimator,
            features_group=features_group,
            non_longitudinal_features=non_longitudinal_features,
            feature_list_names=feature_list_names,
            voting=voting,
            stacking_meta_learner=stacking_meta_learner,
            n_jobs=n_jobs,
            parallel=parallel,
            num_cpus=num_cpus,
        )


class SepWavStackingDecisionTree(SepWav):
    def __init__(
        self,
        estimator=None,
        features_group=None,
        non_longitudinal_features=None,
        feature_list_names=None,
        voting=LongitudinalEnsemblingStrategy.STACKING,
        stacking_meta_learner=DecisionTreeClassifier(),
        n_jobs: int = None,
        parallel: bool = False,
        num_cpus: int = -1,
    ):
        super().__init__(
            estimator=estimator,
            features_group=features_group,
            non_longitudinal_features=non_longitudinal_features,
            feature_list_names=feature_list_names,
            voting=voting,
            stacking_meta_learner=stacking_meta_learner,
            n_jobs=n_jobs,
            parallel=parallel,
            num_cpus=num_cpus,
        )


class SepWavStackingRandomForest(SepWav):
    def __init__(
        self,
        estimator=None,
        features_group=None,
        non_longitudinal_features=None,
        feature_list_names=None,
        voting=LongitudinalEnsemblingStrategy.STACKING,
        stacking_meta_learner=RandomForestClassifier(),
        n_jobs: int = None,
        parallel: bool = False,
        num_cpus: int = -1,
    ):
        super().__init__(
            estimator=estimator,
            features_group=features_group,
            non_longitudinal_features=non_longitudinal_features,
            feature_list_names=feature_list_names,
            voting=voting,
            stacking_meta_learner=stacking_meta_learner,
            n_jobs=n_jobs,
            parallel=parallel,
            num_cpus=num_cpus,
        )


class SepWavDecayLinearWeighting(SepWav):
    def __init__(
        self,
        estimator=None,
        features_group=None,
        non_longitudinal_features=None,
        feature_list_names=None,
        voting=LongitudinalEnsemblingStrategy.DECAY_LINEAR_VOTING,
        stacking_meta_learner=None,
        n_jobs: int = None,
        parallel: bool = False,
        num_cpus: int = -1,
    ):
        super().__init__(
            estimator=estimator,
            features_group=features_group,
            non_longitudinal_features=non_longitudinal_features,
            feature_list_names=feature_list_names,
            voting=voting,
            stacking_meta_learner=stacking_meta_learner,
            n_jobs=n_jobs,
            parallel=parallel,
            num_cpus=num_cpus,
        )


class SepWavDecayExponentialWeighting(SepWav):
    def __init__(
        self,
        estimator=None,
        features_group=None,
        non_longitudinal_features=None,
        feature_list_names=None,
        voting=LongitudinalEnsemblingStrategy.DECAY_EXPONENTIAL_VOTING,
        stacking_meta_learner=None,
        n_jobs: int = None,
        parallel: bool = False,
        num_cpus: int = -1,
    ):
        super().__init__(
            estimator=estimator,
            features_group=features_group,
            non_longitudinal_features=non_longitudinal_features,
            feature_list_names=feature_list_names,
            voting=voting,
            stacking_meta_learner=stacking_meta_learner,
            n_jobs=n_jobs,
            parallel=parallel,
            num_cpus=num_cpus,
        )


class SepWavDecayCrossValidation(SepWav):
    def __init__(
        self,
        estimator=None,
        features_group=None,
        non_longitudinal_features=None,
        feature_list_names=None,
        voting=LongitudinalEnsemblingStrategy.CV_BASED_VOTING,
        stacking_meta_learner=None,
        n_jobs: int = None,
        parallel: bool = False,
        num_cpus: int = -1,
    ):
        super().__init__(
            estimator=estimator,
            features_group=features_group,
            non_longitudinal_features=non_longitudinal_features,
            feature_list_names=feature_list_names,
            voting=voting,
            stacking_meta_learner=stacking_meta_learner,
            n_jobs=n_jobs,
            parallel=parallel,
            num_cpus=num_cpus,
        )


class CascadeForestClassifier(CascadeForestClassifier):
    def __init__(
        self,
        n_bins=255,
        bin_subsample=200000,
        bin_type="percentile",
        max_layers=20,
        criterion="gini",
        n_estimators=2,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        use_predictor=False,
        predictor="forest",
        predictor_kwargs={},
        backend="custom",
        n_tolerant_rounds=2,
        delta=1e-5,
        partial_mode=False,
        n_jobs=None,
        random_state=None,
        verbose=1,
    ):
        self.classes_ = None
        super().__init__(
            n_bins=n_bins,
            bin_subsample=bin_subsample,
            bin_type=bin_type,
            max_layers=max_layers,
            criterion=criterion,
            n_estimators=n_estimators,
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            use_predictor=use_predictor,
            predictor=predictor,
            predictor_kwargs=predictor_kwargs,
            backend=backend,
            n_tolerant_rounds=n_tolerant_rounds,
            delta=delta,
            partial_mode=partial_mode,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def fit(self, X, y):
        if self.classes_ is None:
            self.classes_ = unique_labels(y)
        super().fit(X, y)
        print("CascadeForestClassifier is fitted.")
        return self
