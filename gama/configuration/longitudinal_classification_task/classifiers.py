from typing import List, Dict

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh


class LongitudinalClassifierConfig:
    """Manages the configuration space for classifiers

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
        if any(
            meta_key not in config_space.meta
            for meta_key in ["data_preparation", "estimators"]
        ):
            raise ValueError(
                f"Expected 'data_preparation', 'estimators', "
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
        self.classifiers_setup_map = {
            "Dummy_To_Ignore": self.setup_dummy_to_ignore,
            # Longitudinal Algorithms
            "NestedTreesClassifier": self.setup_nested_trees,
            "LexicoRandomForestClassifier": self.setup_lexico_rf_classifier,
            "LexicoDecisionTreeClassifier": self.setup_lexico_decision_tree_classifier,
            "LexicoDeepForestClassifier": self.setup_lexico_deep_forest_classifier,
            "LexicoGradientBoostingClassifier": self.setup_lexico_gradient_boosting,
            # Conventional Algorithms
            "DecisionTreeClassifier": self.setup_decision_tree,
            "RandomForestClassifier": self.setup_random_forest,
            "ExtraTreesClassifier": self.setup_extra_trees,
            "KNeighborsClassifier": self.setup_k_neighbors,
            "LinearSVC": self.setup_linear_svc,
            "DeepForestClassifier": self.setup_deep_forest,
            "GradientBoostingClassifier": self.setup_gradient_boosting,
        }
        self.forbidden_clauses: List[Dict[str, List[str]]] = [
            {
                "data_preparations": ["MerWavTimePlus"],
                "classifiers_to_forbid": [
                    "DecisionTreeClassifier",
                    "RandomForestClassifier",
                    "ExtraTreesClassifier",
                    "KNeighborsClassifier",
                    "LinearSVC",
                    "DeepForestClassifier",
                    "GradientBoostingClassifier",
                ],
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
                "classifiers_to_forbid": [
                    "NestedTreesClassifier",
                    "LexicoRandomForestClassifier",
                    "LexicoDecisionTreeClassifier",
                    "LexicoDeepForestClassifier",
                    "LexicoGradientBoostingClassifier",
                ],
            },
        ]
        self.cs_data_preparation_name = self.config_space.meta["data_preparation"]
        self.cs_estimators_name = self.config_space.meta["estimators"]

    @property
    def shared_hyperparameters(self):
        return {
            "criterion": ["gini", "entropy"],
            "max_depth": {"lower": 2, "upper": 10, "default_value": 2},
            "min_samples_split": {"lower": 2, "upper": 20, "default_value": 2},
            "min_samples_leaf": {"lower": 1, "upper": 20, "default_value": 1},
            "n_estimators": {"lower": 100, "upper": 1000, "default_value": 100},
            "n_estimators_deep_forests": {"lower": 2, "upper": 3, "default_value": 2},
            "bootstrap": [True, False],
            "threshold_gain": {
                "choices": [
                    0.0,
                    0.001,
                    0.0015,
                    0.002,
                    0.0025,
                    0.003,
                    0.0035,
                    0.004,
                    0.0045,
                    0.005,
                    0.01,
                ],
                "default_value": 0.0,
            },
        }

    def setup_classifiers(self):
        classifiers_choices = list(self.classifiers_setup_map.keys())

        if not classifiers_choices:
            raise ValueError("No classifiers to add to config space")

        classifiers = csh.CategoricalHyperparameter(
            name=self.cs_estimators_name,
            choices=classifiers_choices,
            default_value=classifiers_choices[0],
        )
        self.config_space.add_hyperparameter(classifiers)

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
                    cs.ForbiddenInClause(classifiers, clause["classifiers_to_forbid"]),
                    cs.ForbiddenInClause(data_preparation, clause["data_preparations"]),
                )
                for clause in self.forbidden_clauses
                if all(
                    data_preparation_name in data_preparation_choices
                    for data_preparation_name in clause["data_preparations"]
                )
            ]
            self.config_space.add_forbidden_clauses(forbidden_clauses)

        for classifier_name in classifiers_choices:
            if setup_func := self.classifiers_setup_map.get(classifier_name):
                setup_func(classifiers)

    def _add_hyperparameters_and_equals_conditions(
        self,
        local_vars: dict,
        estimator_name: str,
    ):
        if "classifiers" not in local_vars or not isinstance(
            local_vars["classifiers"], csh.CategoricalHyperparameter
        ):
            raise ValueError(
                "Expected 'classifiers' key with a CategoricalHyperparameter in local"
                "vars"
            )

        hyperparameters_to_add = [
            hyperparameter
            for hyperparameter in local_vars.values()
            if isinstance(hyperparameter, csh.Hyperparameter)
            and hyperparameter != local_vars["classifiers"]
        ]

        conditions_to_add = [
            cs.EqualsCondition(
                hyperparameter, local_vars["classifiers"], estimator_name
            )
            for hyperparameter in hyperparameters_to_add
        ]

        self.config_space.add_hyperparameters(hyperparameters_to_add)
        self.config_space.add_conditions(conditions_to_add)

    def setup_dummy_to_ignore(self, classifiers: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_nested_trees(self, classifiers: csh.CategoricalHyperparameter):
        max_outer_depth = csh.UniformIntegerHyperparameter(
            "max_outer_depth__NestedTreesClassifier",
            lower=5,
            upper=10,
            default_value=10,
        )
        max_inner_depth = csh.UniformIntegerHyperparameter(
            "max_inner_depth__NestedTreesClassifier",
            lower=2,
            upper=5,
            default_value=5,
        )
        min_node_size = csh.UniformIntegerHyperparameter(
            "min_outer_samples__NestedTreesClassifier",
            lower=2,
            upper=10,
            default_value=2,
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "NestedTreesClassifier"
        )

    def setup_lexico_rf_classifier(self, classifiers: csh.CategoricalHyperparameter):
        min_samples_split = csh.UniformIntegerHyperparameter(
            "min_samples_split__LexicoRandomForestClassifier",
            **self.shared_hyperparameters["min_samples_split"],
        )
        n_estimators = csh.UniformIntegerHyperparameter(
            "n_estimators__LexicoRandomForestClassifier",
            **self.shared_hyperparameters["n_estimators"],
        )
        min_samples_leaf = csh.UniformIntegerHyperparameter(
            "min_samples_leaf__LexicoRandomForestClassifier",
            **self.shared_hyperparameters["min_samples_leaf"],
        )
        bootstrap = csh.CategoricalHyperparameter(
            "bootstrap__LexicoRandomForestClassifier",
            self.shared_hyperparameters["bootstrap"],
        )
        threshold_gain = csh.CategoricalHyperparameter(
            "threshold_gain__LexicoRandomForestClassifier",
            **self.shared_hyperparameters["threshold_gain"],
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "LexicoRandomForestClassifier"
        )

    def setup_lexico_decision_tree_classifier(
        self, classifiers: csh.CategoricalHyperparameter
    ):
        min_samples_split = csh.UniformIntegerHyperparameter(
            "min_samples_split__LexicoDecisionTreeClassifier",
            **self.shared_hyperparameters["min_samples_split"],
        )
        min_samples_leaf = csh.UniformIntegerHyperparameter(
            "min_samples_leaf__LexicoDecisionTreeClassifier",
            **self.shared_hyperparameters["min_samples_leaf"],
        )
        max_depth = csh.UniformIntegerHyperparameter(
            "max_depth__LexicoDecisionTreeClassifier",
            **self.shared_hyperparameters["max_depth"],
        )
        threshold_gain = csh.CategoricalHyperparameter(
            "threshold_gain__LexicoDecisionTreeClassifier",
            **self.shared_hyperparameters["threshold_gain"],
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "LexicoDecisionTreeClassifier"
        )

    def setup_lexico_deep_forest_classifier(
        self, classifiers: csh.CategoricalHyperparameter
    ):
        single_classifier_type = csh.CategoricalHyperparameter(
            "single_classifier_type__LexicoDeepForestClassifier",
            ["LexicoRandomForestClassifier", "LexicoCompleteRFClassifier"],
        )
        single_count = csh.UniformIntegerHyperparameter(
            "single_count__LexicoDeepForestClassifier",
            **self.shared_hyperparameters["n_estimators_deep_forests"],
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "LexicoDeepForestClassifier"
        )

    def setup_decision_tree(self, classifiers: csh.CategoricalHyperparameter):
        criterion = csh.CategoricalHyperparameter(
            "criterion__DecisionTreeClassifier",
            self.shared_hyperparameters["criterion"],
        )
        min_samples_split = csh.UniformIntegerHyperparameter(
            "min_samples_split__DecisionTreeClassifier",
            **self.shared_hyperparameters["min_samples_split"],
        )
        min_samples_leaf = csh.UniformIntegerHyperparameter(
            "min_samples_leaf__DecisionTreeClassifier",
            **self.shared_hyperparameters["min_samples_leaf"],
        )
        max_depth = csh.UniformIntegerHyperparameter(
            "max_depth__DecisionTreeClassifier",
            **self.shared_hyperparameters["max_depth"],
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "DecisionTreeClassifier"
        )

    def setup_random_forest(self, classifiers: csh.CategoricalHyperparameter):
        criterion = csh.CategoricalHyperparameter(
            "criterion__RandomForestClassifier",
            self.shared_hyperparameters["criterion"],
        )
        min_samples_split = csh.UniformIntegerHyperparameter(
            "min_samples_split__RandomForestClassifier",
            **self.shared_hyperparameters["min_samples_split"],
        )
        n_estimators = csh.UniformIntegerHyperparameter(
            "n_estimators__RandomForestClassifier",
            **self.shared_hyperparameters["n_estimators"],
        )
        min_samples_leaf = csh.UniformIntegerHyperparameter(
            "min_samples_leaf__RandomForestClassifier",
            **self.shared_hyperparameters["min_samples_leaf"],
        )
        bootstrap = csh.CategoricalHyperparameter(
            "bootstrap__RandomForestClassifier",
            self.shared_hyperparameters["bootstrap"],
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "RandomForestClassifier"
        )

    def setup_extra_trees(self, classifiers: csh.CategoricalHyperparameter):
        criterion = csh.CategoricalHyperparameter(
            "criterion__ExtraTreesClassifier",
            self.shared_hyperparameters["criterion"],
        )
        n_estimators = csh.UniformIntegerHyperparameter(
            "n_estimators__ExtraTreesClassifier",
            **self.shared_hyperparameters["n_estimators"],
        )
        min_samples_split = csh.UniformIntegerHyperparameter(
            "min_samples_split__ExtraTreesClassifier",
            **self.shared_hyperparameters["min_samples_split"],
        )
        min_samples_leaf = csh.UniformIntegerHyperparameter(
            "min_samples_leaf__ExtraTreesClassifier",
            **self.shared_hyperparameters["min_samples_leaf"],
        )
        bootstrap = csh.CategoricalHyperparameter(
            "bootstrap__ExtraTreesClassifier",
            self.shared_hyperparameters["bootstrap"],
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "ExtraTreesClassifier"
        )

    def setup_k_neighbors(self, classifiers: csh.CategoricalHyperparameter):
        n_neighors = csh.CategoricalHyperparameter(
            "n_neighbors__KNeighborsClassifier",
            [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50],
        )
        weights = csh.CategoricalHyperparameter(
            "weights__KNeighborsClassifier", ["uniform", "distance"]
        )
        p = csh.CategoricalHyperparameter("p__KNeighborsClassifier", [1, 2])
        self._add_hyperparameters_and_equals_conditions(
            locals(), "KNeighborsClassifier"
        )

    def setup_linear_svc(self, classifiers: csh.CategoricalHyperparameter):
        loss = csh.CategoricalHyperparameter(
            "loss__LinearSVC", ["hinge", "squared_hinge"]
        )
        penalty = csh.CategoricalHyperparameter("penalty__LinearSVC", ["l1", "l2"])
        dual = csh.CategoricalHyperparameter("dual__LinearSVC", [True, False])
        tol = csh.CategoricalHyperparameter(
            "tol__LinearSVC", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        )
        C = csh.CategoricalHyperparameter(
            "C__LinearSVC",
            [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "LinearSVC")

        # Forbidden clause: Penalty 'l1' cannot be used with loss 'hinge'
        forbidden_penalty_loss = cs.ForbiddenAndConjunction(
            cs.ForbiddenEqualsClause(self.config_space["penalty__LinearSVC"], "l1"),
            cs.ForbiddenEqualsClause(self.config_space["loss__LinearSVC"], "hinge"),
        )
        # Forbidden clause: Penalty 'l2' cannot be used
        # with loss 'hinge' and dual 'False'
        forbidden_penalty_loss_dual_false = cs.ForbiddenAndConjunction(
            cs.ForbiddenEqualsClause(self.config_space["penalty__LinearSVC"], "l2"),
            cs.ForbiddenEqualsClause(self.config_space["loss__LinearSVC"], "hinge"),
            cs.ForbiddenEqualsClause(self.config_space["dual__LinearSVC"], False),
        )
        self.config_space.add_forbidden_clauses(
            [forbidden_penalty_loss, forbidden_penalty_loss_dual_false]
        )

    def setup_deep_forest(self, classifiers: csh.CategoricalHyperparameter):
        n_estimators = csh.UniformIntegerHyperparameter(
            "n_estimators__DeepForestClassifier",
            **self.shared_hyperparameters["n_estimators_deep_forests"],
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "DeepForestClassifier"
        )

    def setup_gradient_boosting(self, classifiers: csh.CategoricalHyperparameter):
        max_depth = csh.UniformIntegerHyperparameter(
            "max_depth__GradientBoostingClassifier",
            lower=2,
            upper=10,
            default_value=2,
        )
        learning_rate = csh.UniformFloatHyperparameter(
            "learning_rate__GradientBoostingClassifier",
            lower=0.01,
            upper=0.5,
            default_value=0.1,
        )
        n_estimators = csh.UniformIntegerHyperparameter(
            "n_estimators__GradientBoostingClassifier",
            **self.shared_hyperparameters["n_estimators"],
        )

        self._add_hyperparameters_and_equals_conditions(
            locals(), "GradientBoostingClassifier"
        )

    def setup_lexico_gradient_boosting(
        self, classifiers: csh.CategoricalHyperparameter
    ):
        max_depth = csh.UniformIntegerHyperparameter(
            "max_depth__LexicoGradientBoostingClassifier",
            lower=2,
            upper=10,
            default_value=2,
        )
        learning_rate = csh.UniformFloatHyperparameter(
            "learning_rate__LexicoGradientBoostingClassifier",
            lower=0.01,
            upper=0.5,
            default_value=0.1,
        )
        n_estimators = csh.UniformIntegerHyperparameter(
            "n_estimators__LexicoGradientBoostingClassifier",
            **self.shared_hyperparameters["n_estimators"],
        )
        threshold_gain = csh.CategoricalHyperparameter(
            "threshold_gain__LexicoGradientBoostingClassifier",
            **self.shared_hyperparameters["threshold_gain"],
        )

        self._add_hyperparameters_and_equals_conditions(
            locals(), "LexicoGradientBoostingClassifier"
        )
