import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh


class PreprocessorConfig:
    """Manages the configuration space for preprocessors in supervised learning contexts

    PreprocessorConfig oversees the configuration space of preprocessors used in
    supervised machine learning tasks. This class facilitates the addition of
    new preprocessors and the modification of existing ones in the configuration space
    via standardised methods. The ConfigSpace library is used to designate the
    configuration space, enabling the creation of complex and adaptable
    configuration setups. For additional information on using constraints and
    various types of hyperparameters with ConfigSpace, refer to
    the ConfigSpace documentation, available at:
    https://automl.github.io/ConfigSpace/main/quickstart.html

    Add a preprocessor 💡
    ----------------

    1️⃣ To add a new preprocessor, define its setup method following the naming
    convention `setup_preprocessorName`. This method should:
        * Define hyperparameters specific to the preprocessor.
        * Use `_add_hyperparameters_and_equals_conditions` to add these
        hyperparameters to the config space with appropriate conditions.

    2️⃣ Next, your setup method needs to be added to the `preprocessors_setup_map` in
    the `__init__` method, where the key should be the Sci-kit learn name of your
    preprocessor, and the value should be pointing to your newly setup method.

    voila! 🎉 You are done! Your preprocessor is now added to the config space.

    How to use the shared hyperparameters 🪢
    -------------------------------------

    The shared hyperparameters are hyperparameters that are shared across multiple
    preprocessors. These hyperparameters are defined in the `shared_hyperparameters`
    property. To use these hyperparameters, simply add them to the setup method of
    the preprocessor you are adding. For example, to add the `gamma` hyperparameter to
    the `Nystroem` preprocessor, add the following line to the `setup_nystroem` method:

    >>>    gamma = csh.CategoricalHyperparameter(
    >>>        "gamma__Nystroem", self.shared_hyperparameters["gamma"]
    >>>    )

    voila! 🎉 The `gamma` hyperparameter is now added to the Nystroem preprocessor
    with the shared value available in the `shared_hyperparameters` property.

        How to name my hyperparameters ✍️
    ------------------------------

    The hyperparameters you add to the config space should be named in the following
    format if similar hyperparameter names can be found in other preprocessors:

    >>>    <hyperparameter_name>__<PreprocessorName>

    For example, the `gamma` hyperparameter for the `Nystroem` preprocessor should
    be named `gamma__Nystroem` given that the `gamma` hyperparameter is also
    available in the `RBFSampler` preprocessor. This naming convention is used to ensure
    that the hyperparameters are added to the correct preprocessor in the config space.

    If your hyperparameter name is unique to your preprocessor, you can name it as you
    please without the need to have `__<PreprocessorName>` at the end of the name.
    Nonetheless, following the naming convention would in any way not cause any issues.

    Modify an existing preprocessor 💅
    -------------------

    To modify an existing preprocessor, adjust its respective setup method and the
    shared hyperparameters property as needed by modifying the values of the
    hyperparameters. For example, to change the value of the `gamma` hyperparameter for
    the `Nystroem` preprocessor, change the value of the `gamma` hyperparameter
    in the `shared_hyperparameters` property by:

    >>>    "gamma": {"lower": 0.001, "upper": 0.8, "default_value": 0.5},

    The `gamma` hyperparameter will then be added to the config space with the
    appropriate value. However, be cautious, if you change values in the shared
    hyperparameters property, it will be changed for all preprocessors that use that
    hyperparameter. If you want this change to only apply to a specific preprocessor,
    you should add the hyperparameter to the setup method of that preprocessor.
    E.g. if you want to change the value of the `gamma` hyperparameter for the
    `Nystroem` preprocessor, and only want this change to apply to the `Nystroem`
    preprocessor, add the following line to the `setup_nystroem` method:

    >>>    gamma = csh.CategoricalHyperparameter(
    >>>        "gamma__Nystroem", {"lower": 0.001, "upper": 0.8, "default_value": 0.5},
    >>>    )

    The `gamma` hyperparameter will be added as-is for the `Nystroem` preprocessor
    and the value of the `gamma` hyperparameter for other preprocessors will be as
    available in the `shared_hyperparameters` property – iff they use the `gamma`
    hyperparameter of the `shared_hyperparameters` property.


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
        if "preprocessors" not in config_space.meta:
            raise ValueError("Expected 'preprocessors' key in meta of config_space")
        self.config_space = config_space
        self.preprocessors_setup_map = {
            "Dummy_To_Ignore": self.setup_dummy_to_ignore,
            "SelectFwe": self.setup_select_fwe,
            "Binarizer": self.setup_binarizer,
            "FastICA": self.setup_fast_ica,
            "FeatureAgglomeration": self.setup_feature_agglomeration,
            "MaxAbsScaler": self.setup_max_abs_scaler,
            "MinMaxScaler": self.setup_min_max_scaler,
            "Normalizer": self.setup_normalizer,
            "Nystroem": self.setup_nystroem,
            "PCA": self.setup_pca,
            "PolynomialFeatures": self.setup_polynomial_features,
            "RBFSampler": self.setup_rbf_sampler,
            "RobustScaler": self.setup_robust_scaler,
            "StandardScaler": self.setup_standard_scaler,
            "SelectPercentile": self.setup_select_percentile,
            "VarianceThreshold": self.setup_variance_threshold,
        }
        self.cs_preprocessors_name = config_space.meta["preprocessors"]

    @property
    def shared_hyperparameters(self):
        return {
            "gamma": {"lower": 0.01, "upper": 1.01, "default_value": 1.0},
        }

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

    def setup_dummy_to_ignore(self, preprocessors: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_select_fwe(self, preprocessors: csh.CategoricalHyperparameter):
        alpha = csh.UniformFloatHyperparameter(
            "alpha__SelectFwe", 0.01, 0.05, default_value=0.05
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "SelectFwe")

    def setup_binarizer(self, preprocessors: csh.CategoricalHyperparameter):
        threshold = csh.UniformFloatHyperparameter(
            "threshold__Binarizer", 0.0, 1.01, default_value=0.05
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "Binarizer")

    def setup_fast_ica(self, preprocessors: csh.CategoricalHyperparameter):
        whiten = csh.CategoricalHyperparameter("whiten", ["unit-variance"])
        tol = csh.UniformFloatHyperparameter(
            "tol__FastICA", 0.0, 1.01, default_value=0.05
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "FastICA")

    def setup_feature_agglomeration(self, preprocessors: csh.CategoricalHyperparameter):
        linkage = csh.CategoricalHyperparameter(
            "linkage__FeatureAgglomeration", ["ward", "complete", "average"]
        )
        metric = csh.CategoricalHyperparameter(
            "metric__FeatureAgglomeration",
            ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"],
        )
        self._add_hyperparameters_and_equals_conditions(
            locals(), "FeatureAgglomeration"
        )

        # Forbidden clause: Linkage is different from 'ward' and metric is 'euclidean'
        forbidden_penalty_loss = cs.ForbiddenAndConjunction(
            cs.ForbiddenInClause(
                self.config_space["linkage__FeatureAgglomeration"],
                ["complete", "average"],
            ),
            cs.ForbiddenEqualsClause(
                self.config_space["metric__FeatureAgglomeration"], "euclidean"
            ),
        )
        self.config_space.add_forbidden_clause(forbidden_penalty_loss)

    def setup_max_abs_scaler(self, preprocessors: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_min_max_scaler(self, preprocessors: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_normalizer(self, preprocessors: csh.CategoricalHyperparameter):
        norm = csh.CategoricalHyperparameter("norm", ["l1", "l2", "max"])
        self._add_hyperparameters_and_equals_conditions(locals(), "Normalizer")

    def setup_nystroem(self, preprocessors: csh.CategoricalHyperparameter):
        kernel = csh.CategoricalHyperparameter(
            "kernel",
            [
                "rbf",
                "cosine",
                "chi2",
                "laplacian",
                "polynomial",
                "poly",
                "linear",
                "additive_chi2",
                "sigmoid",
            ],
        )
        gamma = csh.UniformFloatHyperparameter(
            "gamma__Nystroem", **self.shared_hyperparameters["gamma"]
        )
        n_components = csh.UniformIntegerHyperparameter("n_components", 1, 11)
        self._add_hyperparameters_and_equals_conditions(locals(), "Nystroem")

    def setup_pca(self, preprocessors: csh.CategoricalHyperparameter):
        svd_solver = csh.CategoricalHyperparameter("svd_solver", ["randomized"])
        iterated_power = csh.UniformIntegerHyperparameter("iterated_power", 1, 11)
        self._add_hyperparameters_and_equals_conditions(locals(), "PCA")

    def setup_polynomial_features(self, preprocessors: csh.CategoricalHyperparameter):
        degree = csh.CategoricalHyperparameter("degree", [2])
        include_bias = csh.CategoricalHyperparameter("include_bias", [False])
        interaction_only = csh.CategoricalHyperparameter("interaction_only", [False])
        self._add_hyperparameters_and_equals_conditions(locals(), "PolynomialFeatures")

    def setup_rbf_sampler(self, preprocessors: csh.CategoricalHyperparameter):
        gamma = csh.UniformFloatHyperparameter(
            "gamma__RBFSampler", **self.shared_hyperparameters["gamma"]
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "RBFSampler")

    def setup_robust_scaler(self, preprocessors: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_standard_scaler(self, preprocessors: csh.CategoricalHyperparameter):
        # No hyperparameters
        pass

    def setup_select_percentile(self, preprocessors: csh.CategoricalHyperparameter):
        percentile = csh.UniformIntegerHyperparameter("percentile", 1, 100)
        self._add_hyperparameters_and_equals_conditions(locals(), "SelectPercentile")

    def setup_variance_threshold(self, preprocessors: csh.CategoricalHyperparameter):
        threshold = csh.UniformFloatHyperparameter(
            "threshold__VarianceThreshold", 0.05, 1.01, default_value=0.05
        )
        self._add_hyperparameters_and_equals_conditions(locals(), "VarianceThreshold")
