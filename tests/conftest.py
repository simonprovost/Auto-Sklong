import pytest
from scikit_longitudinal.data_preparation import LongitudinalDataset

from gama import GamaClassifier
from gama.GamaLongitudinalClassifier import GamaLongitudinalClassifier
from gama.configuration.testconfiguration import config_space as test_config_space
from gama.genetic_programming.compilers.scikitlearn import compile_individual
from gama.genetic_programming.components import Individual


@pytest.fixture
def config_space():
    gc = GamaClassifier(
        search_space=test_config_space, scoring="accuracy", store="nothing"
    )
    yield gc.search_space
    gc.cleanup("all")


@pytest.fixture
def opset():
    gc = GamaClassifier(
        search_space=test_config_space, scoring="accuracy", store="nothing"
    )
    yield gc._operator_set
    gc.cleanup("all")


@pytest.fixture
def GNB(config_space):
    return Individual.from_string("GaussianNB(data)", config_space, compile_individual)


@pytest.fixture
def RS_MNB(config_space):
    return Individual.from_string(
        "MultinomialNB(RobustScaler(data), alpha=1.0, fit_prior=True)",
        config_space,
        compile_individual,
    )


@pytest.fixture
def SS_BNB(config_space):
    return Individual.from_string(
        "BernoulliNB(StandardScaler(data), alpha=0.1, fit_prior=True)",
        config_space,
        compile_individual,
    )


@pytest.fixture
def SS_RBS_SS_BNB(config_space):
    return Individual.from_string(
        "BernoulliNB("
        "StandardScaler(RobustScaler("
        "StandardScaler(data))), alpha=0.1, fit_prior=True)",
        # noqa: E501
        config_space,
        compile_individual,
    )


@pytest.fixture
def LinearSVC(config_space):
    individual_str = """LinearSVC(data,
            LinearSVC.C=0.001,
            LinearSVC.dual=True,
            LinearSVC.loss='squared_hinge',
            LinearSVC.penalty='l2',
            LinearSVC.tol=1e-05)"""
    individual_str = "".join(individual_str.split()).replace(",", ", ")
    return Individual.from_string(individual_str, config_space, None)


@pytest.fixture
def ForestPipeline(config_space):
    individual_str = """RandomForestClassifier(
            FeatureAgglomeration(
                    data,
                    FeatureAgglomeration.metric='l2',
                    FeatureAgglomeration.linkage='complete'
                    ),
            RandomForestClassifier.bootstrap=True,
            RandomForestClassifier.criterion='gini',
            RandomForestClassifier.min_samples_leaf=7,
            RandomForestClassifier.min_samples_split=6,
            RandomForestClassifier.n_estimators=100)"""
    individual_str = "".join(individual_str.split()).replace(",", ", ")

    return Individual.from_string(individual_str, config_space, None)


@pytest.fixture
def InvalidLinearSVC(config_space):
    individual_str = """LinearSVC(data,
            LinearSVC.C=0.001,
            LinearSVC.dual=True,
            LinearSVC.loss='squared_hinge',
            LinearSVC.penalty='l1',
            LinearSVC.tol=1e-05)"""
    individual_str = "".join(individual_str.split()).replace(",", ", ")
    return Individual.from_string(individual_str, config_space, compile_individual)


# Longitudinal fixtures


def _get_private_dataset():
    """
    Currently, is being loaded a dummy dataset.
    Could be replaced with a real dataset of your choice.
    Nonetheless, either public via e.g OpenML or local but if local,
    do not forget to change the path prior to pushing.
    """
    data_file = "tests/data/stroke_longitudinal.csv"
    target_column = "stroke_w2"

    longitudinal_data = LongitudinalDataset(data_file)
    longitudinal_data.load_data_target_train_test_split(
        target_column, remove_target_waves=True, random_state=42
    )
    return longitudinal_data


@pytest.fixture
def longitudinal_config_space():
    longitudinal_data = _get_private_dataset()
    features_group = longitudinal_data.feature_groups()
    non_longitudinal_features = longitudinal_data.non_longitudinal_features()
    feature_list_names = longitudinal_data.data.columns.tolist()

    gc = GamaLongitudinalClassifier(
        features_group=features_group,
        non_longitudinal_features=non_longitudinal_features,
        feature_list_names=feature_list_names,
        scoring="accuracy",
        store="nothing",
        n_jobs=1,
        random_state=42,
    )
    yield gc.search_space
    gc.cleanup("all")


@pytest.fixture
def longitudinal_opset():
    longitudinal_data = _get_private_dataset()
    features_group = longitudinal_data.feature_groups()
    non_longitudinal_features = longitudinal_data.non_longitudinal_features()
    feature_list_names = longitudinal_data.data.columns.tolist()

    gc = GamaLongitudinalClassifier(
        features_group=features_group,
        non_longitudinal_features=non_longitudinal_features,
        feature_list_names=feature_list_names,
        scoring="accuracy",
        store="nothing",
        n_jobs=1,
        random_state=42,
    )
    yield gc._operator_set
    gc.cleanup("all")


def compile_longitudinal_individual(individual):
    return longitudinal_opset._compile(individual)


@pytest.fixture
def Longitudinal_DT_NO_TERMS(longitudinal_config_space):
    return Individual.from_string(
        "DecisionTreeClassifier(MerWavTimeMinus(data))",
        longitudinal_config_space,
        compile_longitudinal_individual,
    )


@pytest.fixture
def Longitudinal_DT(longitudinal_config_space):
    return Individual.from_string(
        "DecisionTreeClassifier("
        "MerWavTimeMinus(data), DecisionTreeClassifier.min_samples_leaf=4)",
        longitudinal_config_space,
        compile_longitudinal_individual,
    )


@pytest.fixture
def Longitudinal_CFS_DT(longitudinal_config_space):
    return Individual.from_string(
        "DecisionTreeClassifier("
        "CorrelationBasedFeatureSelection("
        "MerWavTimeMinus(data)), DecisionTreeClassifier.min_samples_leaf=7)",
        longitudinal_config_space,
        compile_longitudinal_individual,
    )


@pytest.fixture
def LongitudinalLinearSVC(longitudinal_config_space):
    individual_str = """LinearSVC(MerWavTimeMinus(data),
            LinearSVC.C=0.001,
            LinearSVC.dual=True,
            LinearSVC.loss='squared_hinge',
            LinearSVC.penalty='l2',
            LinearSVC.tol=1e-05)"""
    individual_str = "".join(individual_str.split()).replace(",", ", ")
    return Individual.from_string(individual_str, longitudinal_config_space, None)


@pytest.fixture
def LongitudinalForestPipeline(longitudinal_config_space):
    individual_str = """RandomForestClassifier(
            MerWavTimeMinus(data),
            RandomForestClassifier.bootstrap=True,
            RandomForestClassifier.criterion='gini',
            RandomForestClassifier.min_samples_leaf=7,
            RandomForestClassifier.min_samples_split=6,
            RandomForestClassifier.n_estimators=100)"""
    individual_str = "".join(individual_str.split()).replace(",", ", ")

    return Individual.from_string(individual_str, longitudinal_config_space, None)


@pytest.fixture
def LongitudinalInvalidLinearSVC(longitudinal_config_space):
    individual_str = """LinearSVC(MerWavTimePlus(data),
            LinearSVC.C=0.001,
            LinearSVC.dual=True,
            LinearSVC.loss='squared_hinge',
            LinearSVC.penalty='l1',
            LinearSVC.tol=1e-05)"""
    individual_str = "".join(individual_str.split()).replace(",", ", ")
    return Individual.from_string(
        individual_str, longitudinal_config_space, compile_individual
    )
