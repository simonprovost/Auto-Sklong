import copy
import random
from typing import List, Optional, Any, Union, Callable, Tuple

import ConfigSpace as cs

from gama.genetic_programming.components import (
    Primitive,
    PrimitiveNode,
    DATA_TERMINAL,
    Terminal,
)
from gama.utilities.config_space import (
    get_internal_output_types,
    get_hyperparameter_sklearn_name,
    get_estimator_by_name,
)


def _sample_and_ignore_dummy_techniques(
    config_space: cs.ConfigurationSpace,
    config: cs.Configuration,
    technique_name_to_ignore: str = "Dummy_To_Ignore",
) -> cs.Configuration:
    """ConfigSpace do not allow adding forbidden
    clauses on the default choice of a hyperparameter.

    Dummy_To_Ignore becomes default and is ignored / found
    an alternative when picked by sample_configuration.
    Nonetheless, for better maintainability, "Dummy_To_Ignore" can be changed,
    for example in the future it could be in the meta of the config so that
    it can be changed in one place by the user, i.e in /configuratins.
    """
    is_preprocessor_to_ignore = (
        "preprocessors" in config_space.meta
        and config[config_space.meta["preprocessors"]] == technique_name_to_ignore
    )
    is_estimator_to_ignore = (
        "estimators" in config_space.meta
        and config[config_space.meta["estimators"]] == technique_name_to_ignore
    )
    if is_estimator_to_ignore or is_preprocessor_to_ignore:
        while is_estimator_to_ignore or is_preprocessor_to_ignore:
            temp_config = config_space.sample_configuration()
            is_preprocessor_to_ignore = (
                "preprocessors" in config_space.meta
                and temp_config[config_space.meta["preprocessors"]]
                == technique_name_to_ignore
            )
            is_estimator_to_ignore = (
                "estimators" in config_space.meta
                and temp_config[config_space.meta["estimators"]]
                == technique_name_to_ignore
            )
            if not is_estimator_to_ignore and not is_preprocessor_to_ignore:
                return temp_config
    return config


def random_primitive_node(
    output_type: str,
    config_space: cs.ConfigurationSpace,
    exclude: Optional[Primitive] = None,
) -> PrimitiveNode:
    """Create a PrimitiveNode with specified output_type and random terminals."""
    if output_type not in get_internal_output_types():
        raise ValueError(f"Output type {output_type} not supported")

    if exclude is not None:
        temp_config_space = copy.deepcopy(config_space)
        if output_type not in temp_config_space.meta:
            raise ValueError(f"Output type {output_type} not in config_space meta.")
        temp_config_space.add_forbidden_clause(
            cs.ForbiddenEqualsClause(
                temp_config_space.get_hyperparameter(
                    temp_config_space.meta[output_type]
                ),
                exclude.__str__(),
            )
        )
        config = _sample_and_ignore_dummy_techniques(
            temp_config_space, temp_config_space.sample_configuration()
        )
    else:
        config = _sample_and_ignore_dummy_techniques(
            config_space, config_space.sample_configuration()
        )

    if output_type in [DATA_TERMINAL, "preprocessors"]:
        (
            preprocessor_primitive,
            preprocessor_terminals,
        ) = _config_component_to_primitive_node(
            "preprocessors",
            config,
            config_space.meta,
            config_space.get_conditions(),
            ["estimators", "preprocessors"],
        )
        return PrimitiveNode(
            preprocessor_primitive,
            data_node=DATA_TERMINAL,
            terminals=preprocessor_terminals,
        )
    estimator_primitive, estimator_terminals = _config_component_to_primitive_node(
        "estimators",
        config,
        config_space.meta,
        config_space.get_conditions(),
        ["estimators", "preprocessors"],
    )
    return PrimitiveNode(
        primitive=estimator_primitive,
        data_node=DATA_TERMINAL,
        terminals=estimator_terminals,
    )


def create_random_expression(
    config_space: cs.ConfigurationSpace,
    min_length: int = 1,
    max_length: int = 3,
) -> PrimitiveNode:
    """Create at least min_length and at most max_length chained PrimitiveNodes."""
    individual_length = random.randint(min_length, max_length)
    config = _sample_and_ignore_dummy_techniques(
        config_space, config_space.sample_configuration()
    )
    return _config_to_primitive_node(
        config=config,
        config_meta=config_space.meta,
        conditions=config_space.get_conditions(),
        config_length=individual_length,
    )


def _config_to_primitive_node(
    config: cs.Configuration,
    config_meta: dict,
    conditions: List[Any],
    config_length: Optional[int] = None,
) -> PrimitiveNode:
    """Create a PrimitiveNode from a configuration. If config_length is specified, the
    PrimitiveNode will have at most config_length PrimitiveNodes."""
    if isinstance(config_length, int) and config_length <= 1:
        estimator_primitive, estimator_terminals = _config_component_to_primitive_node(
            "estimators",
            config,
            config_meta,
            conditions,
            ["estimators", "preprocessors"],
        )
        return PrimitiveNode(
            estimator_primitive, data_node=DATA_TERMINAL, terminals=estimator_terminals
        )
    estimator_primitive, estimator_terminals = _config_component_to_primitive_node(
        "estimators", config, config_meta, conditions, ["estimators", "preprocessors"]
    )

    (
        preprocessor_primitive,
        preprocessor_terminals,
    ) = _config_component_to_primitive_node(
        "preprocessors",
        config,
        config_meta,
        conditions,
        ["estimators", "preprocessors"],
    )

    # Create a PrimitiveNode for the preprocessor
    preprocessor_node = PrimitiveNode(
        preprocessor_primitive,
        data_node=DATA_TERMINAL,
        terminals=preprocessor_terminals,
    )

    # Create a PrimitiveNode for the classifier and chain it to the preprocessor
    return PrimitiveNode(
        estimator_primitive, data_node=preprocessor_node, terminals=estimator_terminals
    )


def _config_component_to_primitive_node(
    component_type: str,
    config: cs.Configuration,
    config_meta: dict,
    conditions: List[Any],
    exclude_keys: List[str],
    output_type: Optional[str] = None,
    retrieve_estimator: Callable = get_estimator_by_name,
) -> Tuple[Primitive, List[Terminal]]:
    """
    Create a PrimitiveNode from a configuration of type ConfigSpace
    (estimator or preprocessor).

    This function generalizes the creation of a PrimitiveNode for either an estimator or
    a preprocessor, based on the component type specified. It creates a Primitive for
    the selected component, determines the valid hyperparameters for the component
    based on the conditions, and then creates a Terminal for each valid hyperparameter.

    Parameters
    ----------
    component_type : str
        The type of component (e.g., 'estimators' or 'preprocessors').
    config : cs.Configuration
        A configuration of type ConfigSpace.
    config_meta : dict
        The meta information of the ConfigSpace.
    conditions : List[Any]
        The conditions of the ConfigSpace.
    exclude_keys : List[str]
        Keys to exclude when considering hyperparameters.
    output_type : str, optional
        The output type of the PrimitiveNode, by default None.
    """
    if (
        component_type not in config_meta
        or config_meta[component_type] not in config.get_dictionary()
    ):
        raise ValueError(
            f"Configuration {config} does not contain a `{component_type}` ConfigSpace"
            "Hyperparameter. Cannot construct PrimitiveNode."
        )

    if output_type is None:
        output_type = component_type

    # Create a Primitive for the selected component
    component_primitive = Primitive(
        identifier=retrieve_estimator(config[config_meta[component_type]]),
        output=output_type,
        input=tuple(
            get_hyperparameter_sklearn_name(hp)
            for hp in config
            if hp not in [config_meta.get(key) for key in exclude_keys]
        ),
    )

    # Determine valid hyperparameters for the component based on conditions
    component_valid_hyperparameters = [
        name
        for condition in conditions
        if (
            name := extract_valid_hyperparameters(
                condition, config, config_meta, component_type
            )
        )
        is not None
    ]

    # Create a Terminal for each valid hyperparameter for the component
    component_terminals = [
        Terminal(
            identifier=get_hyperparameter_sklearn_name(param),
            value=value,
            output=get_hyperparameter_sklearn_name(param),
            config_space_name=param,
        )
        for param, value in config.items()
        if param in component_valid_hyperparameters
        and param not in [config_meta.get(key) for key in exclude_keys]
    ]

    return component_primitive, component_terminals


def extract_valid_hyperparameters(
    cond: cs.conditions, config: cs.Configuration, config_meta: dict, meta_key: str
) -> Union[str, None]:
    """Extract valid hyperparameters from a condition.

    For each supported ConfigSpace condition type, extract the valid hyperparameters
    from the condition. The valid hyperparameters are the hyperparameters that are
    valid for the given condition and configuration based on the meta_key.

    Supported ConfigSpace condition types:
    - EqualsCondition
    - AndConjunction

    Readers are encouraged to add support for more ConfigSpace condition types if
    needed. Open an issue or pull request on the GAMA GitHub repository.

    Parameters
    ----------
    cond : cs.conditions
        A condition of type ConfigSpace.
    config : cs.Configuration
        A configuration of type ConfigSpace.
    config_meta : dict
        The meta information of the ConfigSpace.
    meta_key : str
        The meta key of the ConfigSpace.
    """
    if meta_key not in config_meta:
        raise ValueError(f"Meta key {meta_key} not in config_meta")
    if type(cond) not in [cs.conditions.EqualsCondition, cs.conditions.AndConjunction]:
        raise ValueError(
            f"Condition type {type(cond)} not supported. Refer to "
            f"docstring for supported condition types."
        )
    if isinstance(cond, cs.conditions.EqualsCondition):
        if (
            cond.parent.name == config_meta[meta_key]
            and cond.value == config[config_meta[meta_key]]
        ):
            return cond.child.name
    elif isinstance(cond, cs.conditions.AndConjunction):
        for component in cond.components:
            if (
                component.parent.name == config_meta[meta_key]
                and component.value == config[config_meta[meta_key]]
            ):
                return component.child.name
    return None
