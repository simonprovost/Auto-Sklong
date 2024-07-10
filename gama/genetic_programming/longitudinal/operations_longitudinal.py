import copy
import random
from typing import List, Optional, Any

import ConfigSpace as cs

from gama.genetic_programming.components import (
    Primitive,
    PrimitiveNode,
    DATA_TERMINAL,
)
from gama.genetic_programming.operations import _config_component_to_primitive_node
from gama.utilities.config_space import (
    get_longitudinal_estimator_by_name,
    get_internal_longitudinal_output_types,
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
    is_data_preparation_to_ignore = (
        "data_preparation" in config_space.meta
        and config[config_space.meta["data_preparation"]] == technique_name_to_ignore
    )
    is_preprocessor_to_ignore = (
        "preprocessors" in config_space.meta
        and config[config_space.meta["preprocessors"]] == technique_name_to_ignore
    )
    is_estimator_to_ignore = (
        "estimators" in config_space.meta
        and config[config_space.meta["estimators"]] == technique_name_to_ignore
    )

    if (
        is_data_preparation_to_ignore
        or is_estimator_to_ignore
        or is_preprocessor_to_ignore
    ):
        while (
            is_data_preparation_to_ignore
            or is_estimator_to_ignore
            or is_preprocessor_to_ignore
        ):
            temp_config = config_space.sample_configuration()
            is_data_preparation_to_ignore = (
                "data_preparation" in config_space.meta
                and temp_config[config_space.meta["data_preparation"]]
                == technique_name_to_ignore
            )
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
            if (
                not is_data_preparation_to_ignore
                and not is_estimator_to_ignore
                and not is_preprocessor_to_ignore
            ):
                return temp_config
    return config


def random_longitudinal_primitive_node(
    output_type: str,
    config_space: cs.ConfigurationSpace,
    exclude: Optional[Primitive] = None,
    exclude_options_list: Optional[List[str]] = None,
) -> PrimitiveNode:
    """Create a PrimitiveNode with specified output_type and random terminals.

    copy from gama/genetic_programming/operations.py with some modification, which are:
    1. We use longitudinal output types check instead of normal output types check
    2. We now support exclude_options_list, which is a list of options that we want to
    exclude from the search space but cannot be represented within ConfigSpace given
    other conditions. We then repetitively sample from the search space until we get
    a valid option, which is always possible to find.
    3. _config_component_to_primitive_node uses a different exclude_keys list, which
    is ["data_preparation", "estimators", "preprocessors"] instead of
    ["estimators", "preprocessors"]
    """
    if output_type not in get_internal_longitudinal_output_types():
        raise ValueError(f"Output type {output_type} not supported")
    excludes = exclude_options_list.copy() if exclude_options_list else None

    if exclude is not None or excludes is not None:
        temp_config_space = copy.deepcopy(config_space)
        if output_type not in temp_config_space.meta:
            raise ValueError(f"Output type {output_type} not in config_space meta.")

        exclude_options_path = True
        if excludes:
            if exclude is not None:
                excludes.append(exclude.__str__())
            try:
                temp_config_space.add_forbidden_clause(
                    cs.ForbiddenInClause(
                        temp_config_space.get_hyperparameter(
                            temp_config_space.meta[output_type]
                        ),
                        excludes,
                    )
                )
                exclude_options_path = False
            except ValueError:
                exclude_options_path = True
        elif exclude:
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
        if exclude_options_path and excludes:
            while config[config_space.meta[output_type]] in excludes:
                config = _sample_and_ignore_dummy_techniques(
                    temp_config_space, temp_config_space.sample_configuration()
                )

    else:
        config = _sample_and_ignore_dummy_techniques(
            config_space, config_space.sample_configuration()
        )

    (
        data_preparation_primitive,
        data_preparation_terminals,
    ) = _config_component_to_primitive_node(
        "data_preparation",
        config,
        config_space.meta,
        config_space.get_conditions(),
        ["data_preparation", "estimators", "preprocessors"],
        retrieve_estimator=get_longitudinal_estimator_by_name,
    )
    if output_type == "data_preparation":
        return PrimitiveNode(
            data_preparation_primitive,
            data_node=DATA_TERMINAL,
            terminals=data_preparation_terminals,
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
            ["data_preparation", "estimators", "preprocessors"],
            retrieve_estimator=get_longitudinal_estimator_by_name,
        )
        data_preparation_node = PrimitiveNode(
            data_preparation_primitive,
            data_node=DATA_TERMINAL,
            terminals=data_preparation_terminals,
        )
        return PrimitiveNode(
            preprocessor_primitive,
            data_node=data_preparation_node,
            terminals=preprocessor_terminals,
        )
    estimator_primitive, estimator_terminals = _config_component_to_primitive_node(
        "estimators",
        config,
        config_space.meta,
        config_space.get_conditions(),
        ["data_preparation", "estimators", "preprocessors"],
        retrieve_estimator=get_longitudinal_estimator_by_name,
    )
    data_preparation_node = PrimitiveNode(
        data_preparation_primitive,
        data_node=DATA_TERMINAL,
        terminals=data_preparation_terminals,
    )
    return PrimitiveNode(
        estimator_primitive,
        data_node=data_preparation_node,
        terminals=estimator_terminals,
    )


def create_longitudinal_random_expression(
    config_space: cs.ConfigurationSpace,
    min_length: int = 2,
    max_length: int = 3,
) -> PrimitiveNode:
    """Create at least min_length and at most max_length chained PrimitiveNodes.

    copy from gama/genetic_programming/operations.py with some modification, which are:
    1. Min length is 2 instead of 1, because we need at least one data preparation
    and one estimator.
    """
    individual_length = random.randint(min_length, max_length)
    config = _sample_and_ignore_dummy_techniques(
        config_space, config_space.sample_configuration()
    )
    return _config_to_longitudinal_primitive_node(
        config=config,
        config_meta=config_space.meta,
        conditions=config_space.get_conditions(),
        config_length=individual_length,
    )


def _config_to_longitudinal_primitive_node(
    config: cs.Configuration,
    config_meta: dict,
    conditions: List[Any],
    config_length: Optional[int] = None,
) -> PrimitiveNode:
    """Create a PrimitiveNode from a configuration. If config_length is specified, the
    PrimitiveNode will have at most config_length PrimitiveNodes.

    copy from gama/genetic_programming/operations.py with some modification, which are:
    1. We prepare the data_preparation and estimator first.
    2. We use _config_component_to_primitive_node with a different exclude_keys list,
    which is ["data_preparation", "estimators", "preprocessors"] instead of
    ["estimators", "preprocessors"]
    3. If the preprocessor option is None, we don't create a preprocessor node.
    4. If config_length is 2, we don't create a preprocessor node.
    """
    (
        data_preparation_primitive,
        data_preparation_terminals,
    ) = _config_component_to_primitive_node(
        "data_preparation",
        config,
        config_meta,
        conditions,
        ["data_preparation", "estimators", "preprocessors"],
        retrieve_estimator=get_longitudinal_estimator_by_name,
    )
    estimator_primitive, estimator_terminals = _config_component_to_primitive_node(
        "estimators",
        config,
        config_meta,
        conditions,
        ["data_preparation", "estimators", "preprocessors"],
        retrieve_estimator=get_longitudinal_estimator_by_name,
    )
    data_preparation_node = PrimitiveNode(
        data_preparation_primitive,
        data_node=DATA_TERMINAL,
        terminals=data_preparation_terminals,
    )
    if isinstance(config_length, int) and config_length <= 2:
        # Create a PrimitiveNode for the data_preparation
        return PrimitiveNode(
            estimator_primitive,
            data_node=data_preparation_node,
            terminals=estimator_terminals,
        )
    preprocessor_node = None
    if config[config_meta["preprocessors"]] != "None":
        (
            preprocessor_primitive,
            preprocessor_terminals,
        ) = _config_component_to_primitive_node(
            "preprocessors",
            config,
            config_meta,
            conditions,
            ["data_preparation", "estimators", "preprocessors"],
            retrieve_estimator=get_longitudinal_estimator_by_name,
        )
        # Create a PrimitiveNode for the preprocessor and chain it to the
        # data_preparation
        preprocessor_node = PrimitiveNode(
            preprocessor_primitive,
            data_node=data_preparation_node,
            terminals=preprocessor_terminals,
        )

    # Create a PrimitiveNode for the classifier and chain it to the preprocessor
    return PrimitiveNode(
        estimator_primitive,
        data_node=preprocessor_node or data_preparation_node,
        terminals=estimator_terminals,
    )
