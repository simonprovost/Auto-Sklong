"""
Contains mutation functions for genetic programming.
Each mutation function takes an individual and modifies it in-place.
"""
import logging
import random
from functools import partial
from typing import Callable, Optional, List, Dict, cast

import ConfigSpace as cs

from gama.genetic_programming.components import PrimitiveNode
from .constraints_longitudinal import LongitudinalSearchSpaceConstraint
from ..mutation import has_multiple_options, mut_replace_terminal
from ..components import Individual
from .operations_longitudinal import random_longitudinal_primitive_node
from ...configuration.longitudinal_classification import LongitudinalConstraints

# Avoid stopit from logging warnings every time a pipeline evaluation times out
logging.getLogger("stopit").setLevel(logging.ERROR)
log = logging.getLogger(__name__)


def mut_longitudinal_shrink(
    individual: Individual,
    _config_space: Optional[cs.ConfigurationSpace] = None,
    shrink_by: Optional[int] = None,
) -> None:
    """Mutates an Individual in-place by removing any number of Longitudinal
    primitive nodes.

    copy from gama/genetic_programming/mutation.py with some modification, which are:
    1. Shrink by is now between 1 and n_primitives - 2, instead of 1 and
    n_primitives - 1 – This is because we need at least one data preparation node
    and one estimator node
    2. We do not complete the individual by a DATA_TERMINAL, or nothing, we add at the
    end of the shrink, the data preparation node initially available in the individual
    """
    n_primitives = len(list(individual.primitives))
    if shrink_by is not None and n_primitives <= shrink_by:
        raise ValueError(f"Can't shrink size {n_primitives} individual by {shrink_by}.")
    if shrink_by is None:
        shrink_by = random.randint(1, n_primitives - 2)

    data_preparation_node = (
        LongitudinalSearchSpaceConstraint.find_data_preparation_node(individual)
    )
    current_primitive_node = individual.main_node
    primitives_left = n_primitives - 2
    while primitives_left > shrink_by:
        current_primitive_node = cast(PrimitiveNode, current_primitive_node._data_node)
        primitives_left -= 1
    current_primitive_node._data_node = data_preparation_node


def mut_replace_longitudinal_primitive(
    individual: Individual, config_space: cs.ConfigurationSpace
) -> None:
    """Mutates an Individual in-place by replacing one of its Primitives.

    copy from gama/genetic_programming/mutation.py with some modification, which are:
    1. We use LongitudinalSearchSpaceConstraint.primitive_replaceable instead of
    primitive_replaceable. This allows to take into account constraints on the
    longitudinal search space. Further information can be found in the following class:
    gama/genetic_programming/longitudinal/constraints_longitudinal.py
    2. We use random_longitudinal_primitive_node instead of random_primitive_node.
    """

    if not (
        primitives := list(
            filter(
                lambda primitive: (
                    LongitudinalSearchSpaceConstraint.primitive_replaceable(
                        primitive, config_space
                    )
                ),
                enumerate(individual.primitives),
            )
        )
    ):
        raise ValueError("Individual has no primitives suitable for replacement.")

    primitive_index, old_primitive_node = random.choice(primitives)
    exclude_options_list = LongitudinalConstraints.get_exclusions_for_node(
        individual, old_primitive_node._primitive.output
    )

    primitive_node = random_longitudinal_primitive_node(
        output_type=old_primitive_node._primitive.output,
        config_space=config_space,
        exclude=old_primitive_node._primitive,
        exclude_options_list=exclude_options_list,
    )
    individual.replace_primitive(primitive_index, primitive_node)


def mut_longitudinal_insert(
    individual: Individual, config_space: cs.ConfigurationSpace
) -> None:
    """Mutate an Individual in-place by inserting a PrimitiveNode at a random location.

    copy from gama/genetic_programming/mutation.py with some modification, which are:
    1. We use LongitudinalConstraints to get the exclusion list for the new primitive
    node. Further information can be found in the following class:
    gama/genetic_programming/longitudinal/constraints_longitudinal.py
    2. We use random_longitudinal_primitive_node instead of random_primitive_node.
    """
    parent_node = individual.main_node
    data_preparation_node = parent_node._data_node

    exclude_options_list = LongitudinalConstraints.get_exclusions_for_node(
        individual, "preprocessors"
    )

    new_primitive_node = random_longitudinal_primitive_node(
        output_type="preprocessors",
        config_space=config_space,
        exclude_options_list=exclude_options_list,
    )

    new_primitive_node._data_node = data_preparation_node
    parent_node._data_node = new_primitive_node


def random_longitudinal_valid_mutation_in_place(
    individual: Individual,
    config_space: cs.ConfigurationSpace,
    max_length: Optional[int] = None,
) -> Callable:
    """Apply a random valid mutation in place.

    copy from gama/genetic_programming/mutation.py with some modification.
    """
    n_primitives = len(list(individual.primitives))
    available_mutations: List[Callable[[Individual, Dict], None]] = []
    if max_length is not None and n_primitives > max_length:
        available_mutations.append(
            partial(mut_longitudinal_shrink, shrink_by=n_primitives - max_length)
        )
    else:
        replaceable_primitives = list(
            filter(
                lambda primitive: (
                    LongitudinalSearchSpaceConstraint.primitive_replaceable(
                        primitive, config_space
                    )
                ),
                enumerate(individual.primitives),
            )
        )

        if len(list(replaceable_primitives)) > 1:
            available_mutations.append(mut_replace_longitudinal_primitive)

        if max_length is None or n_primitives < max_length:
            available_mutations.append(mut_longitudinal_insert)
        if n_primitives > 2:
            available_mutations.append(mut_longitudinal_shrink)

        replaceable_terminals = filter(
            lambda t: has_multiple_options(
                config_space.get_hyperparameter(t.config_space_name)
            ),
            individual.terminals,
        )
        if len(list(replaceable_terminals)) > 1:
            available_mutations.append(mut_replace_terminal)

    if not available_mutations:
        log.warning(
            f"Individual {individual} has no valid mutations. "
            f"Returning original individual."
        )
        return lambda ind, config: ind

    mut_fn = random.choice(available_mutations)
    mut_fn(individual, config_space)
    return mut_fn
