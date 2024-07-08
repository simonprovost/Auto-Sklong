""" Functions which take two Individuals and produce at least one new Individual. """
import random
from typing import List, Callable, Optional, Tuple

from gama.configuration.longitudinal_classification import LongitudinalConstraints
from gama.genetic_programming.crossover import (
    crossover_terminals,
    _shared_terminals,
    crossover_primitives,
)

from gama.genetic_programming.components import Individual


def random_longitudinal_crossover(
    ind1: Individual, ind2: Individual, max_length: Optional[int] = None
) -> Tuple[Individual, Individual]:
    """Random valid crossover between two individuals in-place, if it can be done.

    copy from gama/genetic_programming/crossover.py with some modification, which are:
    1. Now we use valid_longitudinal_crossover_functions instead of
    valid_crossover_functions
    """
    if max_length is not None and len(ind1.primitives) > max_length:
        raise ValueError(f"`individual1` ({ind1}) exceeds `max_length` ({max_length}).")
    if max_length is not None and len(ind2.primitives) > max_length:
        raise ValueError(f"`individual2` ({ind2}) exceeds `max_length` ({max_length}).")

    crossover_choices = _valid_longitudinal_crossover_functions(ind1, ind2)
    if len(crossover_choices) == 0:
        raise ValueError(f"{ind1.pipeline_str()} and {ind2.pipeline_str()} can't mate.")
    ind1, ind2 = random.choice(crossover_choices)(ind1, ind2)
    if max_length is not None and len(ind1.primitives) > max_length:
        return ind2, ind1
    return ind1, ind2


def _valid_longitudinal_crossover_functions(
    ind1: Individual, ind2: Individual
) -> List[Callable]:
    """Find all crossover functions that can produce new individuals from this input.

    copy from gama/genetic_programming/crossover.py with some modification, which are:
    1. We added one more check to see if the two longitudinal individuals can be
    crossovered, refer to the following class for further infroamtion:
    gama/genetic_programming/longitudinal/constraints_longitudinal.py
    """
    crossover_choices = []
    if list(_shared_terminals(ind1, ind2)):
        crossover_choices.append(crossover_terminals)
    if (
        len(list(ind1.primitives)) >= 2
        and len(list(ind2.primitives)) >= 2
        and LongitudinalConstraints.can_crossover(ind1, ind2)
    ):
        crossover_choices.append(crossover_primitives)
    return crossover_choices
