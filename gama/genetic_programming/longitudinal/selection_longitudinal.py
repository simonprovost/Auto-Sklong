""" Selection operators. """
import random
from typing import List

from gama.genetic_programming.components import Individual
from .crossover_longitudinal import _valid_longitudinal_crossover_functions
from gama.genetic_programming.nsga2 import nsga2_select
from gama.genetic_programming.operator_set import OperatorSet


def create_longitudinal_from_population(
    operator_shell: OperatorSet,
    pop: List[Individual],
    n: int,
    cxpb: float,
    mutpb: float,
) -> List[Individual]:
    """Creates n new individuals based on the population.

    copy from gama/genetic_programming/selection.py with some modification, which are:
    1. We now use valid_longitudinal_crossover_functions instead of
    valid_crossover_functions.
    """
    offspring = []
    metrics = [lambda ind: ind.fitness.values[0], lambda ind: ind.fitness.values[1]]
    parent_pairs = nsga2_select(pop, n, metrics)
    for ind1, ind2 in parent_pairs:
        if (
            random.random() < cxpb
            and len(_valid_longitudinal_crossover_functions(ind1, ind2)) > 0
        ):
            ind1 = operator_shell.mate(ind1, ind2)
        else:
            ind1 = operator_shell.mutate(ind1)
        offspring.append(ind1)
    return offspring
