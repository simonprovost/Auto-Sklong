import pytest

from gama.genetic_programming.crossover import (
    crossover_primitives,
    _shared_terminals,
    crossover_terminals,
)
from gama.genetic_programming.longitudinal.crossover_longitudinal import (
    random_longitudinal_crossover,
)


def test_shared_longitudinal_terminals(
    Longitudinal_DT, Longitudinal_CFS_DT, LongitudinalLinearSVC
):
    """Test shared terminals are found, if they exist."""
    assert 1 == len(
        list(_shared_terminals(Longitudinal_DT, Longitudinal_DT, value_match="equal"))
    )
    assert 1 == len(
        list(_shared_terminals(Longitudinal_DT, Longitudinal_CFS_DT, value_match="all"))
    )
    assert 0 == len(
        list(
            _shared_terminals(
                Longitudinal_DT, LongitudinalLinearSVC, value_match="different"
            )
        )
    )


def test_crossover_longitudinal_primitives(Longitudinal_DT, Longitudinal_CFS_DT):
    """Two individuals of at least length 2 produce two new ones with crossover."""
    ind1_copy, ind2_copy = (
        Longitudinal_DT.copy_as_new(),
        Longitudinal_CFS_DT.copy_as_new(),
    )

    # Cross-over is in-place
    crossover_primitives(Longitudinal_DT, Longitudinal_CFS_DT)
    # Both parents and children should be unique
    all_individuals = [Longitudinal_DT, Longitudinal_CFS_DT, ind1_copy, ind2_copy]

    assert 4 == len({ind.pipeline_str() for ind in all_individuals})
    assert ind1_copy.pipeline_str() != Longitudinal_DT.pipeline_str()


def test_crossover_longitudinal_terminal(Longitudinal_DT, Longitudinal_CFS_DT):
    """Two individuals with shared Terminals produce two new ones with crossover."""
    ind1_copy, ind2_copy = (
        Longitudinal_DT.copy_as_new(),
        Longitudinal_CFS_DT.copy_as_new(),
    )
    # Cross-over is in-place
    crossover_terminals(Longitudinal_DT, Longitudinal_CFS_DT)
    # Both parents and children should be unique
    all_individuals = [Longitudinal_DT, Longitudinal_CFS_DT, ind1_copy, ind2_copy]

    assert 4 == len({ind.pipeline_str() for ind in all_individuals})
    assert ind1_copy.pipeline_str() != Longitudinal_DT.pipeline_str()


def test_longitudinal_crossover(Longitudinal_DT, Longitudinal_CFS_DT):
    """Two eligible individuals should produce two new individuals with crossover."""
    ind1_copy, ind2_copy = (
        Longitudinal_DT.copy_as_new(),
        Longitudinal_CFS_DT.copy_as_new(),
    )
    # Cross-over is in-place
    random_longitudinal_crossover(Longitudinal_DT, Longitudinal_CFS_DT)
    # Both parents and children should be unique
    all_individuals = [Longitudinal_DT, Longitudinal_CFS_DT, ind1_copy, ind2_copy]
    assert 4 == len({ind.pipeline_str() for ind in all_individuals})
    assert ind1_copy.pipeline_str() != Longitudinal_DT.pipeline_str()


def test_longitudinal_crossover_max_length_exceeded(
    Longitudinal_CFS_DT, Longitudinal_DT
):
    """Raise ValueError if either provided individual exceeds `max_length`."""
    with pytest.raises(ValueError) as _:
        random_longitudinal_crossover(
            Longitudinal_CFS_DT, Longitudinal_DT, max_length=2
        )

    with pytest.raises(ValueError) as _:
        random_longitudinal_crossover(
            Longitudinal_DT, Longitudinal_CFS_DT, max_length=2
        )


def test_longitudinal_crossover_max_length(Longitudinal_CFS_DT):
    """Setting `max_length` affects only maximum produced length."""
    primitives_in_parent = len(Longitudinal_CFS_DT.primitives)
    produced_lengths = []
    for _ in range(60):  # guarantees all length pipelines are produced with prob >0.999
        ind1, ind2 = random_longitudinal_crossover(
            Longitudinal_CFS_DT.copy_as_new(),
            Longitudinal_CFS_DT.copy_as_new(),
            max_length=primitives_in_parent,
        )
        # Only the first child is guaranteed to contain at most `max_length` primitives.
        produced_lengths.append(len(ind1.primitives))
    assert {2, 3} == set(produced_lengths)
