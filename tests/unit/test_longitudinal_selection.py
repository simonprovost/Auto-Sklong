from gama.genetic_programming.components import Fitness

from gama.genetic_programming.longitudinal.selection_longitudinal import (
    create_longitudinal_from_population,
)


def test_create_longitudinal_from_population(
    opset, Longitudinal_DT, LongitudinalForestPipeline, LongitudinalLinearSVC
):
    Longitudinal_DT.fitness = Fitness((3, -2), 0, 0, 0)
    LongitudinalForestPipeline.fitness = Fitness((4, -2), 0, 0, 0)
    LongitudinalLinearSVC.fitness = Fitness((3, -1), 0, 0, 0)
    parents = [Longitudinal_DT, LongitudinalForestPipeline, LongitudinalLinearSVC]

    new = create_longitudinal_from_population(
        opset, pop=parents, n=1, cxpb=0.5, mutpb=0.5
    )
    assert 1 == len(new)
    assert new[0]._id not in map(lambda i: i._id, parents)
    assert new[0].pipeline_str() not in map(lambda i: i.pipeline_str(), parents)

    # Not sure how to test NSGA2 selection is applied correctly
    # Can do it many times and see if the best individuals are parent more
    # With these fixtures, crossover can't be tested either.
