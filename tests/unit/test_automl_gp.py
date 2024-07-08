def test_individual_length(GNB, ForestPipeline, LinearSVC):
    assert 1 == len(list(GNB.primitives))
    assert 2 == len(list(ForestPipeline.primitives))
    assert 1 == len(list(LinearSVC.primitives))


def test_longitudinal_individual_length(
    Longitudinal_DT,
    Longitudinal_CFS_DT,
    LongitudinalLinearSVC,
    LongitudinalForestPipeline,
):
    assert 2 == len(list(Longitudinal_DT.primitives))
    assert 3 == len(list(Longitudinal_CFS_DT.primitives))
    assert 2 == len(list(LongitudinalLinearSVC.primitives))
    assert 2 == len(list(LongitudinalForestPipeline.primitives))
