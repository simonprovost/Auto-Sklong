import pytest

from gama.GamaPipeline import GamaPipelineType, GamaPipeline

happy_path_values = [
    (GamaPipelineType.ScikitLearn, "sklearn.pipeline.Pipeline"),
    (
        GamaPipelineType.ScikitLongitudinal,
        "scikit_longitudinal.pipeline.LongitudinalPipeline",
    ),
]

error_case_values = [
    (GamaPipelineType.ScikitLearn, "NonExistentPipeline"),
    (GamaPipelineType.ScikitLongitudinal, "NonExistentPipeline"),
]


@pytest.mark.parametrize("pipeline_type, expected_import_str", happy_path_values)
def test_required_import_happy_path(pipeline_type, expected_import_str):
    result_import_str = pipeline_type.required_import
    import_module = expected_import_str.split(".")[0]
    assert import_module in result_import_str


@pytest.mark.parametrize("pipeline_type, expected_class_str", happy_path_values)
def test_import_pipeline_class_happy_path(pipeline_type, expected_class_str):
    result_class = pipeline_type.import_pipeline_class()
    assert f"{result_class.__module__}.{result_class.__name__}" == expected_class_str


def test_no_steps_gama_pipeline_error_case():
    with pytest.raises(ValueError):
        GamaPipeline(
            steps=[],
        )
