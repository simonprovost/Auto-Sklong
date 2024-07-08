from enum import Enum
from typing import List, Tuple, Any, Union, Optional

import importlib


GamaPipelineTypeUnion = Union[
    "sklearn.pipeline.Pipeline",  # type: ignore # noqa: F821
    "scikit_longitudinal.pipeline.LongitudinalPipeline",  # type: ignore # noqa: F821
]


class GamaPipelineType(Enum):
    ScikitLearn = ("sklearn.pipeline", "Pipeline")
    ScikitLongitudinal = ("scikit_longitudinal.pipeline", "LongitudinalPipeline")

    def import_pipeline_class(self):
        module_name, class_name = self.value
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"Could not import module {module_name}") from e
        return getattr(module, class_name)

    @property
    def required_import(self) -> str:
        module_name, class_name = self.value
        if class_name != "Pipeline":
            return f"from {module_name} import {class_name} as Pipeline"
        return f"from {module_name} import {class_name}"


class GamaPipeline:
    def __new__(  # type: ignore
        cls,
        steps: List[Tuple[str, Any]],
        pipeline_type: Optional[GamaPipelineType] = None,
        *args,
        **kwargs,
    ) -> GamaPipelineTypeUnion:
        if steps is None or not steps:
            raise ValueError("Pipeline steps cannot be None or empty")

        if pipeline_type is None:
            raise ValueError("Pipeline type cannot be None")
        PipelineClass = pipeline_type.import_pipeline_class()
        return PipelineClass(steps, *args, **kwargs)
