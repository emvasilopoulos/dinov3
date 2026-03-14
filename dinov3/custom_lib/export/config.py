import dataclasses
from typing import NamedTuple, List

import coremltools as ct

SpatialShape = NamedTuple("SpatialShape", [("height", int), ("width", int)])


@dataclasses.dataclass
class BaseMinMax:
    minimum: int
    maximum: int

    def mid(self) -> int:
        return (self.minimum + self.maximum) // 2


class BatchSizeRange(BaseMinMax):
    pass


class HeightRange(BaseMinMax):
    pass


class WidthRange(BaseMinMax):
    pass


class ExportedProgramConfig:

    def __init__(self, batch_size_min: int, batch_size_max: int,
                 height_min: int, height_max: int, width_min: int,
                 width_max: int):

        self.batch_size_range = BatchSizeRange(minimum=batch_size_min,
                                               maximum=batch_size_max)
        self.height_range = HeightRange(minimum=height_min, maximum=height_max)
        self.width_range = WidthRange(minimum=width_min, maximum=width_max)


class CoreMLConfig:

    def __init__(self, spatial_shapes: List[SpatialShape], fp16: bool):
        self.spatial_shapes = spatial_shapes
        self.tensor_type = ct.converters.mil.mil.types.fp32 if not fp16 else ct.converters.mil.mil.types.fp16
        self.model_compute_precision = ct.precision.FLOAT32 if not fp16 else ct.precision.FLOAT16


def _serialize_min_max_range(range_config: BaseMinMax):
    if dataclasses.is_dataclass(range_config):
        return dataclasses.asdict(range_config)
    return {
        "minimum": range_config.minimum,
        "maximum": range_config.maximum,
    }


def get_metadata(exported_program_config: ExportedProgramConfig,
                 coreml_config: CoreMLConfig,
                 extra_notes: List[str] = []):
    return {
        "exported_program_config": {
            "batch_size_range":
            _serialize_min_max_range(exported_program_config.batch_size_range),
            "height_range":
            _serialize_min_max_range(exported_program_config.height_range),
            "width_range":
            _serialize_min_max_range(exported_program_config.width_range),
        },
        "coreml_config": {
            "convert_to":
            "mlprogram",
            "batch_size_range": {
                "minimum": exported_program_config.batch_size_range.minimum,
                "maximum": exported_program_config.batch_size_range.maximum,
            },
            "spatial_shapes":
            [list(shape) for shape in coreml_config.spatial_shapes],
            "input_name":
            "input",
            "output_name":
            "output",
            "tensor_type":
            str(coreml_config.tensor_type),
            "model_compute_precision":
            str(coreml_config.model_compute_precision),
        },
        "notes": extra_notes
    }
