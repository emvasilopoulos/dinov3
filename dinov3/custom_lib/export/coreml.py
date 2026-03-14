from typing import Any

import torch
import coremltools as ct
from dinov3.custom_lib.export.config import CoreMLConfig, BatchSizeRange


def to_coreml(exported_program: torch.export.ExportedProgram,
              ct_input_shape: ct.EnumeratedShapes,
              coreml_config: CoreMLConfig) -> Any:
    return ct.convert(
        exported_program,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="input",
                          shape=ct_input_shape,
                          dtype=coreml_config.tensor_type)
        ],
        outputs=[
            ct.TensorType(name="output", dtype=coreml_config.tensor_type)
        ],
        compute_precision=coreml_config.model_compute_precision,
    )


def build_ct_input_shape(
        batch_sizes: BatchSizeRange,
        spatial_shapes: list[tuple[int, int]]) -> ct.EnumeratedShapes:
    shapes: list[list[int]] = []
    for height, width in spatial_shapes:
        for batch_size in range(batch_sizes.minimum, batch_sizes.maximum):
            shapes.append([batch_size, 3, height, width])
    return ct.EnumeratedShapes(shapes=shapes)
