from typing import Any

import torch
import coremltools as ct
from dinov3.custom_lib.export.config import CoreMLConfig, BatchSizeRange, HeightRange, WidthRange


def to_exported_program(
        model: torch.nn.Module, batch_size: BatchSizeRange,
        height_range: HeightRange,
        width_range: WidthRange) -> torch.export.ExportedProgram:
    # Define dynamic dims for export
    B = torch.export.Dim("batch",
                         min=batch_size.minimum,
                         max=batch_size.maximum)
    H = torch.export.Dim("height",
                         min=height_range.minimum,
                         max=height_range.maximum)
    W = torch.export.Dim("width",
                         min=width_range.minimum,
                         max=width_range.maximum)

    dummy_input = torch.randn(batch_size.mid(),
                              3,
                              height_range.mid(),
                              width_range.mid(),
                              dtype=torch.float32)

    dynamic_shapes = {"x": {0: B, 2: H, 3: W}}
    with torch.no_grad():
        exported_program_model = torch.export.export(
            model,
            (dummy_input, ),
            dynamic_shapes=dynamic_shapes,
        )
        return exported_program_model.run_decompositions()


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
