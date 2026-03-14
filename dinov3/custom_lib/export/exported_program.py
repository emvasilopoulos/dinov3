import torch
from dinov3.custom_lib.export.config import BatchSizeRange, HeightRange, WidthRange

DynamicShapes = dict[str, dict[int, torch.export.Dim]]


def to_exported_program(
    model: torch.nn.Module, batch_size: BatchSizeRange,
    height_range: HeightRange, width_range: WidthRange
) -> tuple[torch.export.ExportedProgram, DynamicShapes]:
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

    dynamic_shapes: DynamicShapes = {"x": {0: B, 2: H, 3: W}}
    with torch.no_grad():
        exported_program_model = torch.export.export(
            model,
            (dummy_input, ),
            dynamic_shapes=dynamic_shapes,
        )
        return exported_program_model.run_decompositions(), dynamic_shapes
