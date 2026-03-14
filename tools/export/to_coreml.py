import dataclasses
import json

import torch
import torch.nn as nn

import dinov3.custom_lib.utils
import dinov3.custom_lib.export.coreml
import dinov3.custom_lib.export.config as export_config
"""
works with python3.11 on ubuntu 22
and pytorch 2.10
and coremltools 9.0
NOTE: coremltools will fail on aarch64
"""

EXPORTED_PROGRAM_CONFIG = export_config.ExportedProgramConfig(batch_size_min=1,
                                                              batch_size_max=8,
                                                              height_min=384,
                                                              height_max=768,
                                                              width_min=384,
                                                              width_max=768)
COREML_CONFIG = export_config.CoreMLConfig(spatial_shapes=[(512, 384),
                                                           (768, 512)],
                                           fp16=False)
OUTPUT_PATH = "convnext_small_with_preprocess.mlpackage"


class ConvNextWithPreProcess(nn.Module):

    def __init__(self, backbone_model: torch.nn.Module) -> None:
        super().__init__()
        self.model = backbone_model

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.model(x)


def main() -> None:

    # Load PyTorch Model
    model = dinov3.custom_lib.utils.load_convnext_small_pretrained_pytorch()
    full_model = ConvNextWithPreProcess(backbone_model=model)
    full_model.eval()

    # Export to ExportedProgram with dynamic shapes
    exported_program_model_decomposed = dinov3.custom_lib.export.coreml.to_exported_program(
        model=full_model,
        batch_size=EXPORTED_PROGRAM_CONFIG.batch_size_range,
        height_range=EXPORTED_PROGRAM_CONFIG.height_range,
        width_range=EXPORTED_PROGRAM_CONFIG.width_range)

    # Prepare CoreML input shape
    ct_input_shape = dinov3.custom_lib.export.coreml.build_ct_input_shape(
        batch_sizes=EXPORTED_PROGRAM_CONFIG.batch_size_range,
        spatial_shapes=COREML_CONFIG.spatial_shapes,
    )

    # Export to CoreML
    coreml_model = dinov3.custom_lib.export.coreml.to_coreml(
        exported_program_model_decomposed, ct_input_shape, COREML_CONFIG)
    coreml_model.save(OUTPUT_PATH)
    print(f"Saved Core ML model to {OUTPUT_PATH}")

    # Save export metadata
    with open("convnext_small_with_preprocess.mlpackage/metadata.json",
              "w") as f:
        export_info = export_config.get_metadata(
            exported_program_config=EXPORTED_PROGRAM_CONFIG,
            coreml_config=COREML_CONFIG,
            extra_notes=[
                "mlpackage is integrated with pixel normalization to ImageNet before inference"
            ])
        json.dump(export_info, f)


if __name__ == "__main__":
    main()
