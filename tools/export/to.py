import json
import enum
import pathlib

import torch
import torch.onnx

import dinov3.custom_lib.export.exported_program
import dinov3.custom_lib.export.coreml
import dinov3.custom_lib.export.config as export_config
import dinov3.custom_lib.export.models
import dinov3.custom_lib.load.pytorch
"""
works with python3.11 on ubuntu 22
and pytorch 2.10
and coremltools 9.0
NOTE: coremltools will fail on aarch64
"""


class ExportFormat(enum.Enum):
    COREML = enum.auto()
    ONNX = enum.auto()


EXPORTED_PROGRAM_CONFIG = export_config.ExportedProgramConfig(batch_size_min=1,
                                                              batch_size_max=8,
                                                              height_min=384,
                                                              height_max=768,
                                                              width_min=384,
                                                              width_max=768)
COREML_CONFIG = export_config.CoreMLConfig(spatial_shapes=[(512, 256),
                                                           (512, 384),
                                                           (768, 512)],
                                           fp16=False)

OUTPUT_PATH = "convnext_small_with_preprocess.mlpackage"


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Export ConvNext Small with Preprocessing to CoreML")
    parser.add_argument("--format",
                        choices=[f.name for f in ExportFormat],
                        default=ExportFormat.COREML.name,
                        type=str)
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="weights",
        help="Directory to save the exported weights (for ONNX)")
    return parser.parse_args()


def main(export_format: ExportFormat, weights_dir: pathlib.Path) -> None:

    # Load PyTorch Model
    model = dinov3.custom_lib.load.pytorch.load_convnext_small_pretrained_pytorch(
    )
    full_model = dinov3.custom_lib.export.models.ConvNextWithPreProcess(
        backbone_model=model)
    full_model.eval()

    # Export to ExportedProgram with dynamic shapes
    exported_program_model_decomposed, dynamic_shapes = dinov3.custom_lib.export.exported_program.to_exported_program(
        model=full_model,
        batch_size=EXPORTED_PROGRAM_CONFIG.batch_size_range,
        height_range=EXPORTED_PROGRAM_CONFIG.height_range,
        width_range=EXPORTED_PROGRAM_CONFIG.width_range)

    if export_format == ExportFormat.COREML:
        # Prepare CoreML input shape
        ct_input_shape = dinov3.custom_lib.export.coreml.build_ct_input_shape(
            batch_sizes=EXPORTED_PROGRAM_CONFIG.batch_size_range,
            spatial_shapes=COREML_CONFIG.spatial_shapes,
        )

        # Export to CoreML
        coreml_model = dinov3.custom_lib.export.coreml.to_coreml(
            exported_program_model_decomposed, ct_input_shape, COREML_CONFIG)
        output_path = weights_dir / "coreml" / OUTPUT_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        coreml_model.save(output_path)
        print(f"Saved Core ML model to {output_path}")

        # Save export metadata
        with open(output_path / "metadata.json", "w") as f:
            export_info = export_config.get_metadata(
                exported_program_config=EXPORTED_PROGRAM_CONFIG,
                coreml_config=COREML_CONFIG,
                extra_notes=[
                    "mlpackage is integrated with pixel normalization to ImageNet before inference"
                ])
            json.dump(export_info, f)
    elif export_format == ExportFormat.ONNX:
        onnx_path = weights_dir / "onnx" / "convnext_small_with_preprocess.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        INPUT_NAMES = ["x"]
        assert set(dynamic_shapes.keys()) == set(
            INPUT_NAMES), "Dynamic shapes keys must match input names"
        onnx_program = torch.onnx.export(
            exported_program_model_decomposed,
            args=(),
            dynamo=True,
            input_names=["x"],
            output_names=["y"],
        )
        onnx_program.save(onnx_path)


if __name__ == "__main__":
    args = parse_args()
    weights_dir = pathlib.Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    main(ExportFormat[args.format], weights_dir)
