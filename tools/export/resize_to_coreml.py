"""
FAILS due to ResizePreprocess > F.interpolate module --> Not Supported in CoreML.
"""
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
import coremltools as ct


class ResizePreprocess(nn.Module):

    def __init__(
            self,
            resize_width: int = 768,
            resize_height: int = 1024,
            patch_size: int = 16,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
    ):
        super().__init__()

        self.resize_width = resize_width
        self.resize_height = resize_height
        self.patch_size = patch_size

        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, 3, H, W) in range [0, 1] as float32

        Returns:
            Preprocessed tensor ready for model input
        """
        x = (x - self.mean) / self.std

        # Resize (bilinear matches torchvision default for RGB)
        x = F.interpolate(
            x,
            size=(self.resize_height, self.resize_width),
            mode="bilinear",
            align_corners=False,
        )
        return x


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export DINOv3 ConvNeXt to Core ML.")
    parser.add_argument("--input-width",
                        type=int,
                        default=384,
                        help=f"Input image width")
    parser.add_argument("--input-height",
                        type=int,
                        default=768,
                        help=f"Input image height")
    parser.add_argument("--resize-width",
                        type=int,
                        default=384,
                        help=f"Resize image width.")
    parser.add_argument("--resize-height",
                        type=int,
                        default=512,
                        help=f"Resize image height.")
    return parser.parse_args()


def main():
    args = parse_args()

    full_model = ResizePreprocess(
        resize_height=args.resize_height,
        resize_width=args.resize_width,
    )
    full_model.eval()

    # Pick any representative size within your intended ranges
    dummy = torch.randn(3, 3, 512, 384, dtype=torch.float32)

    # Define dynamic dims for export
    B = torch.export.Dim("batch", min=1, max=4)
    H = torch.export.Dim("height", min=384, max=1024)
    W = torch.export.Dim("width", min=384, max=1024)
    dynamic_shapes = {"x": {0: B, 2: H, 3: W}}
    with torch.no_grad():
        exported_program_model = torch.export.export(
            full_model,
            (dummy, ),
            dynamic_shapes=dynamic_shapes,
        )
        exported_program_model_decomposed = exported_program_model.run_decompositions(
        )

    coreml_model = ct.convert(
        exported_program_model_decomposed,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )

    output_path = "convnext_small_with_preprocess.mlpackage"
    coreml_model.save(output_path)
    print(f"Saved Core ML model to {output_path}")
    with open("convnext_small_with_preprocess.mlpackage/metadata.json",
              "w") as f:
        resize_info = {
            "input_width": args.input_width,
            "input_height": args.input_height,
            "resize_width": args.resize_width,
            "resize_height": args.resize_height,
            "expected_input_range": [0.0, 1.0],
            "expected_input_dtype": "float32",
        }
        json.dump(resize_info, f)


if __name__ == "__main__":
    main()
