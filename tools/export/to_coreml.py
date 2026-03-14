import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import numpy as np

import dinov3.models.convnext

"""
works with python3.11 on ubuntu 22
and pytorch 2.10
and coremltools 9.0
NOTE: coremltools will fail on aarch64
"""

HELP = """
def resize_transform(
    mask_image: Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))

    return TF.to_tensor(
        TF.resize(mask_image,
                  (h_patches * patch_size, w_patches * patch_size)))
for an image of shape (3, 600, 300), patch_size=16 & image_size=768 the resize will be (3, 384, 768)
"""

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

        self.mean_r = mean[0]
        self.mean_g = mean[1]
        self.mean_b = mean[2]
        self.std_r = std[0]
        self.std_g = std[1]
        self.std_b = std[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, 3, H, W) in range [0, 1] as float32

        Returns:
            Preprocessed tensor ready for model input
        """

        # Resize (bilinear matches torchvision default for RGB)
        x = F.interpolate(
            x,
            size=(self.resize_height, self.resize_width),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize
        x[:,0,:,:] = (x[:,0,:,:] - self.mean_r) / self.std_r
        x[:,1,:,:] = (x[:,1,:,:] - self.mean_g) / self.std_g
        x[:,2,:,:] = (x[:,2,:,:] - self.mean_b) / self.std_b
        
        return x


class ConvNextWithPreProcess(nn.Module):

    def __init__(self, backbone_model, resize_width=384, resize_height=512):
        super().__init__()
        self.preprocess = ResizePreprocess(resize_width=resize_width, resize_height=resize_height)
        self.model = backbone_model

    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Export DINOv3 ConvNeXt to Core ML.")
    parser.add_argument("--input-width", type=int, default=384, help=f"Input image width")
    parser.add_argument("--input-height", type=int, default=768, help=f"Input image height")
    parser.add_argument("--resize-width", type=int, default=384, help=f"Resize image width. To understand the image width run each step of {HELP} by hand")
    parser.add_argument("--resize-height", type=int, default=512, help=f"Resize image height. To understand the image height run each step of {HELP} by hand")
    return parser.parse_args()


def main():
    args = parse_args()

    small_convnext = dinov3.models.convnext.get_convnext_arch("convnext_small")
    model = small_convnext(
        patch_size=16,
        drop_path_rate=0.0,
    )

    cpkt = torch.load(
        "weights/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
        map_location="cpu",
    )
    model.load_state_dict(cpkt, strict=True)

    full_model = ConvNextWithPreProcess(
        model,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
    )
    full_model.eval()

    # Pick any representative size within your intended ranges
    dummy = torch.randn(3, 3, 512, 384, dtype=torch.float32)

    # Define dynamic dims for export
    B = torch.export.Dim("batch", min=1, max=4)
    H = torch.export.Dim("height", min=384, max=1024)
    W = torch.export.Dim("width",  min=384, max=1024)
    dynamic_shapes = {
        "x": {0: B, 2: H, 3: W}
    }
    with torch.no_grad():
        exported_program_model = torch.export.export(
            full_model,
            (dummy,),
            dynamic_shapes=dynamic_shapes,
        )
        exported_program_model_decomposed = exported_program_model.run_decompositions()

    ct_input_shape = ct.EnumeratedShapes(
        shapes=[
            [1, 3, 512, 384],
            [2, 3, 512, 384],
            [3, 3, 512, 384],
            [4, 3, 512, 384],
            [1, 3, 768, 576],
            [2, 3, 768, 576],
            [3, 3, 768, 576],
            [4, 3, 768, 576],
            [1, 3, 1024, 768],
            [2, 3, 1024, 768],
            [3, 3, 1024, 768],
            [4, 3, 1024, 768],
        ]
    )
    # heights_range = ct.RangeDim(lower_bound=384, upper_bound=1024)
    # widths_range = ct.RangeDim(lower_bound=384, upper_bound=1024)
    # batch_range = ct.RangeDim(lower_bound=1, upper_bound=4)
    coreml_model = ct.convert(
        exported_program_model_decomposed,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )

    output_path = "convnext_small_with_preprocess.mlpackage"
    coreml_model.save(output_path)
    print(f"Saved Core ML model to {output_path}")
    with open("convnext_small_with_preprocess.mlpackage/metadata.json", "w") as f:
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
