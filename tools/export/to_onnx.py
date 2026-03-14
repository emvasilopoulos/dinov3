import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # Register mean/std as buffers so they export correctly to ONNX
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

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
        x = (x - self.mean) / self.std

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

    dummy = torch.randn(1, 3, args.input_height, args.input_width, dtype=torch.float32)

    torch.onnx.export(
        full_model,
        dummy,
        "convnext_small_with_preprocess.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"x": [None, 3, None, None]},
        opset_version=18,
    )

if __name__ == "__main__":
    main()
