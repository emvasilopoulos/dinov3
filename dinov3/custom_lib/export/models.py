import torch
import torch.nn as nn


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

    @staticmethod
    def resize_image_vertical_small(x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x,
                                         size=(512, 384),
                                         mode="bilinear",
                                         align_corners=False)

    @staticmethod
    def resize_image_square_small(x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x,
                                         size=(384, 384),
                                         mode="bilinear",
                                         align_corners=False)

    @staticmethod
    def resize_image_vertical_large(x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x,
                                         size=(768, 512),
                                         mode="bilinear",
                                         align_corners=False)

    @staticmethod
    def resize_image_square_large(x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x,
                                         size=(512, 512),
                                         mode="bilinear",
                                         align_corners=False)

    @staticmethod
    def resize_image_vertical_large2(x: torch.Tensor) -> torch.Tensor:
        # Bad results - keeping for reference
        return nn.functional.interpolate(x,
                                         size=(768, 384),
                                         mode="bilinear",
                                         align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.model(x)
