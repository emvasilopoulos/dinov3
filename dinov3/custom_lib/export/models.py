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
    def resize_image(x: torch.Tensor, new_width: int,
                     new_height: int) -> torch.Tensor:
        return nn.functional.interpolate(x,
                                         size=(new_height, new_width),
                                         mode="bilinear",
                                         align_corners=False)

    @staticmethod
    def resize_pad_image(x: torch.Tensor, new_width: int,
                         new_height: int) -> torch.Tensor:
        # output shape (512, 384)

        # keep aspect ratio
        h, w = x.shape[2], x.shape[3]
        scale = min(new_height / h, new_width / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = nn.functional.interpolate(x,
                                            size=(new_h, new_w),
                                            mode="bilinear",
                                            align_corners=False)
        # pad to target size
        pad_h = new_height - new_h
        pad_w = new_width - new_w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        padded = nn.functional.pad(resized,
                                   (pad_left, pad_right, pad_top, pad_bottom),
                                   mode="constant",
                                   value=1.0)
        return padded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.model(x)
