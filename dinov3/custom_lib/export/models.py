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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.model(x)
