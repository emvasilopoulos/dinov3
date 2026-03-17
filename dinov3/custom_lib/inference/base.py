import abc
import pathlib

import torch


class BaseInferenceModel:
    """
    A class to represent an ONNX model for inference.

    Attributes:
        session (ort.InferenceSession): The ONNX Runtime session for the model.
        expected_inputs (list): A list of OnnxExpectedInput instances representing the expected inputs for the model.
    """

    def __init__(self, model_path: pathlib.Path) -> None:
        self.session = self.load_model(model_path)

    @abc.abstractmethod
    def load_model(self, model_path: pathlib.Path) -> any:
        """
        Load the model from the specified path.
        This method should be implemented by subclasses to handle specific model loading logic.
        """
        pass

    @abc.abstractmethod
    def inference(self, torch_image_batch: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on a batch of images using the model.
        This method should be implemented by subclasses to handle specific inference logic.
        Args:
            torch_image_batch (torch.Tensor): A batch of images as a PyTorch tensor. (B, C, H, W)
        Returns:
            torch.Tensor: The output from the model as a PyTorch tensor. (B, D) where D is the output dimension of the model.
        """
        pass
    
    def __call__(self, torch_image_batch: torch.Tensor) -> torch.Tensor:
        return self.inference(torch_image_batch)