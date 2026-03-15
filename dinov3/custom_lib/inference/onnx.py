import dataclasses

import torch

import onnxruntime as ort


@dataclasses.dataclass
class OnnxExpectedInput:
    """
    A dataclass to represent the expected input for the ONNX model.

    Attributes:
        name (str): The name of the input.
        shape (tuple): The shape of the input tensor.
        dtype (type): The data type of the input tensor.
    """
    name: str
    shape: tuple
    dtype: type
    data: any = None


class OnnxInferenceModel:
    """
    A class to represent an ONNX model for inference.

    Attributes:
        session (ort.InferenceSession): The ONNX Runtime session for the model.
        expected_inputs (list): A list of OnnxExpectedInput instances representing the expected inputs for the model.
    """

    def __init__(self, model_path: str, expected_inputs: list):
        self.session = ort.InferenceSession(model_path)
        self.expected_inputs = expected_inputs

    # compatibility with current codebase
    def __call__(self, torch_image_batch: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on a batch of images using the ONNX model.
        Args:
            torch_image_batch (torch.Tensor): A batch of images as a PyTorch tensor. (B, C, H, W)
        Returns:
            torch.Tensor: The output from the model as a PyTorch tensor. (B, D) where D is the output dimension of the model.
        """
        # Convert the PyTorch tensor to a NumPy array
        input_data = torch_image_batch.cpu().numpy()

        # Perform inference using the ONNX Runtime session
        output = inference(self.session, input_data)

        # Convert the output back to a PyTorch tensor
        output_tensor = torch.from_numpy(output[next(iter(output))])

        return output_tensor


def inference(session: ort.InferenceSession, input_data: dict) -> dict:
    """
    Perform inference using the ONNX Runtime session.

    Args:
        session (ort.InferenceSession): The ONNX Runtime session for the model.
        input_data (dict): A dictionary containing the input data for the model.

    Returns:
        dict: A dictionary containing the output from the model.
    """
    # Get the input name for the model
    input_name = session.get_inputs()[0].name

    # Run inference
    outputs = session.run(None, {input_name: input_data})

    # Get the output name for the model
    output_name = session.get_outputs()[0].name

    return {output_name: outputs[0]}
