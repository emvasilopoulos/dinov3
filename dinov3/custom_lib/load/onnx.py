import pathlib

import onnxruntime as ort


def load_onnx_model(model_path: pathlib.Path) -> ort.InferenceSession:
    """
    Load an ONNX model from the specified path.

    Args:
        model_path (pathlib.Path): The path to the ONNX model file.
    Returns:
        ort.InferenceSession: An ONNX Runtime inference session for the loaded model.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model file not found at: {model_path}")

    session = ort.InferenceSession(model_path.as_posix())
    return session
