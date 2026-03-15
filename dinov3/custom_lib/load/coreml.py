import pathlib

import coremltools as ct


def load_coreml_model(model_path: pathlib.Path) -> ct.models.MLModel:
    """Load a Core ML model from the specified path.

    Args:
        model_path (pathlib.Path): The path to the Core ML model file.
    Returns:
        ct.models.MLModel: The loaded Core ML model.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = ct.models.MLModel(model_path.as_posix())
    return model
