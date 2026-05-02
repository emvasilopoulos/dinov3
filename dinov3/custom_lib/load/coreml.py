import pathlib

import coremltools as ct


def load_coreml_model(model_path: pathlib.Path) -> ct.models.MLModel:
    """Load a Core ML model from the specified path.

    Args:
        model_path (pathlib.Path): The path to the Core ML model file.
    Returns:
        ct.models.MLModel: The loaded Core ML model.
    """
    if not model_path.suffix == ".mlpackage" and not model_path.is_dir():
        raise ValueError(f"Expected a .mlpackage folder, got {model_path.suffix}")

    return ct.models.MLModel(model_path.as_posix())
