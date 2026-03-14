import numpy as np
import torch
import coremltools as ct
import onnxruntime as ort


# ----------------------------
# Configuration
# ----------------------------
COREML_MODEL_PATH = "model.mlmodel"   # or .mlpackage
ONNX_MODEL_PATH = "model.onnx"

# Expected input shape (adjust if needed)
B, C, H, W = 1, 3, 384, 512

# CoreML input name (must match model spec)
COREML_INPUT_NAME = "input"   # <-- change if needed


# ----------------------------
# Utility: Compare tensors
# ----------------------------
def compare_outputs(a, b, atol=1e-4, rtol=1e-4):
    abs_diff = np.abs(a - b)
    max_abs = abs_diff.max()
    mean_abs = abs_diff.mean()

    rel_diff = abs_diff / (np.abs(b) + 1e-8)
    max_rel = rel_diff.max()

    close = np.allclose(a, b, atol=atol, rtol=rtol)

    print("\n=== Comparison ===")
    print(f"Max abs diff : {max_abs:.6e}")
    print(f"Mean abs diff: {mean_abs:.6e}")
    print(f"Max rel diff : {max_rel:.6e}")
    print(f"Allclose     : {close}")

    return close


# ----------------------------
# Create test input
# ----------------------------
torch_input = torch.rand(B, C, H, W, dtype=torch.float32)
np_input = torch_input.numpy()


# ----------------------------
# Load ONNX model
# ----------------------------
print("Loading ONNX model...")
onnx_session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

onnx_input_name = onnx_session.get_inputs()[0].name
onnx_output_name = onnx_session.get_outputs()[0].name

onnx_output = onnx_session.run(
    [onnx_output_name],
    {onnx_input_name: np_input}
)[0]


# ----------------------------
# Load CoreML model
# ----------------------------
print("Loading CoreML model...")
coreml_model = ct.models.MLModel(COREML_MODEL_PATH)

coreml_output_dict = coreml_model.predict({
    COREML_INPUT_NAME: np_input
})

# If single output:
coreml_output = list(coreml_output_dict.values())[0]


# ----------------------------
# Ensure shapes match
# ----------------------------
print("\nONNX output shape   :", onnx_output.shape)
print("CoreML output shape :", coreml_output.shape)

if onnx_output.shape != coreml_output.shape:
    raise ValueError("Output shapes do not match.")


# ----------------------------
# Compare
# ----------------------------
compare_outputs(coreml_output, onnx_output)