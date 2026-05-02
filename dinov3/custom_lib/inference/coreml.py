import pathlib

import coremltools as ct
import torch

from dinov3.custom_lib.inference.base import BaseInferenceModel
import dinov3.custom_lib.load.coreml
class CoreMLInferenceModel(BaseInferenceModel):
    
    def load_model(self, model_path: pathlib.Path) -> ct.models.MLModel:
        return dinov3.custom_lib.load.coreml.load_coreml_model(model_path)

    def inference(self, torch_image_batch: torch.Tensor) -> torch.Tensor:
        # Convert the PyTorch tensor to a NumPy array
        input_data = torch_image_batch.cpu().numpy()

        # Perform inference using the CoreML model
        output = self.session.predict({'input': input_data})

        # Convert the output back to a PyTorch tensor
        output_tensor = torch.from_numpy(output['output'])

        return output_tensor