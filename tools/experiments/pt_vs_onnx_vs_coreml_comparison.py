import torch
import onnxruntime
import coremltools as ct

import pathlib

import pandas as pd
import torch

import dinov3.custom_lib.bottle_recognition_analysis.dataset
import dinov3.custom_lib.bottle_recognition_analysis.config
from dinov3.custom_lib.export.models import ConvNextWithPreProcess
import dinov3.custom_lib.inference.onnx
import dinov3.custom_lib.inference.coreml
import dinov3.custom_lib.load.pytorch
import dinov3.custom_lib.load.onnx

EXP_DIR = pathlib.Path("experiment_runs/pt_vs_onnx_vs_coreml_comparison")
EXP_CONFIG = dinov3.custom_lib.bottle_recognition_analysis.config.BottleRecognitionTestConfig(
    preprocess_normalize_pixels_imagenet=True,
    preprocess_resize_pad=True,
    preprocess_new_width=256,
    preprocess_new_height=512,
    model="convnext_small",
)
TOLERATED_DIFFERENCE = 5e-5


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=
        "Compare PyTorch, ONNX, and CoreML models for bottle recognition")
    parser.add_argument(
        "--bottles-dir",
        type=str,
        default="bottles_dataset",
        help="Directory containing the bottle images for testing")
    parser.add_argument("--onnx-path",
                        type=str,
                        required=False,
                        help="Path to the ONNX model file")
    parser.add_argument("--mlpackage-path",
                        type=str, required=True,
                        help="Path to the CoreML .mlpackage file"
                        )
    return parser.parse_args()


def main(bottles_dir: pathlib.Path, onnx_path: pathlib.Path, mlpackage_path: pathlib.Path):

    # Load PyTorch Model
    model = dinov3.custom_lib.load.pytorch.load_convnext_small_pretrained_pytorch(
    )
    if EXP_CONFIG.preprocess_normalize_pixels_imagenet:
        full_model_pt = ConvNextWithPreProcess(backbone_model=model)
    else:
        full_model_pt = model
    full_model_pt.eval()

    # Read dataset - Images & compute feature vectors
    dataset_pt = dinov3.custom_lib.bottle_recognition_analysis.dataset.load_dataset(
        bottles_dir,
        full_model_pt,
        ConvNextWithPreProcess.resize_pad_image,
        resize_fn_width=EXP_CONFIG.preprocess_new_width,
        resize_fn_height=EXP_CONFIG.preprocess_new_height)

    model_onnx_wrapper = dinov3.custom_lib.inference.onnx.OnnxInferenceModel(
        onnx_path)
    dataset_onnx = dinov3.custom_lib.bottle_recognition_analysis.dataset.load_dataset(
        bottles_dir,
        model_onnx_wrapper,
        ConvNextWithPreProcess.resize_pad_image,
        resize_fn_width=EXP_CONFIG.preprocess_new_width,
        resize_fn_height=EXP_CONFIG.preprocess_new_height)
    
    model_coreml_wrapper = dinov3.custom_lib.inference.coreml.CoreMLInferenceModel(
        mlpackage_path)
    dataset_coreml = dinov3.custom_lib.bottle_recognition_analysis.dataset.load_dataset(
        bottles_dir,
        model_coreml_wrapper,
        ConvNextWithPreProcess.resize_pad_image,
        resize_fn_width=EXP_CONFIG.preprocess_new_width,
        resize_fn_height=EXP_CONFIG.preprocess_new_height)

    # ONNX vs PyTorch Comparison
    for pt_bottle_data, onnx_bottle_data in zip(dataset_pt, dataset_onnx):
        pt_bottle_data = pt_bottle_data[1].images_and_vectors
        onnx_bottle_data = onnx_bottle_data[1].images_and_vectors
        for (pt_image_path, pt_vector), (onnx_image_path, onnx_vector) in zip(
                pt_bottle_data, onnx_bottle_data):
            assert pt_image_path == onnx_image_path, f"Image paths do not match: {pt_image_path} vs {onnx_image_path}"
            assert pt_vector.shape == onnx_vector.shape, f"Vector shapes do not match for image {pt_image_path}: {pt_vector.shape} vs {onnx_vector.shape}"
            if not torch.allclose(
                    pt_vector, onnx_vector, atol=TOLERATED_DIFFERENCE):
                print(
                    f"Vectors do not match for image {pt_image_path}. Max absolute difference: {(pt_vector - onnx_vector).abs().max().item()}"
                )
            else:
                print(f"Vectors match for image {pt_image_path}")

    # CoreML vs PyTorch Comparison
    for pt_bottle_data, coreml_bottle_data in zip(dataset_pt, dataset_coreml):
        pt_bottle_data = pt_bottle_data[1].images_and_vectors
        coreml_bottle_data = coreml_bottle_data[1].images_and_vectors
        for (pt_image_path, pt_vector), (coreml_image_path, coreml_vector) in zip(
                pt_bottle_data, coreml_bottle_data):
            assert pt_image_path == coreml_image_path, f"Image paths do not match: {pt_image_path} vs {coreml_image_path}"
            assert pt_vector.shape == coreml_vector.shape, f"Vector shapes do not match for image {pt_image_path}: {pt_vector.shape} vs {coreml_vector.shape}"
            if not torch.allclose(
                    pt_vector, coreml_vector, atol=TOLERATED_DIFFERENCE):
                print(
                    f"Vectors do not match for image {pt_image_path}. Max absolute difference: {(pt_vector - coreml_vector).abs().max().item()}"
                )
            else:
                print(f"Vectors match for image {pt_image_path}")


if __name__ == "__main__":
    args = parse_args()
    bottles_dir = pathlib.Path(args.bottles_dir)
    main(bottles_dir, onnx_path=pathlib.Path(args.onnx_path), mlpackage_path=pathlib.Path(args.mlpackage_path))

"""
OUTPUT for CoreML vs PyTorch Comparison:
Vectors match for image datasets/Test-Bottles-cropped/pepsi/pepsi-1.png
Vectors match for image datasets/Test-Bottles-cropped/pepsi/pepsi-2.png
Vectors match for image datasets/Test-Bottles-cropped/pepsi/pepsi-3.png
Vectors match for image datasets/Test-Bottles-cropped/glassbottle/glassbottle-1.png
Vectors do not match for image datasets/Test-Bottles-cropped/glassbottle/glassbottle-2.png. Max absolute difference: 6.413459777832031e-05
Vectors match for image datasets/Test-Bottles-cropped/zero/zero-1.png
Vectors match for image datasets/Test-Bottles-cropped/zero/zero-2.png
Vectors match for image datasets/Test-Bottles-cropped/zero/zero-3.png
Vectors do not match for image datasets/Test-Bottles-cropped/thermos/thermos-1.png. Max absolute difference: 9.131431579589844e-05
Vectors do not match for image datasets/Test-Bottles-cropped/thermos/thermos-2.png. Max absolute difference: 0.00011947751045227051
Vectors do not match for image datasets/Test-Bottles-cropped/thermos/thermos-3.png. Max absolute difference: 0.00011730194091796875

According to these results, I conclude it's best to create the reference vectors using the exported format runtime engine. So
for CoreML I should generate a separate .db of vectors using CoreML engine and for Android another .db using LiteRT engine.
"""