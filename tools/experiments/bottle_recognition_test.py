import pathlib

import pandas as pd
import torch

import dinov3.custom_lib.utils
import dinov3.custom_lib.bottle_recognition_analysis.dataset
import dinov3.custom_lib.bottle_recognition_analysis.features
import dinov3.custom_lib.bottle_recognition_analysis.config
from dinov3.custom_lib.export.models import ConvNextWithPreProcess

EXP_CONFIG = dinov3.custom_lib.bottle_recognition_analysis.config.BottleRecognitionTestConfig(
    preprocess_normalize_pixels_imagenet=True,
    preprocess_resize_pad=True,
    preprocess_new_width=256,
    preprocess_new_height=512,
    model="convnext_small",
)

EXP_DIR = pathlib.Path("experiment_runs/bottle_recognition_test")
if EXP_DIR.exists():
    # increment directory name by 1
    i = 1
    while True:
        new_exp_dir = EXP_DIR.parent / f"{EXP_DIR.name}_{i}"
        if not new_exp_dir.exists():
            EXP_DIR = new_exp_dir
            break
        i += 1
EXP_DIR.mkdir(parents=True, exist_ok=False)
DEBUG_DIR = EXP_DIR / "debug_images"
DEBUG_DIR.mkdir(parents=True, exist_ok=False)

if EXP_CONFIG.preprocess_resize_pad:
    _resize_fn = ConvNextWithPreProcess.resize_pad_image
else:
    _resize_fn = ConvNextWithPreProcess.resize_image

if EXP_CONFIG.model == "convnext_small":
    _load_model_fn = dinov3.custom_lib.utils.load_convnext_small_pretrained_pytorch
elif EXP_CONFIG.model == "convnext_base":
    _load_model_fn = dinov3.custom_lib.utils.load_convnext_base_pretrained_pytorch
elif EXP_CONFIG.model == "vit_small":
    _load_model_fn = dinov3.custom_lib.utils.load_vit_small_pretrained_pytorch
else:
    raise ValueError(
        f"Unsupported model '{EXP_CONFIG.model}'. Supported models: 'convnext_small', 'convnext_base', 'vit_small'."
    )


def _create_record(current_class: str, opposing_class: str, distance: float,
                   image_1: str, image_2: str) -> dict:
    return {
        "current_class": current_class,
        "opposing_class": opposing_class,
        "distance": distance,
        "image_1": image_1,
        "image_2": image_2,
    }


def _compute_same_class_distances(
        class_name: str, vectors: list[torch.Tensor],
        images_and_vectors: list[tuple[pathlib.Path,
                                       torch.Tensor]]) -> list[dict]:
    records = []
    # Same class distances
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dist = dinov3.custom_lib.bottle_recognition_analysis.features.distance(
                vectors[i], vectors[j])
            print(
                f"- Distance between {images_and_vectors[i][0].name} and {images_and_vectors[j][0].name}: {dist:.4f}"
            )
            record = _create_record(
                current_class=class_name,
                opposing_class=class_name,
                distance=dist,
                image_1=images_and_vectors[i][0].name,
                image_2=images_and_vectors[j][0].name,
            )
            records.append(record)
    return records


def _compute_different_class_distances(
    class_name: str, vectors: list[torch.Tensor],
    current_class_images_and_vectors: list[tuple[pathlib.Path, torch.Tensor]],
    other_class_name: str,
    other_class_images_and_vectors: list[tuple[pathlib.Path, torch.Tensor]]
) -> list[dict]:
    records = []
    for i in range(len(vectors)):
        for j in range(len(other_class_images_and_vectors)):
            dist = dinov3.custom_lib.bottle_recognition_analysis.features.distance(
                vectors[i], other_class_images_and_vectors[j][1])
            print(
                f"- Distance between {current_class_images_and_vectors[i][0].name} and {other_class_images_and_vectors[j][0].name}: {dist:.4f}"
            )
            record = _create_record(
                current_class=class_name,
                opposing_class=other_class_name,
                distance=dist,
                image_1=current_class_images_and_vectors[i][0].name,
                image_2=other_class_images_and_vectors[j][0].name,
            )
            records.append(record)
    return records


def main(bottles_dir: pathlib.Path):

    # Load PyTorch Model
    model = _load_model_fn()
    if EXP_CONFIG.preprocess_normalize_pixels_imagenet:
        full_model = ConvNextWithPreProcess(backbone_model=model)
    else:
        full_model = model
    full_model.eval()

    # Read dataset - Images & compute feature vectors
    dataset = dinov3.custom_lib.bottle_recognition_analysis.dataset.load_dataset(
        bottles_dir,
        full_model,
        _resize_fn,
        resize_fn_width=EXP_CONFIG.preprocess_new_width,
        resize_fn_height=EXP_CONFIG.preprocess_new_height,
        debug_images_dir=DEBUG_DIR)

    records = []
    # Calculate distances between feature vectors of images in the dataset
    for bottle_class_i in range(len(dataset)):
        class_name, class_data = dataset[bottle_class_i]
        print(f"Distances for class {class_name}:")
        vectors = [vec for _, vec in class_data.images_and_vectors]

        records.extend(
            _compute_same_class_distances(class_name, vectors,
                                          class_data.images_and_vectors))

        for y in range(bottle_class_i + 1, len(dataset)):
            other_class_name, other_class_data = dataset[y]
            records.extend(
                _compute_different_class_distances(
                    class_name, vectors, class_data.images_and_vectors,
                    other_class_name, other_class_data.images_and_vectors))

    df = pd.DataFrame(records)
    CSV_PATH = EXP_DIR / "distances.csv"
    df.to_csv(CSV_PATH.as_posix(), index=False)
    EXP_CONFIG_PATH = EXP_DIR / "config.txt"
    with open(EXP_CONFIG_PATH, "w") as f:
        f.write(f"Dataset directory: {bottles_dir}\n")
        f.write(str(EXP_CONFIG))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Bottle Recognition Experiment")
    parser.add_argument("--bottles_dir",
                        type=pathlib.Path,
                        default=pathlib.Path("bottles_dataset"),
                        help="Path to the bottles dataset directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(bottles_dir=args.bottles_dir)
