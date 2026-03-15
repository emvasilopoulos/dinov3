import dataclasses
import pathlib
from typing import Callable, Optional

import torch
import dinov3.custom_lib.utils
from dinov3.custom_lib.export.models import ConvNextWithPreProcess


@dataclasses.dataclass
class BottleClassData:
    name: str
    images_and_vectors: list[tuple[pathlib.Path, torch.Tensor]]


def _debug_save_image_tensor(image_tensor: torch.Tensor,
                             bottle_class_name: str, image_name: str,
                             save_dir: pathlib.Path):
    debug_image_path = save_dir / f"{bottle_class_name}_{image_name}"
    debug_image_path.parent.mkdir(parents=True, exist_ok=True)
    dinov3.custom_lib.utils.save_image_tensor(image_tensor,
                                              debug_image_path,
                                              denormalize=False)


def load_dataset(
    dir_path: pathlib.Path,
    model: ConvNextWithPreProcess,
    resize_fn: Callable[[torch.Tensor], torch.Tensor],
    resize_fn_width: int,
    resize_fn_height: int,
    debug_images_dir: Optional[pathlib.Path] = None,
) -> dict[str, BottleClassData]:
    subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
    dataset = []
    for subdir in subdirs:
        bottle_class_name = subdir.name

        images_and_vectors = []
        for image_path in sorted(subdir.glob("*.png")):
            with torch.no_grad():
                # Load image
                image_tensor = dinov3.custom_lib.utils.load_image_for_pretrained_model(
                    image_path, normalize=False).unsqueeze(0)
                # Resize image
                resized_image_tensor = resize_fn(image_tensor,
                                                 new_width=resize_fn_width,
                                                 new_height=resize_fn_height)

                if debug_images_dir is not None:
                    # debug image tensor - store
                    _debug_save_image_tensor(resized_image_tensor,
                                             bottle_class_name,
                                             image_path.name, debug_images_dir)

                # produce feature vector
                feature_vector = model(resized_image_tensor).squeeze(0).cpu()

            images_and_vectors.append((image_path, feature_vector))
        dataset.append(
            (bottle_class_name,
             BottleClassData(name=bottle_class_name,
                             images_and_vectors=images_and_vectors)))
    return dataset
