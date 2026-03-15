import pathlib

from torch import Tensor
import torch
import torchvision.transforms as T
import PIL.Image

# LVD-1689M (web images dataset) training normalization values
IMAGE_TRANSFORM_NORM_LVD: T.Compose = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

IMAGE_TRANSFORM_TO_TENSOR: T.Compose = T.Compose([
    T.ToTensor(),
])


def load_image_for_pretrained_model(image_path: pathlib.Path,
                                    normalize: bool = True) -> Tensor:
    image = PIL.Image.open(image_path).convert("RGB")
    if normalize:
        return IMAGE_TRANSFORM_NORM_LVD(image)
    else:
        return IMAGE_TRANSFORM_TO_TENSOR(image)


def save_image_tensor(image_tensor: Tensor,
                      save_path: pathlib.Path,
                      denormalize: bool = True) -> None:
    # image_tensor shape (1, 3, H, W) or (3, H, W)
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    if denormalize:
        # unnormalize
        mean = torch.tensor([0.485, 0.456, 0.406],
                            dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           dtype=torch.float32).view(3, 1, 1)
        unnormalized_image_tensor = image_tensor * std + mean
    else:
        unnormalized_image_tensor = image_tensor
    # clamp to [0, 1]
    unnormalized_image_tensor = torch.clamp(unnormalized_image_tensor, 0.0,
                                            1.0)
    # convert to PIL Image and save
    unnormalized_image = T.ToPILImage()(unnormalized_image_tensor)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    unnormalized_image.save(save_path.as_posix())
