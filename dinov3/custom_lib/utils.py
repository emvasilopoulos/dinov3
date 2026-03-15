import pathlib

from torch import Tensor
import torch
import torchvision.transforms as T
import dinov3.hub.backbones
import dinov3.models.convnext
import PIL.Image

CURRENT_FILE_PATH: pathlib.Path = pathlib.Path(__file__)
TORCH_WEIGHTS_DIR: pathlib.Path = CURRENT_FILE_PATH.parent.parent.parent / "weights" / "pytorch"


def _load_pretrained_convnext(arch_name: str, weights_filename: str,
                              **model_kwargs) -> torch.nn.Module:
    convnext_arch = dinov3.models.convnext.get_convnext_arch(arch_name)
    model = convnext_arch(**model_kwargs)
    weights_path = TORCH_WEIGHTS_DIR / weights_filename
    checkpoint = torch.load(weights_path.as_posix(),
                            map_location="cpu",
                            weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    return model


def load_convnext_small_pretrained_pytorch() -> torch.nn.Module:
    return _load_pretrained_convnext(
        "convnext_small",
        "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
        patch_size=16,
        drop_path_rate=0.0,
    )


def load_convnext_base_pretrained_pytorch() -> torch.nn.Module:
    return _load_pretrained_convnext(
        "convnext_base",
        "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
        drop_path_rate=0.0,
    )


def load_vit_small_pretrained_pytorch() -> torch.nn.Module:
    vit = dinov3.hub.backbones.dinov3_vits16(pretrained=False)
    weights_filename = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    weights_path = TORCH_WEIGHTS_DIR / weights_filename
    checkpoint = torch.load(weights_path.as_posix(),
                            map_location="cpu",
                            weights_only=True)
    vit.load_state_dict(checkpoint, strict=True)
    return vit


IMAGE_TRANSFORM: T.Compose = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image_for_pretrained_model(image_path: pathlib.Path,
                                    normalize: bool = True) -> Tensor:
    image = PIL.Image.open(image_path).convert("RGB")
    if normalize:
        return IMAGE_TRANSFORM(image)
    else:
        return T.ToTensor()(image)
