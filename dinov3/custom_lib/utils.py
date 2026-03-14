import pathlib

from torch import Tensor
import torch
import torchvision.transforms as T
import dinov3.models.convnext
import PIL.Image

CURRENT_FILE_PATH: pathlib.Path = pathlib.Path(__file__)
TORCH_WEIGHTS_DIR: pathlib.Path = CURRENT_FILE_PATH.parent.parent.parent / "weights" / "pytorch"


def load_convnext_small_pretrained_pytorch() -> torch.nn.Module:
    small_convnext = dinov3.models.convnext.get_convnext_arch("convnext_small")
    model = small_convnext(
        patch_size=16,
        drop_path_rate=0.0,
    )

    weights_path = TORCH_WEIGHTS_DIR / "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"
    cpkt = torch.load(weights_path.as_posix())
    model.load_state_dict(cpkt, strict=True)
    return model


def load_convnext_base_pretrained_pytorch() -> torch.nn.Module:
    base_convnext = dinov3.models.convnext.get_convnext_arch("convnext_base")
    model = base_convnext(
        patch_size=16,
        drop_path_rate=0.0,
    )

    weights_path = TORCH_WEIGHTS_DIR / "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"
    cpkt = torch.load(weights_path.as_posix())
    model.load_state_dict(cpkt, strict=True)
    return model


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
