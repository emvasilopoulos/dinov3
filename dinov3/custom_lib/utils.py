import pathlib

import torch
import torchvision.transforms as T
import dinov3.models.convnext
import PIL.Image

CURRENT_FILE_PATH = pathlib.Path(__file__)
TORCH_WEIGHTS_DIR = CURRENT_FILE_PATH.parent.parent.parent / "weights" / "pytorch"


def load_convnext_small_pretrained_pytorch():
    small_convnext = dinov3.models.convnext.get_convnext_arch("convnext_small")
    model = small_convnext(
        patch_size=16,
        drop_path_rate=0.0,
    )

    weights_path = TORCH_WEIGHTS_DIR / "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"
    cpkt = torch.load(weights_path.as_posix())
    model.load_state_dict(cpkt, strict=True)
    return model


IMAGE_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image_for_pretrained_model(image_path: pathlib.Path):

    image = PIL.Image.open(image_path).convert("RGB")
    return IMAGE_TRANSFORM(image)
