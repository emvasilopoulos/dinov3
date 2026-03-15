import dataclasses


@dataclasses.dataclass(repr=True)
class BottleRecognitionTestConfig:
    preprocess_normalize_pixels_imagenet: bool = True
    preprocess_resize_pad: bool = True
    preprocess_new_width: int = 512
    preprocess_new_height: int = 384
    model: str = "convnext_small"  # or "vit_small"

    def __str__(self):
        return (
            f"Bottle Recognition Test Configuration:\n"
            f"- preprocess_normalize_pixels_imagenet={self.preprocess_normalize_pixels_imagenet} \n"
            f"- preprocess_resize_pad={self.preprocess_resize_pad} \n"
            f"- preprocess_new_width={self.preprocess_new_width} \n"
            f"- preprocess_new_height={self.preprocess_new_height} \n"
            f"- model='{self.model}'")
