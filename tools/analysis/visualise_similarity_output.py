import pathlib

import matplotlib.pyplot as plt
import torch
import dinov3.custom_lib.utils

IMAGES_DIR = pathlib.Path(__file__).parent.parent.parent / "datasets" / "cola"


def _is_vertical_image(image_batch: torch.Tensor) -> bool:
    _, _, h, w = image_batch.shape
    return h > w


def _is_horizontal_image(image_batch: torch.Tensor) -> bool:
    _, _, h, w = image_batch.shape
    return w > h


def main():
    model = dinov3.custom_lib.utils.load_convnext_small_pretrained_pytorch()
    model.eval()
    model.to(device=torch.device("cuda:0"))

    images = IMAGES_DIR.glob("*.JPG")
    for image_path in sorted(images):
        print(f"Processing {image_path.name}...")
        with torch.no_grad():
            torch_image = dinov3.custom_lib.utils.load_image_for_pretrained_model(
                image_path).unsqueeze(0).to(device=torch.device("cuda:0"))
            if _is_vertical_image(torch_image):
                resized_image = torch.nn.functional.interpolate(
                    torch_image,
                    size=(768, 384),
                    mode="bicubic",
                    align_corners=False)
                pass
            elif _is_horizontal_image(torch_image):
                resized_image = torch.nn.functional.interpolate(
                    torch_image,
                    size=(384, 768),
                    mode="bicubic",
                    align_corners=False)
                pass
            else:
                resized_image = torch.nn.functional.interpolate(
                    torch_image,
                    size=(384, 384),
                    mode="bicubic",
                    align_corners=False)
            output = model(resized_image).squeeze(0).cpu()

        # plot as a signal
        plt.figure()
        plt.plot(output)
        plt.title(f"Similarity Output for {image_path.name}")
        plt.xlabel("Feature Index")
        plt.ylabel("Similarity Score")
        # store the plot in the same directory as the image
        plot_path = image_path.parent / f"{image_path.stem}_similarity_plot.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved similarity plot to {plot_path}")


if __name__ == "__main__":
    main()
