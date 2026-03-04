import torch
import torchvision
import numpy as np
import datasets as huggingface_datasets
import albumentations

import dinov3.models.convnext

EPOCHS = 5
LR = 0.001
BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_CLASSES = 101
LOG_LOSS_STEPS = 50

class DinoV3Classifier(torch.nn.Module):

    def __init__(
        self,
        backbone_dinov3: torch.nn.Module,
        image_width: int = 224,
        image_height: int = 224,
        patch_size: int = 16,
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
        num_classes: int = 10,
    ):
        super().__init__()
        self.backbone = backbone_dinov3
        self.image_width = image_width
        self.image_height = image_height
        self.patch_size = patch_size
        self.mean = mean
        self.std = std
        self.num_classes = num_classes

        self.classifier_head = torch.nn.Linear(768, num_classes)

        # freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize using mean and std
        mean = torch.tensor(self.mean).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor(self.std).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # # Preprocess input - already done by albumentations
        # x = self.preprocess(x)

        # Extract features using the frozen backbone
        features = self.backbone(x)
        print(f"Extracted features shape: {features.shape}")

        # Classify using the head
        logits = self.classifier_head(features)

        return logits
    
class HFDatasetTorch(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform, image_width=384, image_height=384):
        self.hf_ds = hf_ds
        self.transform = transform
        self.image_width = image_width
        self.image_height = image_height

    def __len__(self):
        return len(self.hf_ds)

    def _calculate_resize_dimensions(self, x: torch.Tensor) -> tuple[int, int]:
        # Calculate new dimensions while keeping aspect ratio
        original_height, original_width = x.shape[1], x.shape[2]
        aspect_ratio = original_width / original_height
        small_side = min(self.image_width, self.image_height)

        # TODO: something's wrong in aspect ratio < > 1. ==1 is ok.
        if aspect_ratio > 1:  # Wider than tall
            resize_width = self.image_width
            resize_height = int(self.image_width / aspect_ratio)
        elif aspect_ratio < 1:  # Taller than wide
            resize_height = self.image_height
            resize_width = int(self.image_height * aspect_ratio)
        else:
            resize_width = small_side
            resize_height = small_side
        return resize_width, resize_height
    
    def resize_pad(self, x: torch.Tensor) -> torch.Tensor:
        # resize input to match model's expected input size, keeping aspect ratio and padding if necessary
        resize_width, resize_height = self._calculate_resize_dimensions(x)
        print(f"Resizing to: {resize_height}x{resize_width}")
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0),
            size=(resize_height, resize_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        
        pad_width = self.image_width - resize_width
        pad_height = self.image_height - resize_height
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=114)

        return x

    def __getitem__(self, idx: int):
        ex = self.hf_ds[int(idx)]
        image = np.array(ex["image"])          # PIL -> HWC uint8
        if image.ndim == 2:               # grayscale -> RGB
            image = np.stack([image]*3, axis=-1)
        elif image.shape[2] == 4:         # RGBA -> RGB
            image = image[:, :, :3]
        out = self.transform(image=image)
        x = out["image"]                       # CHW torch tensor from ToTensorV2
        print(f"Original image shape: {x.shape} | dtype: {x.dtype}")
        x = x.contiguous().clone().float() / 255.0
        x = self.resize_pad(x)
        print(f"Final image shape after resize and pad: {x.shape}")
        # Make collation safe
        y = torch.tensor(ex["label"], dtype=torch.long)
        return x, y
    


def main(image_width, image_height, model_version="convnext_small", lr=0.001):

    if model_version == "convnext_small":
        model_Weights = "weights/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"
        convnext = dinov3.models.convnext.get_convnext_arch("convnext_small")
    elif model_version == "convnext_base":
        model_Weights = "weights/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"
        convnext = dinov3.models.convnext.get_convnext_arch("convnext_base")
    else:
        raise ValueError(f"Unsupported model version: {model_version}")
    model = convnext(
        patch_size=16,
        drop_path_rate=0.0,
        block_chunks=4,
    )
    cpkt = torch.load(model_Weights)
    model.load_state_dict(cpkt, strict=True)
    classifier_model = DinoV3Classifier(backbone_dinov3=model,
                                        num_classes=NUM_CLASSES)
    classifier_model = classifier_model.cuda()

    # load image classification dataset
    train_dataset = huggingface_datasets.load_dataset("ethz/food101", split="train")
    train_ds = HFDatasetTorch(train_dataset, albumentations.pytorch.ToTensorV2(), image_width=image_width, image_height=image_height)
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=NUM_WORKERS)

    val_dataset = huggingface_datasets.load_dataset("ethz/food101", split="validation")
    val_ds = HFDatasetTorch(val_dataset, albumentations.pytorch.ToTensorV2(), image_width=image_width, image_height=image_height)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=NUM_WORKERS)

    # Train the model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((p for p in classifier_model.parameters() if p.requires_grad), lr=lr)
    with open(f"training_log-h{image_width}-w{image_height}.txt", "w") as log_file:
        log_file.write(f"Training DINOv3 ConvNeXt classifier with config:\n")
        log_file.write(f"Image size: {image_width}x{image_height}\n")
        log_file.write(f"Epochs: {EPOCHS}\n")
        log_file.write(f"Learning rate: {lr}\n")
        log_file.write(f"Batch size: {BATCH_SIZE}\n")
        log_file.write(f"Number of classes: {NUM_CLASSES}\n")
        log_file.write(f"-----------------------------\n")
        for epoch in range(EPOCHS):
            classifier_model.train()
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = classifier_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if (i + 1) % LOG_LOSS_STEPS == 0:
                    print(f"Epoch {epoch+1}, Step {i+1}/{len(train_loader)}, Loss: {running_loss/LOG_LOSS_STEPS}")
                    log_file.write(f"Epoch {epoch+1}, Step {i+1}/{len(train_loader)}, Loss: {running_loss/LOG_LOSS_STEPS}\n")
                    running_loss = 0.0
            print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}")
            log_file.write(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}\n")

            # Evaluate on validation set
            classifier_model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.cuda(), labels.cuda()
                    outputs = classifier_model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
            print(f"Epoch {epoch+1}, Val Loss: {val_running_loss/len(val_loader)}")
            log_file.write(f"Epoch {epoch+1}, Val Loss: {val_running_loss/len(val_loader)}\n")
            log_file.write(f"-----------------------------\n")
            
            lr *= 0.9  # Decay learning rate by 10% each epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

if __name__ == "__main__":
    # image_widths = [384, 512, 768]
    image_widths = [384]
    # image_heights = [384, 512, 768]
    image_heights = [512]
    # model_versions = ["convnext_small", "convnext_base"]
    model_versions = ["convnext_base"]
    for model_version in model_versions:
        for image_width in image_widths:
            for image_height in image_heights:
                main(image_width, image_height, model_version=model_version, lr=LR)
