"""Shared, aspect-ratio preserving preprocessing for training and uploaded images."""

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import functional as TF

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ImagePreprocessor:
    def __init__(self, max_size=600, normalize=True, augment=False):
        if max_size < 32:
            raise ValueError("max_size must be at least 32")
        self.max_size = max_size
        self.normalize = normalize
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR,
                                      expand=True),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]) if augment else None

    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        if self.augmentation is not None:
            image = self.augmentation(image)
        # No crop or forced aspect ratio; bound attention cost for large uploads.
        image.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)
        width, height = image.size
        if width < 32 or height < 32:
            image = ImageOps.expand(image, border=(0, 0, max(0, 32-width), max(0, 32-height)))
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, MEAN, STD) if self.normalize else tensor


def pad_image_batch(batch):
    """Allow images of different sizes within the same DataLoader batch."""
    images, labels = zip(*batch)
    height = max(image.shape[-2] for image in images)
    width = max(image.shape[-1] for image in images)
    padded = [torch.nn.functional.pad(image, (0, width-image.shape[-1], 0, height-image.shape[-2]),
                                      mode="replicate") for image in images]
    return torch.stack(padded), torch.as_tensor(labels, dtype=torch.long)
