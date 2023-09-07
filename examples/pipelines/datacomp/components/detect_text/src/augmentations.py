"""
Source: https://raw.githubusercontent.com/locuslab/T-MARS/main/dataset2metadata/augmentations.py.
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import PIL
import numpy as np


class SquarePadResizeNorm:
    """Pad an image to be square, and resize to given shape.

    Credit to Sunding Wei:
    https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/5
    """

    def __init__(
        self,
        img_size,
        norm_mean=(0.5, 0.5, 0.5),
        norm_std=(0.5, 0.5, 0.5),
        do_normalize=True,
    ):
        self.img_size = img_size
        self.resize = T.Resize(
            img_size, interpolation=T.InterpolationMode.BICUBIC
        )  # Image will already be square
        if do_normalize:
            self.normalize = T.Normalize(
                mean=norm_mean,
                std=norm_std,
            )
        else:
            self.normalize = lambda x: x

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
            pad_value = 1.0

        elif isinstance(image, PIL.Image.Image):
            w, h = image.size
            pad_value = 255

        else:
            raise TypeError("Unsupported image type.")

        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, max_wh - w - hp, max_wh - h - vp]

        padded_img = F.pad(image, padding, pad_value, "constant")

        # Calculate offsets with respect to the original image size.
        padding = torch.from_numpy(np.asarray(padding, dtype=np.float32) / max_wh)

        resized_img = self.resize(padded_img)

        if isinstance(resized_img, PIL.Image.Image):
            resized_img = F.to_tensor(resized_img)

        return self.normalize(resized_img).unsqueeze(0), padding.unsqueeze(0)
