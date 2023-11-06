"""This component detects text in images using an mmocr model.

Based on notebook: Run inference with FAST for text detection.ipynb.
"""
import io
import logging
import os
import typing as t

import numpy as np
import pandas as pd
import torch
from fondant.component import PandasTransformComponent
from PIL import Image

from mmengine.config import Config

from augmentations import SquarePadResizeNorm
from models import build_model
from models.utils import fuse_module, rep_model_convert

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"


def poly2bbox(polygon: np.array) -> np.array:
    """Converting a polygon to a bounding box.

    Args:
         polygon (ArrayLike): A polygon. In any form can be converted
             to an 1-D numpy array. E.g. list[float], np.ndarray,
             or torch.Tensor. Polygon is written in
             [x1, y1, x2, y2, ...].

     Returns:
         np.array: The converted bounding box [x1, y1, x2, y2]
    """
    if len(polygon) % 2 != 0:
        raise ValueError("Length of polygon should be even")
    polygon = np.array(polygon, dtype=np.float32)
    x = polygon[::2]
    y = polygon[1::2]
    return np.array([min(x), min(y), max(x), max(y)])


def process_image_batch(
    images: np.ndarray, *, image_transform: SquarePadResizeNorm, device: str
) -> t.List[torch.Tensor]:
    """
    Process image in batches to a list of tensors.

    Args:
        images: The input images as a numpy array containing byte strings.
        image_transform: The image transformation to apply.
        device: The device to move the transformed image to.
    """

    def load(img: bytes) -> Image:
        """Load the bytestring as an image."""
        bytes_ = io.BytesIO(img)
        return Image.open(bytes_).convert("RGB")

    def transform(img: Image) -> torch.Tensor:
        """Transform the image to a tensor and move it to the specified device."""
        return image_transform(img)[0][0].to(device)

    return [transform(load(image)) for image in images]


@torch.no_grad()
def detect_text_batch(
    image_batch: t.List[torch.Tensor],
    *,
    cfg: Config,
    model: torch.nn.Module,
    image_size: int,
    index: pd.Series,
) -> pd.Series:
    """Detext text on a batch of images."""
    imgs = torch.stack(image_batch, dim=0)
    batch_size = imgs.shape[0]

    image_size_tensor = torch.ones((batch_size, 2)).long() * image_size
    img_metas = {
        "filename": [None for i in range(batch_size)],
        "org_img_size": image_size_tensor,
        "img_size": image_size_tensor,
    }

    data = dict()
    data["imgs"] = imgs
    data["img_metas"] = img_metas
    data.update(dict(cfg=cfg))

    outputs = model(**data)

    # get cropped images
    boxes_batch = []
    for i in range(batch_size):
        raw_contours = outputs["results"][i]["bboxes"]

        boxes = []
        for polygon in raw_contours:
            box = poly2bbox(polygon)
            boxes.append(box)
        boxes_batch.append(boxes)

    return pd.Series(boxes_batch, index=index)


class DetectTextComponent(PandasTransformComponent):
    """Component that detects text in images using an mmocr model."""

    def __init__(self, *_, batch_size: int, image_size: int):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")

        self.batch_size = batch_size
        self.image_size = image_size
        self.image_transform = SquarePadResizeNorm(
            img_size=self.image_size,
            norm_mean=(0.485, 0.456, 0.406),
            norm_std=(0.229, 0.224, 0.225),
        )

        cfg = Config.fromfile("config/fast/tt/fast_tiny_tt_512_finetune_ic17mlt.py")
        self.cfg = cfg

        checkpoint_path = hf_hub_download(
            repo_id="ml6team/fast",
            filename="fast_tiny_tt_512_finetune_ic17mlt.pth",
            repo_type="model",
        )

        self.model = build_model(cfg.model)
        self.model = self.init_model(checkpoint_path)

    def init_model(self, checkpoint_path):
        logger.info("Initializing model")

        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["ema"]
        d = dict()
        for key, value in state_dict.items():
            tmp = key.replace("module.", "")
            d[tmp] = value
        self.model.load_state_dict(d)
        self.model = self.model.to(self.device)

        self.model = rep_model_convert(self.model)
        # fuse convolutional and batch norm layers
        self.model = fuse_module(self.model)
        self.model.eval()

        logger.info("Model initialized")

        return self.model

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        images = dataframe["images"]["data"]

        results: t.List[pd.Series] = []
        for batch in np.split(
            images, np.arange(self.batch_size, len(images), self.batch_size)
        ):
            if not batch.empty:
                image_tensors = process_image_batch(
                    batch, image_transform=self.image_transform, device=self.device
                )

                boxes = detect_text_batch(
                    image_tensors,
                    cfg=self.cfg,
                    model=self.model,
                    image_size=self.image_size,
                    index=batch.index,
                ).T

                results.append(boxes)

        result = pd.concat(results).to_frame(name=("images", "boxes"))

        return pd.concat([dataframe, result], axis=1)
