"""This component detects text in images using an mmocr model.

Based on notebook: Run inference with FAST for text detection.ipynb.
"""
import io
import logging
import typing as t

import numpy as np
import pandas as pd
import torch
from fondant.component import PandasTransformComponent
from PIL import Image

logger = logging.getLogger(__name__)

from mmengine.config import Config

# import subprocess
# # make sure to compile the postprocessing scripts
# subprocess.call(["sh", "./compile.sh"])  # nosec

from models import build_model
from models.utils import fuse_module, rep_model_convert

from augmentations import SquarePadResizeNorm

from huggingface_hub import hf_hub_download


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
        raise ValueError("Not ok")
    polygon = np.array(polygon, dtype=np.float32)
    x = polygon[::2]
    y = polygon[1::2]
    return np.array([min(x), min(y), max(x), max(y)])


def process_image(image: bytes, *, image_transform, device: str) -> torch.Tensor:
    """
    Process the image to a tensor.

    Args:
        image: The input image as a byte string.
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

    return transform(load(image))


def detect_text_batch(
    image_batch: pd.DataFrame,
    *,
    cfg,
    model,
) -> pd.Series:
    """Detext text on a batch of images."""
    imgs = torch.stack(list(image_batch), dim=0)
    batch_size = imgs.shape[0]
    # TODO fix this, make this component also take width and height
    # as input in spec to process arbitrarly sized images
    img_metas = {
        "filename": [None for i in range(batch_size)],
        "org_img_size": torch.ones((batch_size, 2)).long() * 256,
        "img_size": torch.ones((batch_size, 2)).long() * 512,
    }

    data = dict()
    data["imgs"] = imgs
    data["img_metas"] = img_metas
    data.update(dict(cfg=cfg))

    # forward
    with torch.no_grad():
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

    return pd.Series(boxes_batch, index=image_batch.index)


class DetectTextComponent(PandasTransformComponent):
    """Component that detects text in images using an mmocr model."""

    def __init__(self, *args, batch_size: int) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        logger.info(f"Device: {self.device}")

        self.batch_size = batch_size
        self.image_transform = SquarePadResizeNorm(
            img_size=512,
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
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["ema"]
        d = dict()
        for key, value in state_dict.items():
            tmp = key.replace("module.", "")
            d[tmp] = value
        self.model.load_state_dict(d)
        self.model = self.model.to(self.device)

        self.model = rep_model_convert(self.model)
        # fuse conv and bn
        self.model = fuse_module(self.model)
        self.model.eval()
        return self.model

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # start of dummy dataframe
        # from PIL import Image
        # import requests
        # import io

        # url = "https://raw.githubusercontent.com/locuslab/T-MARS/main/sample_text_detect.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)
        # image_bytes = io.BytesIO()
        # image.save(image_bytes, format='PNG')
        # image_bytes = image_bytes.getvalue()

        # data = {"data": [image_bytes]}
        # dataframe = pd.DataFrame.from_dict(data)
        # dataframe = pd.concat({"images": dataframe}, axis=1)
        # end of dummy dataframe

        images = pd.Series(dataframe["images"]["data"]).apply(
            process_image,
            image_transform=self.image_transform,
            device=self.device,
        )

        results: t.List[pd.Series] = []
        for batch in np.split(
            images, np.arange(self.batch_size, len(images), self.batch_size)
        ):
            if not batch.empty:
                results.append(
                    detect_text_batch(
                        batch,
                        cfg=self.cfg,
                        model=self.model,
                    ).T,
                )

        result = pd.concat(results).to_frame(name=("images", "boxes"))

        dataframe = pd.concat([dataframe, result], axis=1)

        return dataframe
