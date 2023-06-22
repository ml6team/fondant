"""This component filters image-text pairs using a text spotting model.

As proposed in [Radenovic et al., 2023](https://arxiv.org/abs/2301.02280).
"""
import logging

from mmocr.apis import MMOCRInferencer
import numpy as np
import pandas as pd
import torch

from fondant.component import PandasTransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def text_spotting_batch(urls, captions, ocr):
    # Perform batched inference
    urls = list(urls)
    result = ocr(urls, batch_size=len(urls))

    caption_in_images = []
    for prediction, caption in zip(result["predictions"], captions):
        caption_in_image = False
        for det in prediction["rec_texts"]:
            if det.lower() in caption.lower():
                caption_in_image = True
        caption_in_images.append(caption_in_image)

    return pd.Series(caption_in_images, index=captions.index)


class FilterTextSpotting(PandasTransformComponent):
    """Component that filters image-text pairs using a text spotting model"""

    def setup(self, *, batch_size: int) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")

        self.ocr = MMOCRInferencer(det="DBNet", rec="SAR", device=self.device)
        self.batch_size = batch_size

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        images = dataframe["image"]["url"]
        captions = dataframe["text"]["data"]

        results = []
        images = np.split(
            images, np.arange(self.batch_size, len(images), self.batch_size)
        )
        captions = np.split(
            captions, np.arange(self.batch_size, len(captions), self.batch_size)
        )
        for image_batch, caption_batch in zip(images, captions):
            if not image_batch.empty and not caption_batch.empty:
                results.append(
                    text_spotting_batch(
                        image_batch,
                        caption_batch,
                        ocr=self.ocr,
                    )
                )
        mask = pd.concat(results)

        dataframe = dataframe[~mask]

        return dataframe


if __name__ == "__main__":
    component = FilterTextSpotting.from_args()
    component.run()
