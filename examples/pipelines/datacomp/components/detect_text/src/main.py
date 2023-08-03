"""This component detexts text in images, using CRAFT.
"""
import logging

import dask.dataframe as dd
import numpy as np
import io
from PIL import Image

from huggingface_hub import hf_hub_download

from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
from easyocr.imgproc import normalizeMeanVariance
from easyocr.utils import group_text_box

import torch
import onnxruntime as ort

from fondant.component import DaskTransformComponent
from fondant.executor import DaskTransformExecutor

logger = logging.getLogger(__name__)


def resize_aspect_ratio_pillow(img, square_size, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    img = Image.fromarray(img)
    proc = img.resize((target_w, target_h), resample = Image.BILINEAR)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized, ratio, size_heatmap


def get_boxes(image_data, session):
    try:
      image = Image.open(io.BytesIO(image_data)).convert("RGB")
      image = np.array(image)
    except:
      return [None]

    # Use Pillow instead of cv2
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio_pillow(img=image,
                  square_size=512,
                  mag_ratio=1.0)

    ratio_h = ratio_w = 1 / target_ratio
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    input_name = session.get_inputs()[0].name

    # Prepare input tensor for inference
    inp = {input_name: x.numpy()}

    # Run inference and get output
    y, _ = session.run(None, inp)

    # Extract score and link maps
    score_text = y[0, :, :, 0]
    score_link = y[0, :, :, 1]

    # Post-processing to obtain bounding boxes and polygons
    boxes, _, _ = getDetBoxes(score_text, score_link, 0.5, 0.4, 0.4)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

    # Create horizontal reading list
    polys = []
    for box in boxes:
      poly = np.array(box).astype(np.int32).reshape((-1))
      polys.append(poly)

    horizontal_list, _ = group_text_box(polys)

    return horizontal_list


def get_boxes_dataframe(df, session):
    # process a single partition
    # TODO make column name more flexible
    df["image_detected_boxes"] = df.image_data.apply(lambda x:
        get_boxes(
            image_data=x, session=session,
        ),
    )

    return df


class DetextTextComponent(DaskTransformComponent):
    """Component that detexts text in images, using the CRAFT model.
    """

    def __init__(self, *args) -> None:

        craft_onnx = hf_hub_download(repo_id="ml6team/craft-onnx", filename="craft.onnx", repo_type="model")
        logger.info(f"Device: {ort.get_device()}")
        providers = [('CUDAExecutionProvider', {"cudnn_conv_algo_search": "DEFAULT"}), 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(craft_onnx, providers=providers)

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:

        logger.info(f"Length of the dataframe: {len(dataframe)}")

        # create meta
        # needs to be a dictionary with keys = column names, values = dtypes of columns
        # for each column in the output
        meta = {column: dtype for column, dtype in zip(dataframe.columns, dataframe.dtypes)}
        meta["image_detected_boxes"] = np.dtype(object) 

        logger.info("Detecting texts..")
        dataframe = dataframe.map_partitions(
            get_boxes_dataframe,
            session=self.session,
            meta=meta,
        )

        logger.info(f"Length of the final dataframe: {len(dataframe)}")
        print("First rows of final dataframe:", dataframe.head())

        return dataframe


if __name__ == "__main__":
    executor = DaskTransformExecutor.from_args()
    executor.execute(DetextTextComponent)