"""
Clean cut classifier filter
"""
import logging

import numpy as np
from cv2 import cv2
from scipy import stats

# pylint: disable=import-error
from helpers.logger import get_logger

logger = get_logger(name=__name__, level=logging.INFO)


# pylint: disable=too-many-locals
def single_component_edge_detection(img_path: str, force_alpha=False, positive_margin: int = 20,
                                    negative_margin: int = 1) -> bool:
    """
    Function that detects single component with edge detection approach
    Args:
        img_path (str): the path to the image
        force_alpha (bool): whether to remove the alpha channel to the image
        positive_margin (int): the positive margin to extend the image with
        negative_margin (int): the negative margin to reduce the image with
    Returns:
        bool: whether the image is a single component
    """
    # We start from a grayscale image.
    # For transparent images, we'll use the alpha channel.
    if img_path.endswith('.png') or force_alpha:
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        gray = image[:, :, -1]
    else:
        # TODO: increase contrast?
        image = cv2.imread(img_path)
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # pylint: disable=broad-except
        except Exception as exception:
            logger.error("Failed to convert '%s' to grayscale", img_path)
            logger.error(exception)
            return False

    # Canny edge detection has issues when objects intersect with the image
    # boundaries. To avoid those issues, we look for the most common border
    # image and extend the image by one pixel along each axis.
    edge_pixels = np.concatenate((
        gray[0, 1:],
        gray[1:, -1],
        gray[-1, :-1],
        gray[1:, 0]
    ))

    mode_pixel = stats.mode(edge_pixels)[0][0]
    total_margin = positive_margin + negative_margin
    extended = np.full(shape=tuple(d + positive_margin * 2 for d in gray.shape),
                       fill_value=mode_pixel)
    extended[total_margin:-total_margin, total_margin:-total_margin] \
        = gray[negative_margin:-negative_margin, negative_margin:-negative_margin]

    # Preprocessing, edge detection, edge dilation.
    blurred = cv2.GaussianBlur(extended, (3, 3), 0)  # TODO: should be relative to image size
    edges = cv2.Canny(blurred, 50, 100)  # TODO: params can be tuned
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edges, dilation_kernel, iterations=1)

    # Contour detection
    # TODO:
    #   - params?
    #   - exclude the case where we grab the entire image?
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # TODO: when the dilation was insufficient, some nested components remain
    #   whole object. Tuning the dilation kernel size as one of the parameters
    #   should mostly address this. However, I still feel like there should be
    #   a better solution...

    return len(contours) == 1


# pylint: disable=invalid-name
def single_component_floodfill(img_path, n_clusters=5, blur_kernel_size=3, positive_margin=20,
                               negative_margin=1) -> bool:
    """
    Function that detects single component with floodfill approach
    Args:
        img_path (str): the path to the image
        n_clusters (int): the number of components for the k-mean clustering used to get rid of
         dithering
        blur_kernel_size (int): the size of the blur filter kernel
        positive_margin (int): the positive margin to extend the image with
        negative_margin (int): the negative margin to reduce the image with
    Returns:
        bool: whether the image is a single component
    """
    # Load image
    image = cv2.imread(img_path)
    # Kmeans color quantization (to get rid of dither)
    # pylint: disable=too-many-function-args
    # TODO double check if reshape argument is correct (I think it should be a tuple and pylint is
    # complaining)
    samples = np.float32(image).reshape(-1, 3)
    _, labels, centers = cv2.kmeans(data=samples,
                                    K=n_clusters,
                                    bestLabels=None,
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                              10000, 0.0001),
                                    attempts=1,
                                    flags=cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    # pylint: disable=unsubscriptable-object
    res = centers[labels.flatten()]
    image = res.reshape(image.shape)

    # Identify background colour
    edge_pixel_rows = np.concatenate((
        image[0, 1:],
        image[1:, -1],
        image[-1, :-1],
        image[1:, 0]
    ))
    edge_pixel_rows = np.ascontiguousarray(edge_pixel_rows)
    void_dt = np.dtype(
        (np.void, edge_pixel_rows.dtype.itemsize * np.prod(edge_pixel_rows.shape[1:])))
    _, ids, count = np.unique(edge_pixel_rows.view(void_dt).ravel(), return_index=True,
                              return_counts=True)
    largest_count_id = ids[count.argmax()]
    background_colour = edge_pixel_rows[largest_count_id]

    # Add margin
    total_margin = positive_margin + negative_margin
    extended_shape = np.array(image.shape)
    extended_shape[:2] += positive_margin * 2
    extended = np.full(shape=tuple(extended_shape),
                       fill_value=background_colour)
    extended[total_margin:-total_margin, total_margin:-total_margin] \
        = image[negative_margin:-negative_margin, negative_margin:-negative_margin]
    image = extended

    # Floodfill background
    background_fill_color = (74, 65, 42)
    # ugliest color in existence according to some,
    # TODO check if color not in image, otherwise pick another
    x, y = np.where(np.all(image == background_colour, axis=-1))
    floodfill_seed_point = (y[0], x[0])
    cv2.floodFill(image, None, seedPoint=floodfill_seed_point, newVal=background_fill_color,
                  loDiff=(0, 0, 0, 0),
                  upDiff=(0, 0, 0, 0))

    # Black foreground on white background
    black_and_white = np.zeros(image.shape, dtype="uint8")
    black_and_white[np.where((image == background_fill_color).all(axis=2))] = [255, 255, 255]
    image = cv2.cvtColor(black_and_white, cv2.COLOR_BGR2GRAY)

    # Remove pixel-level noise
    image = cv2.medianBlur(image, blur_kernel_size)

    # Floodfill foreground object in gray (to see if there is only one)
    x, y = np.where(image == 0)  # get two arrays with x and y positions of foreground
    floodfill_seed_point = (y[0], x[0])  # first foreground pixel
    cv2.floodFill(image, None, seedPoint=floodfill_seed_point, newVal=127, loDiff=(0, 0, 0, 0),
                  upDiff=(0, 0, 0, 0))

    return 0 not in image


def single_component_ensemble(image_path: str) -> bool:
    """
    A function that combines both edges and floodfill approach for single component classifier
    Args:
        image_path (str): the path to the image to classify
    Returns:
        bool: whether the image is a single component
    """
    return single_component_edge_detection(image_path) and single_component_floodfill(image_path)
