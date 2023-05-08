from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image
from io import BytesIO

processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
model = UperNetForSemanticSegmentation.from_pretrained(
    "openmmlab/upernet-convnext-small"
)


def generate_segmentation_map(image_bytes):
    """
    Generates a segmentation map for the input image bytes using the UperNet
    model.

    Args:
        image_bytes (bytes): The input image bytes to generate the
                             segmentation map for.

    Returns:
        seg: A segmentation map containing class labels for each pixel in the
             input image.
    """
    if image_bytes is None:
        return None

    image = Image.open(BytesIO(image_bytes))
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    outputs = model(pixel_values)
    seg = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )

    return seg


def generate_segmentation_maps(df):
    """
    Generates segmentation maps for a DataFrame containing images in an
    'images' column.

    Args:
         df (Dask DataFrame): DataFrame containing image data in the 'images'
                              column.

    Returns:
         df (Dask DataFrame): DataFrame with an additional column
                              "image_segmentation_maps" containing
                              segmentation maps for each image in the 'images'
                              column.
    """
    df["image_segmentation_maps"] = df["images"].apply(
        generate_segmentation_map, meta=("images", "object")
    )

    return df
