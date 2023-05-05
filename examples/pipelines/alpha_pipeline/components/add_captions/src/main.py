import dask.dataframe as dd
from dask.delayed import delayed
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import io


processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")


@delayed
def generate_caption(image_bytes):
    """
    Generate a caption for the input image bytes using a pre-trained model.

    Args:
        image_bytes (bytes): The input image bytes to generate a caption for.

    Returns:
        str: The generated caption for the input image bytes, or None if 
             image_bytes is None.
    """
    if image_bytes is None:
        return None

    image = Image.open(io.BytesIO(image_bytes))
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]

    return generated_caption


def add_captions(df):
    """
    Generate captions for all images in a Dask DataFrame using the 
    generate_caption function.

    Args:
        df (dask.dataframe.DataFrame): A Dask DataFrame containing a column
                                       named 'images' with image bytes.

    Returns:
        dask.dataframe.DataFrame: The input DataFrame with an additional 
                                  'image_captions' column containing
                                  generated captions.
    """
    df["image_captions"] = df['images'].apply(
        generate_caption, meta=('image_captions', 'object'))

    return df
