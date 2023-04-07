"""
This component adds a data source to the manifest by embedding the images.
"""
import logging

from express.components.hf_datasets_components import HFDatasetsTransformComponent
from express.components.common import Manifest
from express.logger import configure_logging

import torch

from transformers import CLIPProcessor, CLIPVisionModelWithProjection

configure_logging()
logger = logging.getLogger(__name__)


cuda_available = torch.cuda.is_available()
device = "cuda" if cuda_available else "cpu"
logger.info("CUDA device availability:%s", cuda_available)

if cuda_available:
    logger.info(torch.cuda.get_device_name(0))
    logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    logger.info("Num of GPUs: %s", torch.cuda.device_count())


@torch.no_grad()
def embed(examples, processor, model):
    images = examples["image"]

    # prepare images for the model
    inputs = processor(images=images, return_tensors="pt").to(device)

    # embed to get (batch_size, hidden_size) embeddings
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds

    # flatten into list of embeddings
    examples["embeddings"] = image_embeds.cpu().tolist()

    return examples


class EmbeddingComponent(HFDatasetsTransformComponent):
    """
    Component that embeds the images using a CLIP model from Hugging Face.
    """

    @classmethod
    def transform(
        cls,
        manifest,
        args,
    ) -> Manifest:
        """
        Args:
            manifest (HFDatasetsDataset[TIndex, TData]): Fondant manifest
            args (Optional[Dict[str, Union[str, int, float, bool]]): optional args to pass to
             the function
        Returns:
            Manifest: output manifest
        """

        # 1) Get one particular data source from the manifest
        # TODO check whether we can leverage streaming
        logger.info("Loading image dataset...")
        image_dataset = manifest.load(data_source="images")

        # 2) Create embedding dataset
        logger.info("Loading CLIP...")
        processor = CLIPProcessor.from_pretrained(args["model_id"])
        model = CLIPVisionModelWithProjection.from_pretrained(args["model_id"])
        model.to(device)

        logger.info("Embedding images...")
        embedded_dataset = image_dataset.map(
            embed,
            batched=True,
            batch_size=args["batch_size"],
            fn_kwargs=dict(processor=processor, model=model),
            remove_columns=["image", "width", "height", "byte_size"],
        )

        # 3) Create output manifest
        logger.info("Creating output manifest...")
        data_sources = {"embeddings": embedded_dataset}
        output_manifest = manifest.add_data_sources(data_sources)

        return output_manifest


if __name__ == "__main__":
    EmbeddingComponent.run()
