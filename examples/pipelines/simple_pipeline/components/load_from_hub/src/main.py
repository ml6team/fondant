"""
This component loads a seed dataset from the hub.
"""
import logging

from datasets import load_dataset

from express.dataset import FondantComponent
from express.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def create_image_metadata(batch):
    images = batch["image"]

    # add width and height columns
    widths, heights = zip(*[image.size for image in images])
    batch["images_width"] = list(widths)
    batch["images_height"] = list(heights)

    return batch


class LoadFromHubComponent(FondantComponent):
    """Component that loads a dataset from the hub and adds it to the manifest."""
    type = "load"

    @classmethod
    def load(cls, args):
        """
        Args:
            args: additional arguments passed to the component
        
        Returns:
            Dataset: HF dataset
        """
        # 1) Create data source
        logger.info("Loading caption dataset from the hub...")
        dataset = load_dataset(args.dataset_name, split="train")

        # 2) Create index
        logger.info("Creating index...")
        index_list = [idx for idx in range(len(dataset))]

        # 3) Add index to the dataset, rename columns
        dataset = dataset.add_column("id", index_list)
        
        # 4) Add metadata columns
        dataset = dataset.map(
            create_image_metadata,
            batched=True,
            batch_size=args.batch_size,
        )

        # 5) Rename columns
        dataset = dataset.rename_column("image", "images_data")
        dataset = dataset.rename_column("text", "captions_data")
        
        return dataset


if __name__ == "__main__":
    LoadFromHubComponent.run()