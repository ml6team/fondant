"""
This component extends the images data source by retrieving similar images from LAION.
"""
import logging
from typing import Optional, Union, Dict

from utils.embedding_utils import get_average_embedding

from express.components.hf_datasets_components import (
    HFDatasetsTransformComponent,
    HFDatasetsDataset,
    HFDatasetsDatasetDraft,
)
from express.logger import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


class CLIPRetrievalComponent(HFDatasetsTransformComponent):
    """
    Component that retrieves similar images from the LAION dataset.
    """

    @classmethod
    def transform(
        cls,
        data: HFDatasetsDataset,
        extra_args: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> HFDatasetsDatasetDraft:
        """
        An example function showcasing the data transform component using Express functionalities

        Args:
            data (HFDatasetsDataset[TIndex, TData]): express dataset providing access to data of a
             given type
            extra_args (Optional[Dict[str, Union[str, int, float, bool]]): optional args to pass to
             the function
        Returns:
            HFDatasetsDatasetDraft: a dataset draft that creates a plan for an output manifest
        """

        # 1) Get one particular data source from the manifest
        # TODO check whether we can leverage streaming
        # TODO just column in the load method in the future
        logger.info("Loading embedding dataset...")
        dataset = data.load(data_source="embeddings", columns=["index", "embeddings"])

        logger.info("Calculating average embedding...")
        centroid_embedding = get_average_embedding(dataset)

        print("Centroid embedding:", centroid_embedding)

        # # This component uses local SSD mounting to enable faster querying
        # of the laion dataset. The
        # # local SSDs are mounted in the cache directory
        # laion_index_folder = os.path.join('/cache', 'laion_dataset')
        # laion_metadata_folder = os.path.join(laion_index_folder, 'metadata')
        # laion_indices_folder = os.path.join(laion_index_folder, 'image.index')
        # os.makedirs(laion_metadata_folder, exist_ok=True)
        # os.makedirs(laion_indices_folder, exist_ok=True)

        #  # Download laion indices and metadata from storage
        # start_indices = time.time()
        # copy_folder(extra_args["laion_index_url"], laion_indices_folder)
        # logger.info('Laion index download complete: it took %s minutes to download the
        # laion indices',
        #             round((time.time() - start_indices) / 60))
        # start_metadata = time.time()
        # copy_folder(extra_args["laion_metadata_url"], laion_metadata_folder)
        # logger.info(
        #     'Laion metadata download complete: it took %s minutes to download the laion metadata',
        #     round((time.time() - start_metadata) / 60))

        # # Setup KNN service
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     tmp_embedding_dir_path = os.path.join(tmp_dir, 'embed_dir')
        #     os.makedirs(tmp_embedding_dir_path, exist_ok=True)
        #     laion_indices_folder = os.path.join(tmp_dir, 'index_file.json')
        #     clip_retrieval_runner = ClipRetrievalLaion5B(
        #             laion_index_path=laion_indices_folder,
        #             laion_index_folder=laion_index_folder)
        #     knn_service = clip_retrieval_runner.setup_knn_service()

        # logger.info('Starting centroid clip retrieval')

        # # Run clip retrieval with centroid approach
        # results_centroid = clip_retrieval_runner.run_query(
        #     knn_service=knn_service,
        #     query={'embedding_query': centroid_embedding},
        #     nb_images_request=extra_args["nb_images_centroid"],
        #     deduplicate=True,
        #     benchmark=True)
        # logger.info('Centroid clip retrieval complete')

        # # 3) Create dataset draft
        # logger.info("Creating draft...")
        # data_sources = {}
        # dataset_draft = HFDatasetsDatasetDraft(
        #     data_sources=data_sources, extending_dataset=data
        # )
        # return dataset_draft
        return None


if __name__ == "__main__":
    CLIPRetrievalComponent.run()
