from clip_retrieval.clip_client import ClipClient, Modality


def get_image_urls(text, client):
    """
    Given a text query and a ClipClient instance, this function retrieves
    image URLs related to the query.

    Args:
        text (str): the text query
        client (ClipClient): an instance of ClipClient used to query the
                             images

    Returns:
        results (list): a list of strings, each representing a URL of an image
                        related to the query
    """
    results = client.query(text=text)
    results = [i["url"] for i in results]

    return results


def retrieve_images(df):
    """
    Given a Pandas DataFrame containing a column "prompt_data" with text
    queries, this function retrieves image URLs related to each query from the
    LAION-5B dataset, using a ClipClient instance.

    Args:
        df (Dask DataFrame): a Dask DataFrame containing a column
                             "prompt_data" with text queries

    Returns:
        df (Dask DataFrame): a Dask DataFrame with an additional column
                             "image_urls", where each row contains a list of
                             URLs of images related to the query in that row
    """

    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
        num_images=1000,  # TODO: include as argument and increase for
                          # the purposes of scaling
        aesthetic_score=9,
        aesthetic_weight=0.5,
        modality=Modality.IMAGE,
    )

    df['image_urls'] = df['prompt_data'].map_partitions(
        lambda x: x.apply(get_image_urls, args=(client,)),
        meta=('image_urls', 'object')
    )

    return df
