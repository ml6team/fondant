# Express

Express is a framework that speeds up the creation of KubeFlow pipelines to process big datasets and train [Foundation Models](https://fsi.stanford.edu/publication/opportunities-and-risks-foundation-models)
such as:

- Stable Diffusion
- CLIP
- Large Language Models (LLMs like GPT-3)

on them.

## Installation

Express can be installed using pip:

```
pip install express
```

## Usage

Express is built upon [KubeFlow](https://www.kubeflow.org/), a cloud-agnostic framework built by Google to orchestrate machine learning workflows on Kubernetes. An important aspect of KubeFlow are pipelines, which consist of a set of components being executed, one after the other. This typically involves transforming data and optionally training a machine learning model on it. Check out [this page](https://www.kubeflow.org/docs/components/pipelines/v1/concepts/) if you want to learn more about KubeFlow pipelines and components.

Express offers ready-made components and helper functions that serve as boilerplate which you can use to speed up the creation of KubeFlow pipelines. To implement your own component, simply overwrite one of the components available in Express. In the example below, we leverage the `PandasTransformComponent` and overwrite its `transform` method.

```
import pandas as pd

from express.components.pandas_components import PandasTransformComponent, PandasDataset, PandasDatasetDraft

class MyFirstTransform(PandasTransformComponent):
    @classmethod
    def transform(cls, data: PandasDataset, extra_args: Optional[Dict] = None) -> PandasDatasetDraft:

        # Reading data
        index: List[str] = data.load_index()
        my_data: Scanner = data.load("my_data_source")

        # Transforming data
        table: pa.Table = my_data.to_table()
        df: pd.DataFrame = table.to_pandas()
        # ...
        transformed_table = pa.Table.from_pandas(df)

        # Returning output.
        return data.extend() \
            .with_index(in) \
            .with_data_source("my_transformed_data_source", \
                              Scanner.from_batches(table.to_batches())
```

## Components zoo

Available components include:

- Non-distributed Pandas components: `express.components.pandas_components.{PandasTransformComponent, PandasLoaderComponent}`

Planned components include:

- Spark-based components and base image.
- HuggingFace Datasets components.

With Kubeflow, it's possible to share and re-use components across different pipelines. To see an example, checkout this [sample notebook](https://github.com/Svendegroote91/kfp_samples/blob/master/Reusable%20Components%20101.ipynb) that showcases how you can save and load a component.

Note that Google's [AI Hub](https://aihub.cloud.google.com) also contains components that you can easily re-use. Some interesting examples:

- [Gather training data by querying BigQuery](https://aihub.cloud.google.com/p/products%2F4700cd7e-2826-4ce9-a1ad-33f4a5bf7433)
- [Bigquery to TFRecords converter](https://aihub.cloud.google.com/p/products%2F28a006d0-c833-4c68-98ff-37358eeb7726)
- [Executing an Apache Beam Python job in Cloud Dataflow](https://aihub.cloud.google.com/p/products%2F44999f4a-1668-4d42-a4e3-1269a8786840)
- [Submitting a Cloud ML training job as a pipeline step](https://aihub.cloud.google.com/p/products%2Ffbe29250-9b67-4dfb-8900-d6ce41cdb85a)
- [Deploying a trained model to Cloud Machine Learning Engine](https://aihub.cloud.google.com/p/products%2F7a08de6c-3864-4ccf-8151-4119e1b4e890)
- [Batch predicting using Cloud Machine Learning Engine](https://aihub.cloud.google.com/p/products%2F3d5d2340-0eb2-4b03-aecc-ae34f6105822)

## Pipeline zoo

To do: add ready-made pipelines.

## Examples

Example use cases of Express include:

- collect additional image-text pairs based on a few seed images and fine-tune Stable Diffusion
- filter an image-text dataset to only include "count" examples and fine-tune CLIP to improve its counting capabilities

Check out the [examples folder](examples) for some illustrations.

## Contributing

We use [poetry](https://python-poetry.org/docs/) and pre-commit to enable a smooth developer flow. Run the following commands to 
set up your development environment:

```commandline
pip install poetry
poetry install
pre-commit install
```