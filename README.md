<a id="top"></a>
<p align="center">
    <img src="https://raw.githubusercontent.com/ml6team/fondant/main/docs/art/fondant_banner.svg" height="250px"/>
</p>
<p align="center">
    <i>Large-scale data processing made easy and reusable</i>
    <br>
    <a href="http://fondant.ai"><strong>Explore the docs »</strong></a>
    <br>
    <br>
    <a href="https://discord.gg/HnTdWhydGp"><img alt="Discord" src="https://dcbadge.vercel.app/api/server/HnTdWhydGp?style=flat-square"></a>
    <a href="https://pypi.org/project/fondant/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/fondant?color=brightgreen&style=flat-square"></a>
    <a href="https://fondant.readthedocs.io/en/latest/license/"><img alt="License" src="https://img.shields.io/github/license/ml6team/fondant?style=flat-square&color=brightgreen"></a>
    <a href="https://github.com/ml6team/fondant/actions/workflows/pipeline.yaml"><img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/ml6team/fondant/pipeline.yaml?style=flat-square"></a>
    <a href="https://coveralls.io/github/ml6team/fondant?branch=main"><img alt="Coveralls" src="https://img.shields.io/coverallsCoverage/github/ml6team/fondant?style=flat-square"></a>
</p>

---

🍫**Fondant is an open-source framework that simplifies and speeds up data processing development
through reusable components**

It offers:

<ul>
<li>
    🔧 Plug ‘n’ play composable pipelines for creating datasets for:
    <ul>
        <li>AI image generation model fine-tuning (Stable Diffusion, ControlNet)</li>
        <li>Large language model fine-tuning (LLaMA, Falcon)</li>
    </ul>
</li>
<li>
🧱 Off-the-shelf library of easily adaptable reusable components for:
    <ul>
    <li>
    Data Filtering
    <ul>
        <li>Content: language, visual style, topic, format, aesthetics, NSFW, etc.</li>
        <li>Context: e.g. copyright license, origin, etc.</li>
        <li>Deduplication</li>
    </ul>
    </li>
    <li>
    Data Enrichment
    <ul>
        <li>Captions</li>
        <li>Segmentations</li>
        <li>Embeddings</li>
    </ul>
    </li>
    <li>
    Data Transformation
    <ul>
        <li>Images: cropping, resizing, etc. </li>
        <li>Text: chunking, indexing, etc. </li>
    </ul>
    </li>
    <li>
    Data retrieval
    <ul>
        <li>Common Crawl</li>
        <li>LAION</li>
    </ul>
    </li>
    </ul>
</li>
<li>📖 🖼️ 🎞️ ♾️ Out of the box multimodal capabilities: text, images, video, etc.</li>
<li>🐍 A simple Pandas based interface for creating custom components</li>
<li>🏭 Scalable deployment for production environments.</li>
<li>☁️ Integration with runners across different clouds (Vertex, Sagemaker, Kubeflow)</li>
</ul>

## 🪤 Why Fondant?

In the era of Foundation Models, preserving data control has become paramount. 
Yet, the construction of pipelines for extensive data processing remains a costly affair, particularly in the context of 
complex machine learning-based operations. 
Fondant's mission is to streamline this process by introducing reusable and interchangeable processing components, 
thus establishing easily composed pipelines as the new norm.

<p align="right">(<a href="#top">back to top</a>)</p>

## 💨 Getting Started

Eager to get started? Here is a [step by step guide](https://fondant.readthedocs.io/en/latest/getting_started) to get your first pipeline up and running.

## 🪄 Example pipelines

Curious to see what Fondant can do? Have a look at our example pipelines:

### Filtering creative commons image dataset

We have published an [image dataset containing 25 million images](https://huggingface.co/datasets/fondant-ai/fondant-cc-25m).
As a result, we have provided a [sample pipeline](https://github.com/ml6team/fondant/tree/main/examples/pipelines/filter-cc-25m) that
demonstrates the download and filtering of these images. In the pipeline folder,
you will find detailed instructions on how to execute the pipeline and explore the images.

### Fine-tuning ControlNet

Our
[example pipeline to generate data for ControlNet fine-tuning](https://github.com/ml6team/fondant/tree/main/examples/pipelines/controlnet-interior-design)
allows you to create models that you can control using inpainting, segmentation, and
regeneration. All you need to get started is a set of prompts describing the type of images to
generate.

For instance, using our ControlNet model fine-tuned on interior design images, allows you to
generate the room of your dreams:

| Input image                                                                                                          | Output image                                                                                                           |
| -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| ![input image](https://github.com/ml6team/fondant/blob/main/docs/art/interior_design_controlnet_input1.png?raw=true) | ![output image](https://github.com/ml6team/fondant/blob/main/docs/art/interior_design_controlnet_output1.jpg?raw=true) |

Want to try out the resulting model yourself, head over to our
[Hugging Face space](https://huggingface.co/spaces/ml6team/controlnet-interior-design)!

### Fine-tuning Stable Diffusion

Using our
[example pipeline to fine-tune Stable Diffusion](https://github.com/ml6team/fondant/tree/main/examples/pipelines/finetune_stable_diffusion)
allows you to create models that generate better images within a specific domain. All you need to
get started is a small seed dataset of example images.

Eg. generating logos:

| Stable Diffusion 1.5                                                                                     | Fine-tuned Stable Diffusion 1.5                                                                     |
| -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| ![input image](https://github.com/ml6team/fondant/blob/main/docs/art/logo_stable_diffusion.jpg?raw=true) | ![output image](https://github.com/ml6team/fondant/blob/main/docs/art/logo_finetuned.jpeg?raw=true) |

<p align="right">(<a href="#top">back to top</a>)</p>

## 🧩 Reusable components

Fondant comes with a library of reusable components, which can jumpstart your pipeline, here are a selected few:

| COMPONENT                                                                                                                  | DESCRIPTION                                                                 |
|----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Data loading**                                                                                                           |                                                                             |
| [load_from_hf_hub](https://github.com/ml6team/fondant/tree/main/components/load_from_hf_hub)                               | Load a dataset from the Hugging Face Hub                                    |
| [load_from_parquet](https://github.com/ml6team/fondant/tree/main/components/load_from_parquet)                             | Load a dataset from a parquet file stored on remotely                       |
| **Data Retrieval**                                                                                                         |                                                                             |
| [prompt_based_laion_retrieval](https://github.com/ml6team/fondant/tree/main/components/prompt_based_laion_retrieval)       | Retrieve images-text pairs from LAION using prompt similarity               |
| [embedding_based_laion_retrieval](https://github.com/ml6team/fondant/tree/main/components/embedding_based_laion_retrieval) | Retrieve images-text pairs from LAION using embedding similarity            |
| [download_images](https://github.com/ml6team/fondant/tree/main/components/download_images)                                 | Download images from urls                                                   |
| **Data Writing**                                                                                                           |                                                                             |  
| [write_to_hf_hub](https://github.com/ml6team/fondant/tree/main/components/write_to_hf_hub)                                 | Write a dataset to the Hugging Face Hub                                     |
| [index_weaviate](https://github.com/ml6team/fondant/tree/main/components/index_weaviate)                                   | Index text and writes it to a [Weaviate](https://weaviate.io/) database     |
| **Image processing**                                                                                                       |                                                                             |
| [embed_images](https://github.com/ml6team/fondant/tree/main/components/embed_images)                                       | Create embeddings for images using a model from the HF Hub                  |
| [image_resolution_extraction](https://github.com/ml6team/fondant/tree/main/components/image_resolution_extraction)         | Extract the resolution from images                                          |
| [filter_image_resolution](https://github.com/ml6team/fondant/tree/main/components/filter_image_resolution)                 | Filter images based on their resolution                                     |
| [caption images](https://github.com/ml6team/fondant/tree/main/components/caption_images)                                   | Generate captions for images using a model from the HF Hub                  |
| [segment_images](https://github.com/ml6team/fondant/tree/main/components/segment_images)                                   | Generate segmentation maps for images using a model from the HF Hub         |
| [image_cropping](https://github.com/ml6team/fondant/tree/main/components/image_cropping)                                   | Intelligently crop out image borders                                        |
| **Text processing**                                                                                                        |                                                                             |
| [embed_text](https://github.com/ml6team/fondant/tree/main/components/embed_text)                                           | Create embeddings for images using a model from the HF Hub                  |
| [chunk_text](https://github.com/ml6team/fondant/tree/main/components/chunk_text)                                           | Extract chunks from long text paragraphs                                    |
| [normalize_text](https://github.com/ml6team/fondant/tree/main/components/normalize_text)                                   | Implements several normalization techniques to clean and preprocess textual |
| [filter_text_length](https://github.com/ml6team/fondant/tree/main/components/filter_text_length)                           | Filters text based on character length                                      |


<p align="right">(<a href="#top">back to top</a>)</p>

Check out the [components](https://github.com/ml6team/fondant/tree/main/components) section for a full list of available components.
## ⚒️ Installation

Fondant can be installed using pip:

```
pip install fondant
```

For the latest development version, you might want to install from source instead:

```
pip install git+https://github.com/ml6team/fondant.git
```

### 🧱 Running Fondant pipelines

Fondant pipelines can be run on different platforms.

<style>
    .caption {
        text-align: center;
    }
    .caption strong {
        font-weight: bold;
    }
</style>

<table class="images" width="100%" style="border: 0px solid white; width: 100%;">
    <tr style="border: 0px;">
        <td width="33.33%" style="border: 0px; width: 33.33%">
            <figure>
                <img src="https://github.com/ml6team/fondant/blob/main/docs/art/runners/docker_compose.png?raw=true" />
                <figcaption class="caption"><strong>LocalRunner</strong>: Uses Docker Compose to run locally on your machine – great for developing, testing, and debugging.</figcaption>
            </figure>
        </td>
        <td width="33.33%" style="border: 0px; width: 33.33%">
            <figure>
                <img src="https://github.com/ml6team/fondant/blob/main/docs/art/runners/vertex_ai.png?raw=true" />
                <figcaption class="caption"><strong>VertexRunner</strong>: Runs on VertexAI Pipelines.</figcaption>
            </figure>
        </td>
        <td width="33.33%" style="border: 0px; width: 33.33%">
            <figure>
                <img src="https://github.com/ml6team/fondant/blob/main/docs/art/runners/kubeflow_pipelines.png?raw=true" />
                <figcaption class="caption"><strong>KubeflowRunner</strong>: Runs on Kubeflow Pipelines</figcaption>
            </figure>
        </td>
    </tr>
</table>

<p align="right">(<a href="#top">back to top</a>)</p>

## 👨‍💻 Usage

#### Pipeline

Fondant allows you to easily define data pipelines comprised of both reusable and custom
components. The following pipeline for instance uses the reusable `load_from_hf_hub` component
to load a dataset from the Hugging Face Hub and process it using a custom component:

**_pipeline.py_**
```python
from fondant.pipeline import ComponentOp, Pipeline


pipeline = Pipeline(pipeline_name="example pipeline", base_path="fs://bucket")

load_from_hub_op = ComponentOp.from_registry(
    name="load_from_hf_hub",
    arguments={"dataset_name": "lambdalabs/pokemon-blip-captions"},
)
pipeline.add_op(load_from_hub_op)

custom_op = ComponentOp(
    component_dir="components/custom_component",
    arguments={
        "min_width": 600,
        "min_height": 600,
    },
)
pipeline.add_op(custom_op, dependencies=load_from_hub_op)
```

#### Component

To create a custom component, you first need to describe its contract as a yaml specification.
It defines the data consumed and produced by the component and any arguments it takes.

```yaml
name: Custom component
description: This is a custom component
image: custom_component:latest

consumes:
  images:
    fields:
      data:
        type: binary

produces:
  captions:
    fields:
      data:
        type: utf8

args:
  argument1:
    description: An argument passed to the component at runtime
    type: str
  argument2:
    description: Another argument passed to the component at runtime
    type: str
```

Once you have your component specification, all you need to do is implement a constructor
and a single `.transform` method and Fondant will do the rest. You will get the data defined in
your specification partition by partition as a Pandas dataframe.

```python
import pandas as pd
from fondant.component import PandasTransformComponent


class ExampleComponent(PandasTransformComponent):

    def __init__(self, *args, argument1, argument2) -> None:
        """
        Args:
            argumentX: An argument passed to the component
        """
        # Initialize your component here based on the arguments

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Implement your custom logic in this single method
        Args:
            dataframe: A Pandas dataframe containing the data
        Returns:
            A pandas dataframe containing the transformed data
        """
```

For more advanced use cases, you can use the `DaskTransformComponent` instead.

### Running your pipeline

Once you have a pipeline you can easily run (and compile) it by using the built-in CLI:

```bash
fondant run local pipeline.py
```

To see all available runner and arguments you can check the fondant CLI help pages

```bash
fondant --help
```

Or for a subcommand:

```bash
fondant <subcommand> --help
```

<p align="right">(<a href="#top">back to top</a>)</p>

## 🚧 Current state and roadmap

Fondant is currently in the alpha stage, offering a minimal viable interface. While you should
expect to run into rough edges, the foundations are ready and Fondant should already be able to
speed up your data preparation work.

**The following topics are on our roadmap**

- Non-linear pipeline DAGs
- LLM-focused example pipelines and reusable components
- Data lineage and experiment tracking
- Distributed execution, both on and off cluster
- Support other dataframe libraries such as HF Datasets, Polars, Spark
- Move reusable components into a decentralized component registry
- Create datasets of copy-right free data for fine-tuning
- Create reusable components for bias detection and mitigation

The roadmap and priority are defined based on community feedback. To provide input, you can join
[our discord](https://discord.gg/HnTdWhydGp) or submit an idea in our
[Github Discussions](https://github.com/ml6team/fondant/discussions/categories/ideas).

For a detailed view on the roadmap and day to day development, you can check our [github project
board](https://github.com/orgs/ml6team/projects/1).

<p align="right">(<a href="#top">back to top</a>)</p>

## 👭 Contributing

We welcome [contributions of different kinds](https://fondant.readthedocs.io/en/latest/contributing)

### Environment setup

We use [poetry](https://python-poetry.org/docs/) and [pre-commit](https://pre-commit.com/) to enable a smooth developer flow. Run the following commands to
set up your development environment:

```commandline
pip install poetry
poetry install
pre-commit install
```

<p align="right">(<a href="#top">back to top</a>)</p>
