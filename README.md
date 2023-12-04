<a id="top"></a>
<p align="center">
    <img src="https://raw.githubusercontent.com/ml6team/fondant/main/docs/art/fondant_banner.svg" height="250px"/>
</p>
<p align="center">
    <i>
        <b>Production-ready</b> 
        data processing made 
        <b>easy</b> 
        and 
        <b>shareable</b>
    </i>
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

<table>
  <thead>
    <tr>
      <th width="33%">🚀 Production-ready</th>
      <th width="33%">👶 Easy</th>
      <th width="33%">👫 Shareable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
          Benefit from built-in features such as autoscaling, data lineage, and pipeline caching, and deploy to (managed) platforms such as <i>Vertex AI</i>, <i>Sagemaker</i>, and <i>Kubeflow Pipelines</i>.
      </td>
      <td>
          Implement your custom data processing code using datastructures you know such as <i>Pandas</i> dataframes.
          Move from local development to remote deployment without any code changes.
      </td>
      <td>
          Fondant components are defined by a clear interface, which makes them reusable and shareable.<br>
          Compose your own pipeline using components available on <a href="https://fondant.ai/en/latest/components/hub/"><b>our hub</b></a>.
      </td>
    </tr>
  </tbody>
</table>
<br>

## 🪤 Why Fondant?

With the advent of transfer learning and now foundation models, everyone has started sharing and 
reusing machine learning models. Most of the work now goes into building data processing 
pipelines, which everyone still does from scratch. 
This doesn't need to be the case, though, if processing components would be shareable and pipelines 
composable. Realizing this is the main vision behind Fondant.

Towards that end, Fondant offers:

- 🔧 Plug ‘n’ play composable data processing pipelines
- 🧩 Library containing off-the-shelf reusable components
- 🐼 A simple Pandas based interface for creating custom components
- 📊 Built-in lineage, caching, and data explorer
- 🚀 Production-ready, scalable deployment
- ☁️ Integration with runners across different clouds (Vertex, Sagemaker, Kubeflow)

<p align="right">(<a href="#top">back to top</a>)</p>

## 💨 Getting Started

Eager to get started? Follow our [**step by step guide**](https://fondant.ai/en/latest/guides/first_pipeline/) to get your first pipeline up and running.

<p align="right">(<a href="#top">back to top</a>)</p>

## 🧩 Reusable components

Fondant comes with a library of reusable components that you can leverage to compose your own 
pipeline:

- Data ingestion: _S3, GCS, ABS, Hugging Face, local file system, ..._
- Data Filtering: _Duplicates, language, visual style, topic, format, aesthetics, NSFW, license, 
  ..._
- Data Enrichment: _Captions, segmentations, embeddings, ..._
- Data Transformation: _Image cropping, image resizing, text chunking, ...._
- Data retrieval: _Common Crawl, LAION, ..._

👉 **Check our [Component Hub](https://fondant.ai/en/latest/components/hub/) for an overview of all 
available components**

<p align="right">(<a href="#top">back to top</a>)</p>

## 🪄 Example pipelines

We have created several ready-made example pipelines for you to use as a starting point for exploring Fondant. 
They are hosted as separate repositories containing a notebook tutorial so you can easily clone them and get started:

📖 [**RAG ingestion pipeline**](https://github.com/ml6team/fondant-usecase-RAG)  
An end-to-end Fondant pipeline that prepares documents for a RAG (Retrieval Augmented Generation) system by chunking and embedding them, and writing them to a vector store.

🛋️ [**ControlNet Interior Design Pipeline**](https://github.com/ml6team/fondant-usecase-controlnet)  
An end-to-end Fondant pipeline to collect and process data for the fine-tuning of a ControlNet model, focusing on images related to interior design.

🖼️ [**Filter creative common license images**](https://github.com/ml6team/fondant-usecase-filter-creative-commons)  
An end-to-end Fondant pipeline that starts from our Fondant-CC-25M creative commons image dataset and filters and downloads the desired images.

🔢 [**Datacomp pipeline**](https://github.com/ml6team/fondant-usecase-datacomp)  
An end-to-end Fondant pipeline filtering image-text data to train a CLIP model for the DataComp competition.

## ⚒️ Installation

Fondant can be installed using pip:

```
pip install fondant
```

For more detailed installation options, check the [**installation page**](https://fondant.ai/en/latest/guides/installation/) on our documentation.


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
    arguments={
        "dataset_name": "lambdalabs/pokemon-blip-captions"
    },
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

**_fondant_component.yaml_**
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

**_component/src/main.py_**
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

## 👭 Contributing

We welcome contributions of different kinds:

|                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Issues**                       | If you encounter any issue or bug, please submit them as a [Github issue](https://github.com/ml6team/fondant/issues). You can also submit a pull request directly to fix any clear bugs.                                                                                                                                                                                                                                                                                      |
| **Suggestions and feedback**     | Our roadmap and priorities are defined based on community feedback. To provide input, you can join [our discord](https://discord.gg/HnTdWhydGp) or submit an idea in our [Github Discussions](https://github.com/ml6team/fondant/discussions/categories/ideas).                                                                                                                                                                                                               |
| **Framework code contributions** | If you want to help with the development of the Fondant framework, have a look at the issues marked with the [good first issue](https://github.com/ml6team/fondant/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) label. If you want to add additional functionality, please submit an issue for it first.                                                                                                                                                     |
| **Reusable components**          | Extending our library of reusable components is a great way to contribute. If you built a component which would be useful for other users, please submit a PR adding them to the [components/](https://github.com/ml6team/fondant/tree/main/components) directory. You can find a list of possible contributable components [here](https://github.com/ml6team/fondant/issues?q=is%3Aissue+is%3Aopen+label%3A%22Components%22) or your own ideas are also welcome! |

For a detailed view on the roadmap and day to day development, you can check our [github project
board](https://github.com/orgs/ml6team/projects/1).

You can also check out our [architecture](docs/architecture.md) page to famliarize yourself with the Fondant architecture and repository structure.

### Environment setup

We use [poetry](https://python-poetry.org/docs/) and [pre-commit](https://pre-commit.com/) to enable a smooth developer flow. Run the following commands to
set up your development environment:

```shell
pip install poetry
poetry install --all-extras
pre-commit install
```

<p align="right">(<a href="#top">back to top</a>)</p>
