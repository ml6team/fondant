<p align="center">
    <img src="docs/art/fondant_banner.svg" height="250px"/>
</p>
<p align="center">
    <i>Sweet data-centric foundation model fine-tuning</i>
    <br>
    <a href="https://fondant.readthedocs.io/en/latest/"><strong>Explore the docs »</strong></a>
    <br>
    <br>
    <a href="https://discord.gg/HnTdWhydGp"><img alt="Discord" src="https://img.shields.io/discord/1108377020821405707?label=discord&style=flat-square"></a>
    <a href="https://pypi.org/project/fondant/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/fondant?color=brightgreen&style=flat-square"></a>
    <a href="https://fondant.readthedocs.io/en/latest/license/"><img alt="License" src="https://img.shields.io/github/license/ml6team/fondant?style=flat-square&color=brightgreen"></a>
    <a href="https://github.com/ml6team/fondant/actions/workflows/pipeline.yaml"><img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/ml6team/fondant/pipeline.yaml?style=flat-square"></a>
    <a href="https://coveralls.io/github/ml6team/fondant?branch=main"><img alt="Coveralls" src="https://img.shields.io/coverallsCoverage/github/ml6team/fondant?style=flat-square"></a>
</p>

---
**Fondant helps you create high quality datasets to fine-tune foundation models such as:**

- :art: Stable Diffusion  
- :page_facing_up: GPT-like Large Language Models (LLMs)  
- :mag_right: CLIP  
- :scissors: Segment Anything (SAM)  
- :heavy_plus_sign: And many more

## :mouse_trap: Why Fondant?

Foundation models simplify inference by solving a plethora of tasks across modalities with a 
prompt-based interface. But what they've gained in the front, they've lost in the back. 
**These models require enormous amounts of data, moving complexity towards data preparation**, and 
leaving few parties able to train their own models.

We believe that **innovation is a group effort**, requiring collaboration. While the community has 
been building and sharing models, everyone is still building their data preparation from scratch.
**Fondant is the platform where we meet to build and share data preparation workflows.**

Fondant offers a framework to build **composable data preparation pipelines, with reusable 
components, optimized to handle massive datasets**. Stop building from scratch, and start 
leveraging reusable components instead to:
- Extend your data with public datasets
- Generate new modalities using captioning, segmentation, translation, image generation, ...
- Distill knowledge from existing foundation models
- Filter out low quality data
- Deduplicate data

And create high quality datasets to fine-tune your own foundation models.

<p align="right">(<a href="#chocolate_bar-fondant">back to top</a>)</p>

## :magic_wand: Example pipelines

Curious to see what Fondant can do? Have a look at our example pipelines:

### [Fine-tuning ControlNet for interior design](examples/pipelines/controlnet-interior-design)

| Input image                                                    | Output image                                                     |
|----------------------------------------------------------------|------------------------------------------------------------------|
| ![input image](docs/art/interior_design_controlnet_input1.png) | ![output image](docs/art/interior_design_controlnet_output1.png) |

Want to try out the resulting model yourself, head over to our 
[Hugging Face space](https://huggingface.co/spaces/ml6team/controlnet-interior-design)!

### Fine-tuning Stable Diffusion on a specific domain

TODO

<p align="right">(<a href="#chocolate_bar-fondant">back to top</a>)</p>

## :jigsaw: Reusable components

Fondant comes with a library of reusable components, which can jumpstart your pipeline.

TODO: add links

| Component                       | Description                                                      |
|---------------------------------|------------------------------------------------------------------|
| load_from_hf_hub                | Load a dataset from the Hugging Face Hub                         |
| image_clip_embedding            | Create CLIP embeddings for images                                |
| embedding_based_laion_retrieval | Retrieve images-text pairs from LAION using embedding similarity |
| prompt_based_laion_retrieval    | Retrieve images-text pairs from LAION using prompt similarity    |
| image_downloading               | Download images from urls                                        |
| image_resolution_filtering      | Filter images based on their resolution                          |
| captioning                      | Generate captions for images                                     |
| segmentation                    | Generate segmentation maps for images                            |
| write_to_hf_hub                 | Write a dataset to the Hugging Face Hub.                         |

<p align="right">(<a href="#chocolate_bar-fondant">back to top</a>)</p>

## :hammer_and_pick: Installation

Fondant can be installed using pip:

```
pip install fondant
```

For the latest development version, you might want to install from source instead:

```
pip install git+https://github.com/ml6team/fondant.git
```

### Deploying Fondant

TODO

<p align="right">(<a href="#chocolate_bar-fondant">back to top</a>)</p>

## :man_technologist: Usage

#### Pipeline

Fondant allows you to easily define data pipelines comprised of both reusable and custom 
components. The following pipeline for instance uses the reusable `load_from_hf_hub` component 
to load a dataset from the Hugging Face Hub and process it using a custom component:


```python
from fondant.pipeline import ComponentOp, Pipeline, Client


def build_pipeline():
    pipeline = Pipeline(pipeline_name="example pipeline", base_path="fs://bucket")

    load_from_hub_op = ComponentOp.from_registry(
        name="load_from_hf_hub",
        arguments={"dataset_name": "lambdalabs/pokemon-blip-captions"},
    )
    pipeline.add_op(load_from_hub_op)

    custom_op = ComponentOp(
        component_spec_path="components/custom_component/fondant_component.yaml",
        arguments={
            "min_width": 600,
            "min_height": 600,
        },
    )
    pipeline.add_op(custom_op, dependencies=load_from_hub_op)

    return pipeline
    

if __name__ == "__main__":
    client = Client(host="https://kfp-host.com/")
    pipeline = build_pipeline()
    client.compile_and_run(pipeline=pipeline)
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

Once you have your component specification, all you need to do is implement a single `.transform` 
method and Fondant will do the rest. You will get the data defined in your specification as a 
[Dask](https://www.dask.org/) dataframe, which is evaluated lazily.

```python
from fondant.component import TransformComponent

class ExampleComponent(TransformComponent):

    def transform(self, dataframe, *, argument1, argument2):
        """Implement your custom logic in this single method
        
        Args:
            dataframe: A Dask dataframe containing the data
            argumentX: An argument passed to the component
        """
```

<p align="right">(<a href="#chocolate_bar-fondant">back to top</a>)</p>

## :construction: Current state and roadmap

Fondant is currently in the alpha stage, offering a minimal viable interface. While you should 
expect to run into rough edges, the foundations are ready and Fondant should already be able to 
speed up your data preparation work.

**The following topics are on our roadmap**
- Local pipeline execution
- Non-linear pipeline DAGs
- LLM-focused example pipelines and reusable components
- Static validation, caching, and partial execution of pipelines
- Data lineage and experiment tracking
- Distributed execution, both on and off cluster
- Support other dataframe libraries such as HF Datasets, Polars, Spark
- Move reusable components to decentralized component registry

The roadmap and priority are defined based on community feedback. To provide input, you can join 
[our discord](https://discord.gg/HnTdWhydGp) or submit an idea in our 
[Github Discussions](https://github.com/ml6team/fondant/discussions/categories/ideas).

For a detailed view on the roadmap and day to day development, you can check our [github project 
board](https://github.com/orgs/ml6team/projects/1).

<p align="right">(<a href="#chocolate_bar-fondant">back to top</a>)</p>

## :two_women_holding_hands: Contributing

We welcome contributions of different kinds:

|                                  |                                                                                                                                                                                                                                                                                                                           |
|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Issues**                       | If you encounter any issue or bug, please submit them as a [Github issue](https://github.com/ml6team/fondant/issues). You can also submit a pull request directly to fix any clear bugs.                                                                                                                                  |
| **Suggestions and feedback**     | If you have any suggestions or feedback, please reach out via our [Discord server](https://discord.gg/HnTdWhydGp) or [Github Discussions](https://github.com/ml6team/fondant/discussions)!                                                                                                                                |
| **Framework code contributions** | If you want to help with the development of the Fondant framework, have a look at the issues marked with the [good first issue](https://github.com/ml6team/fondant/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) label. If you want to add additional functionality, please submit an issue for it first. |
| **Reusable components**          | Extending our library of reusable components is a great way to contribute. If you built a component which would be useful for other users, please submit a PR adding them to the [components/](https://github.com/ml6team/fondant/tree/main/components) directory.                                                        |
| **Example pipelines**            | If you built a pipeline with Fondant which can serve as an example to other users, please submit a PR adding them to the [examples/](https://github.com/ml6team/fondant/tree/main/examples) directory.                                                                                                                    |

### Environment setup

We use [poetry](https://python-poetry.org/docs/) and [pre-commit](https://pre-commit.com/) to enable a smooth developer flow. Run the following commands to
set up your development environment:

```commandline
pip install poetry
poetry install
pre-commit install
```

<p align="right">(<a href="#chocolate_bar-fondant">back to top</a>)</p>