# Creating custom components

Fondant makes it easy to build data preparation pipelines leveraging reusable components. Fondant
provides a lot of components out of the box
([overview](https://github.com/ml6team/fondant/tree/main/components), but you can also define your
own custom components.

To make sure components are reusable, they should implement a single logical data processing
step (like captioning images or removing Personal Identifiable Information [PII] from text.)
If a component grows too large, consider splitting it into multiple separate components each
tackling one logical part.

To implement a custom component, a couple of files need to be defined:

- [Fondant component specification](#fondant-component-specification)
- [`main.py` script in a `src` folder](#mainpy-script)
- [Dockerfile](#dockerfile)
- [requirements.txt](#requirementstxt)

## Fondant component specification

Each Fondant component is defined by a specification which describes its interface. This
specification is represented by a single `fondant_component.yaml` file. See the [component
specification page](../components/component_spec.md) for info on how to write the specification for your component.

## Main.py script

The core logic of the component should be implemented in a `main.py` script in a folder called
`src`.
The logic should be implemented as a class, inheriting from one of the base `Component` classes
offered by Fondant.
There are three large types of components:

- **`LoadComponent`**: Load data into your pipeline from an external data source
- **`TransformComponent`**: Implement a single transformation step in your pipeline
- **`WriteComponent`**: Write the results of your pipeline to an external data sink

The easiest way to implement a `TransformComponent` is to subclass the provided
`PandasTransformComponent`. This component streams your data and offers it in memory-sized
chunks as pandas dataframes.

```python
import pandas as pd
from fondant.component import PandasTransformComponent
from fondant.executor import PandasTransformExecutor


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
            dataframe: A Pandas dataframe containing one partition of your data
        Returns:
            A pandas dataframe containing the transformed data
        """
```

The `__init__` method is called once for each component class with custom arguments defined in the
`args` section of the [component specification](../components/component_spec.md).)

The `transform` method is called multiple times, each time containing a pandas `dataframe`
with a partition of your data loaded in memory.

The `dataframes` passed to the `transform` method contains the data specified in the `consumes`
section of the component specification. If a component defines that it consumes an `images` subset
with a `data` field, this data can be accessed using `dataframe["images"]["data"]`.

The `transform` method should return a single dataframe, with the columns complying to the
`[subset][field]` format matching the `produces` section of the component specification.

Note that the `main.py` script can be split up into several Python scripts in case it would become
prohibitively long. See the
[prompt based LAION retrieval component](https://github.com/ml6team/fondant/tree/main/components/prompt_based_laion_retrieval/src)
as an example: the CLIP client itself is defined in a separate script called `clip_client`,
which is then imported in the `main.py` script.

## Dockerfile

The `Dockerfile` defines how to build the component into a Docker image. An example Dockerfile is defined below.

```bash
FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

## System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y

# install requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the working directory to the component folder
WORKDIR /component/src

# Copy over src-files and spec of the component
COPY src/ .

ENTRYPOINT ["fondant", "execute", "main"]
```

## Requirements.txt

A `requirements.txt` file lists the Python dependencies of the component. Note that any Fondant component will always have `Fondant` as the minimum requirement. It's important to also pin the version of each dependency to make sure the component remains working as expected. Below is an example of a component that relies on several Python libraries such as Pillow, PyTorch and Transformers.

```
fondant
Pillow==10.0.1
torch==2.0.1
transformers==4.29.2
```
