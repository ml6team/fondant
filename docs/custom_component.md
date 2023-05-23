# Custom component

The goal of Fondant is to make it easy for people to define new components which can be incorporated into data preparation pipelines. This allows to share and reuse them as building blocks when preparing data for Foundation models.

A Fondant component should implement one logical piece of data preprocessing, like captioning images or removing PPI from text. It's important to not include too much logic into a single component, as it might be beneficial to split up the logic into 2 separate components in that case.

To implement a custom component, a couple of files need to be defined:

- [Fondant component specification](#fondant-component-specification)
- [main.py script in a `src` folder](#mainpy-script)
- [Dockerfile](#dockerfile)
- [requirements.txt](#requirementstxt)

To see an overview of all custom components which Fondant already provides, have a look [here](https://github.com/ml6team/fondant/tree/main/components).

## Fondant component specification

Each Fondant component is defined by a specification which describes its interface. This is a single file called `fondant_component.yaml`. We refer the reader to the [component specification page](component_spec) for all details.

## Main.py script

The core logic of the component should be implemented in a `main.py` script in a folder called `src`. The component itself should be implemented as a class, inheriting from Fondant's `TransformComponent` class. The logic itself should be defined in the `transform` method, which the user can overwrite. This method defines the processing logic. It gets a single dataframe as input and should return a single Dask dataframe.

```python
from fondant.component import TransformComponent

class ExampleComponent(TransformComponent):

    def transform(self, dataframe, *, argument1, argument2):
        """Implement your custom logic in this single method
        
        Args:
            dataframe: A Dask dataframe containing the data
            argumentX: A custom argument passed to the component
        """
```

The idea is that Fondant provides a single dataframe to the user based on the component's specification. The column names of the dataframe always have the format `subset_field`. This means that if a component defines the `images` subset with the `data` field in its `consumes` [section](component_spec) for instance, the dataframe will contain a column called `images_data`.

Next, the user can manipulate this dataframe. Finally, the user should return a single dataframe, which Fondant will then use to update the [manifest](manifest) and potentially write new data to the cloud. Fondant will also verify that the output dataframe matches with the `produces` [section](component_spec) of the component specificaton.

Note that the `transform` method can include additional custom arguments (`argument1` and `argument2` in the example above). These should match with the `args` section defined in the [Fondant specification](component_spec). Examples include the batch size when captioning images, the minimum width and height when filtering images, and so on.

Note that the `main.py` script can be split up into several Python scripts in case it would become prohibitively long. See the [prompt based LAION retrieval component](https://github.com/ml6team/fondant/tree/main/components/prompt_based_laion_retrieval/src) as an example: the CLIP client itself is defined in a separate script called `clip_client`, which is then imported in the `main.py` script.

## Dockerfile

A Dockerfile defines all the commands a user could call on the command line to assemble a Docker image. Docker uses this file to build the Docker image automatically by reading the instructions. An example Dockerfile is defined below.

```bash
FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

## System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y

# install requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the working directory to the compoent folder
WORKDIR /component/src

# Copy over src-files and spec of the component
COPY src/ .
COPY fondant_component.yaml ../

ENTRYPOINT ["python", "main.py"]
```

## Requirements.txt

A `requirements.txt` file lists the Python dependencies of the component. Note that any Fondant component will always have `Fondant` as the minimum requirement. It's important to also pin the version of each dependency to make sure the component remains working as expected. Below is an example of a component that relies on several Python libraries such as Pillow, PyTorch and Transformers.

```
git+https://github.com/ml6team/fondant.git@main
Pillow==9.4.0
torch==2.0.1
transformers==4.29.2
```