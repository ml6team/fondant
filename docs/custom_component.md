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
specification page](component_spec) for info on how to write the specification for your component.

## Main.py script

The core logic of the component should be implemented in a `main.py` script in a folder called 
`src`. 
The logic should be implemented as a class, inheriting from one of the base `Component` classes 
offered by Fondant.
There are three large types of components:
- **`LoadComponent`**: Load data into your pipeline
- **`TransformComponent`**: Implement a single transformation step in your pipeline
- **`WriteComponent`**: Write the results of your pipeline

The easiest way to implement a `TransformComponent` is to subclass the provided 
`PandasTransformComponent`. This component streams your data and offers it in memory-sized 
chunks as pandas dataframes.

```python
from fondant.component import PandasTransformComponent

class ExampleComponent(PandasTransformComponent):
    
    def setup(self, argument1, argument2):
        """This method is called once per component with any custom arguments it received. Use 
        it for component wide setup or to store your arguments as instance attributes to access 
        them in the `transform` method.
        
        Args:
            argumentX: A custom argument passed to the component
        """ 

    def transform(self, dataframe):
        """Implement your custom transformation logic in this single method
        
        Args:
            dataframe: A Pandas dataframe containing one partition of your data
            
        Returns:
            A pandas dataframe with the transformed data
        """
```

The `setup` method is called once for each component class with custom arguments defined in the 
`args` section of the [component specification](component_spec).)

The `transform` method is called multiple times, each time containing a pandas `dataframe` 
loaded in memory.

The `dataframes` passed to the `transform` method contains the data specified in the `produces` 
section of the component specification, with column names formatted as `{subset}_{field}`. So if 
a component defines that it consumes an `images` subset with a `data` field, the dataframe will 
contain a column called `images_data`.

The `transform` method should return a single dataframe, with the columns complying to the 
`{subset}_{field}` format matching the `produces` section of the component specification.

Note that the `main.py` script can be split up into several Python scripts in case it would become 
prohibitively long. See the 
[prompt based LAION retrieval component](https://github.com/ml6team/fondant/tree/main/components/prompt_based_laion_retrieval/src) 
as an example: the CLIP client itself is defined in a separate script called `clip_client`, 
which is then imported in the `main.py` script.

## Dockerfile

The `Dockerfile` defines how to build the component into a Docker image. An example Dockerfile is defined below.

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