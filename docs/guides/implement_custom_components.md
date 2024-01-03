# Implementing custom components

This guide will teach you how to build custom components and integrate them in your pipeline.

## Overview

In the [previous tutorial](/build_a_simple_pipeline.md), you learned how to create your first Fondant pipeline. While the 
example demonstrates how to build a pipeline from reusable components, this is only the beginning.

In this tutorial, we will guide you through the process of implementing your very own custom 
component. We will illustrate this by building a transform component that filters images based on 
file type.

This pipeline is an extension of the one introduced in the previous tutorial. After loading the 
dataset from HuggingFace, it filters out any non-PNG files before downloading them. Finally, we 
write the images to a local directory.

## Setting up the environment

We will be using the [local runner](../runners/local.md) to run this pipelines. To set up your local environment,
please refer to our [installation](installation.md) documentation.

## 1. Building a custom transform component

The typical file structure of a custom component looks like this:

```
|- custom_component
   |- src
   |  |- main.py
   |- Dockerfile
   |- fondant_component.yaml
   |- requirements.txt
```

It contains:

- **`src/main.py`**: The actual Python code to run.
- **`Dockerfile`**: The Dockerfile to package your component.
- **`fondant_component.yaml`**: The component specification defining the contract for the component.
- **`requirements.txt`**: Containing the Python requirements of your component.

Schematically, it can be represented as follows:

![component architecture](https://github.com/ml6team/fondant/blob/main/docs/art/guides/component.png?raw=true)

You can find a more detailed explanation [here](../components/custom_component.md).

### Creating the ComponentSpec

We start by creating the contract of our component:

```yaml title="fondant_component.yaml"
name: Filter file type
description: Component that filters on mime types
image: <my-registry>/filter_image_type:<version>

consumes:
  image_url:
    type: string

args:
  mime_type:
    description: The mime type to filter on
    type: str
```

It begins by specifying the component name, a brief description, and component's Docker image.

!!! note "IMPORTANT"

    Note that you'll need your own container registry to host the image for you custom component

The `consumes` section describes which data the component will consume. In this case, it will 
read a single `"image_url"` column.

[//]: # (TODO: Use a transform instead of filter component here to keep it simple)

Since the component only filters the data, it will not create any new data. Fondant handles your 
data efficiently by keeping track of the index along your pipeline. Only this index will be 
updated when filtering data, which means that we don't need to define a `produces` section in the 
component specification.

Finally, we define the arguments that the component will support. In this case, we only add a 
single `mime_type` argument, which allows us to define which mime type should be filtered.

### Implementing the component

Now, it's time to implement the component logic. To do this, we'll create a `src/main.py` file.

We will subclass the `PandasTransformComponent` offered by Fondant. This is the most basic type 
of component. The following two methods should be implemented:

- **`__init__()`**: This method will receive the arguments define in your component 
  specification. Fondant also inserts some additional keyword arguments for more advanced use 
  cases. Be sure to include a `**kwargs` argument if you're not using those.
- **`transform()`**: This method receives a chunk of the input data as a Pandas `DataFrame`. 
  Fondant automatically chunks your data so you can process larger-than-memory data, and your 
  component is executed in parallel across the available cores.

```python title="src/main.py"
"""A component that filters images based on file type."""
import mimetypes

import pandas as pd
from fondant.component import PandasTransformComponent


class FileTypeFilter(PandasTransformComponent):

    def __init__(self, *, mime_type: str, **kwargs):
        """Custom component to filter on specific file type based on url
        
        Args:
            mime_type: The mime type to filter on (also defined in the component spec)
        """
        self.mime_type = mime_type

    @staticmethod
    def get_mime_type(url):
        """Guess mime type based on the file name"""
        mime_type, _ = mimetypes.guess_type(url)
        return mime_type

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Reduce dataframe to specific mime type"""
        dataframe["mime_type"] = dataframe["url"].apply(self.get_mime_type)
        return dataframe[dataframe["mime_type"] == self.mime_type]
```

We return the filtered dataframe from the `transform` method, which Fondant will use to 
automatically update the index. If we would have specified any output fields in our component 
contract, Fondant would extract and write those as well.

### Defining the requirements

Our component uses two third-party dependencies: `pandas`, and `fondant`. `pandas` comes bundled 
with `fondant` if you install it using the `component` extra though, so our `requirements.txt` will 
look as follows:

```text title="requirements.txt"
fondant[component]
```

### Building the component

To use the component, it should be packaged into a Docker image, for which we need to define a 
Dockerfile.

```bash title="Dockerfile"
FROM --platform=linux/amd64 python:3.8-slim

# Install requirements
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the working directory to the component folder
WORKDIR /component/src

# Copy over src-files
COPY src/ .

ENTRYPOINT ["fondant", "execute", "main"]
```

The entrypoint should be the `fondant execute` command which will execute your component.

### Using the component

We will now update the pipeline we created in the [previous guide](/build_a_simple_pipeline.md) 
to leverage our component.

Our complete file structure looks as follows:
```
|- components
|  |- filter_image_type
|     |- src
|     |  |- main.py
|     |- Dockerfile
|     |- fondant_component.yaml
|     |- requirements.txt
|- pipeline.py
```

```python title="pipeline.py"
from fondant.pipeline import Pipeline
import pyarrow as pa

pipeline = Pipeline(
    name="creative_commons_pipline",
    base_path="./data"
)

dataset = pipeline.read(
    "load_from_hf_hub",
    arguments={
        "dataset_name": "fondant-ai/fondant-cc-25m",
        "n_rows_to_load": 100,
    },
    produces={
        "alt_text": pa.string(),
        "image_url": pa.string(),
        "license_location": pa.string(),
        "license_type": pa.string(),
        "webpage_url": pa.string(),
    },
)

# Our custom component
urls = dataset.apply(
    "components/filter_image_type",
    arguments={
        "mime_type": "image/png"
    }
)

images = urls.apply(
    "download_images",
)

english_images = images.apply(
    "filter_language",
    arguments={
        "language": "en"
    },
    consumes={
        "text": "alt_text"
    }
)
```

Instead of providing the name of the component like we did with the reusable components, we now 
provide the path to our custom component.

Now, you can execute the pipeline once more and examine the results. The final output should 
exclusively consist of PNG images.

We have designed the custom component to be easily adaptable. For example, if you wish to filter 
out JPEG files, you can simply change the argument to `image/jpeg`, and your dataset will be 
populated with JPEGs instead of PNGs

## Next steps

We now have a pipeline that downloads a dataset from the HuggingFace hub, filters the urls by 
image type, downloads the images, and filters them by alt text language.

One final step still remaining, is to write teh final dataset to its destination. You could for 
instance use the [`write_to_hf_hub`](../components/hub.md#write_to_hugging_face_hub#description) component to write it to 
the HuggingFace Hub, or create a custom `WriteComponent`.
