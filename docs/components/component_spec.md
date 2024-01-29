# Component specification

Each Fondant component is defined by a component specification which describes its interface.
The component specification is used for a couple of things:

- To define which input data Fondant should provide to the component, and which output data it should
  write to storage.
- To validate compatibility with other components.
- To execute the component with the correct parameters.

The component specification should be defined by the author of the component.

## Contents

A component specification consists of the following sections:

```yaml
name:
  ...
description:
  ...
image:
  ...

consumes:
  ...

produces: 
  ...

args: 
  ...
```

### Metadata

The metadata tracks metadata about the component, such as its name, description, and the URL of the [Docker](https://www.docker.com/) image used to run it.

```yaml
name: Example component
description: This is an example component
image: example_component:latest
```

### Consumes & produces

The `consumes` and `produces` sections describe which data the component consumes and produces.
The specification below for instance defines a component that creates an embedding from an
image-caption combination.

```yaml
...
consumes:
  images:
    type: binary
  text:
    type: utf8

produces:
  embeddings:
    type: array
    items:
      type: float32
```

The `consumes` and `produces` sections follow the schema below:

```yaml
consumes/produces:
  <field>:
    type: <type>
  additionalProperties: false
```


#### Fields

Each component specification defines a list of `fields` where the fields are the columns of the
dataset. 

- Only those fields defined in the `consumes` section of the component specification are read
  and passed to the component implementation.
- Only those fields defined in the `produces` section of the component specification are written
  to storage

Each field defines the expected data type, which should match the
[types defined by Fondant](https://github.com/ml6team/fondant/blob/main/src/fondant/core/schema.py),
that correespond to [Arrow data types](https://arrow.apache.org/docs/python/api/datatypes.html).

Note that you can always map a field from your dataset with a different name to a specific field name expected by the
component provided they have the same data type. For example, suppose we have a component spec that
consumes a `text` field:

```yaml
consumes:
   text:
      type: string
```           

If your dataset has a field called `custom_text` with type `string`, you can map it to the
`text` field in the component spec as follows:

```python 

dataset = pipeline.read(...)
dataset = dataset.apply(
    "example_component",
    consumes={
        "text": "custom_text"
    }
```

In this example, the `custom_text` field will be mapped to the `text` field to match the 
field expected by the component.

Similarly, you can also the map the output field of a component to a specific field name in the
dataset. Suppose we have a component spec that produces a `width` field:

```yaml
produces:
    width:
        type: int
```

If you want to map the output field to a field called `custom_width` in the dataset, you can do
so as follows:

```python 

dataset = pipeline.read(...)
dataset = dataset.apply(
    "example_component",
    produces={
        "width": "custom_width"
    }
```

In this example, the component produces a field called `width`. This field name is mapped to a custom field 
name `custom_width` which can be referenced in later components or used to change the field name of the final
written dataset. 


#### Dynamic fields

The schema also defines the `additionalProperties` keyword. This can be
used to define dynamic fields that should be produced or consumed when set to `true`. This can be useful in many scenarios,
here are a few examples:

- Components that load/write general fields from/to external source (e.g. a CSV file, HuggingFace dataset, ...)
  can use this to define dynamic fields that should be loaded/written.
- Components that consume or produce optional fields. For example, a
  component that queries a vector database can accept either a text passage or optionally precalculated text embeddings.
- Components that can work on a dynamic amount of fields.

Let's take an example of a component that loads a dataset from a CSV file. The CSV file can contain any number of
columns, so we set `additionalProperties` to `true` to allow any column to be loaded.

```yaml
produces:
    additionalProperties: true
```

Note that the schema of the fields to be produced is not defined as it would usually be 
in the component specification, so we will need to specify the schema of the
fields when defining the components

```python
dataset = pipeline.read(
    "load_from_csv",
    arguments={
        "dataset_uri": "path/to/dataset.csv",
    },
    produces={
        "image": pa.binary(),
        "embedding": pa.list_(pa.binary())
    }
)
```

Here we define the schema of the `image` and `embedding` fields which will be produced by the component. 

Now that we know how to define dynamic fields to be produced, let's take a look at how we can use the `additionalProperties`
to define additional field to be consumed. Building on the previous example, let's take a component that takes
either an `image` or `embedding` field as input to query a certain vector database. The specification 
for such a component can be defined as follows:

```yaml
consumes:
    additionalProperties: true
produces:
    retrieved_images:
        type: binary
```
We can now use the `additionalProperties` to allow the component to accept dynamic fields. This gives us the flexibly choose which field to consume
by the next component. We can either load the `image` field:

```python 

dataset = pipeline.read(
    "load_from_csv",
    arguments={
        "dataset_uri": "path/to/dataset.csv",
    },
    produces={
        "my_image": pa.binary(),
        "my_embedding": pa.list_(pa.binary())
    }
)

dataset = dataset.apply(
    "query_vector_database",
    consumes={
        "image": "my_image"
    }
)
```

or the `embedding` field:

```python 

dataset = pipeline.read(
    "load_from_csv",
    arguments={
        "dataset_uri": "path/to/dataset.csv",
    },
    produces={
        "my_image": pa.binary(),
        "my_embedding": pa.list_(pa.binary())
    }
)

dataset = dataset.apply(
    "query_vector_database",
    consumes={
        "embedding": "my_embedding"
    }
)
```

Where `my_image` and `my_embedding` are the fields produced by the previous component and `image`, `embedding` are the field names
that can be consumed by the `query_vector_database` component. The data type of the consumed field does not need to be specified here
since it can be inferred from the previous component.

Note that in the implementation of the component, there 
should be a custom logic to handle the different cases of the consumed fields based on the passed field name.

For a practical example on using dynamic fields, make sure to check the guide on implementing [your own custom component](../guides/implement_custom_components.md) below to build a better understanding.

### Args

The `args` section describes which arguments the component takes. Each argument is defined by a
`description` and a `type`, which should be one of the builtin Python types. Additionally, you can
set an optional `default` value for each argument.

```yaml
args:
  custom_argument:
    description: A custom argument
    type: str
  default_argument:
    description: A default argument
    type: str
    default: bar
```

These arguments are passed in when the component is instantiated.
If an argument is not explicitly provided, the default value will be used instead if available.

```python
dataset = pipeline.read(
    "custom_component",
    arguments={
        "custom_argument": "foo"
    },
)
```

Afterwards, we pass all keyword arguments to the `__init__()` method of the component.

```python
import pandas as pd
from fondant.component import PandasTransformComponent


class ExampleComponent(PandasTransformComponent):

  def __init__(self, *, custom_argument, default_argument) -> None:
    """
    Args:
        x_argument: An argument passed to the component
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

Afterwards, we pass all keyword arguments to the `__init__()` method of the component.


You can also use the a `teardown()` method to perform any cleanup after the component has been executed.
This is a good place to close any open connections or files.

```python
import pandas as pd
from fondant.component import PandasTransformComponent
from my_library import Client

  def __init__(self, *, client_url) -> None:
    """
    Args:
        x_argument: An argument passed to the component
    """
    # Initialize your component here based on the arguments
    self.client = Client(client_url)
    
  def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Implement your custom logic in this single method

    Args:
        dataframe: A Pandas dataframe containing the data

    Returns:
        A pandas dataframe containing the transformed data
    """
    
  def teardown(self):
    """Perform any cleanup after the component has been executed
    """
    self.client.shutdown()
```