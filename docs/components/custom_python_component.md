# Creating custom python components

Python components are a great way to implement custom data processing steps in your pipeline. 
They are easy to implement and can be reused across different pipelines. If you want to 
build more complex components that require additional dependencies (e.g. GPU support), you can
also build a containerized component. See the [containerized component guide](../components/custom_containerized_component.md) for more info.

To implement a custom python component, you simply need to create a python script that implements 
the component logic. Here is an example of a pipeline composed of two custom components,
one that creates a dataset and one that adds a number to a column of the dataset:

```python title="pipeline.py"
from fondant.component import DaskLoadComponent, PandasTransformComponent
from fondant.pipeline import lightweight_component
import dask.dataframe as dd
import pandas as pd

@lightweight_component
class CreateData(DaskLoadComponent):
    def load(self) -> dd.DataFrame:
        df = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [4, 5, 6],
            },
            index=pd.Index(["a", "b", "c"], name="id"),
        )
        return dd.from_pandas(df, npartitions=1)

@lightweight_component
class AddNumber(PandasTransformComponent):
    def __init__(self, n: int):
        self.n = n

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["z"] = dataframe["x"].map(lambda x: x + self.n)
        return dataframe
```

Notice that we use the `@lightweight_component` decorator to define our components. This decorator
is used to package the component into a containerized component and can also be used to 
define additional functionalities.

To register those components to a pipeline, we can use the `read` and `apply` method for the 
first and second component respectively:

```python title="pipeline.py"
import pyarrow as pa
from fondant.pipeline import Pipeline

pipeline = Pipeline(
    name="dummy-pipeline",
    base_path="./data",
)

dataset = pipeline.read(
    ref=CreateData,
    produces={"x": pa.int32(), "y": pa.int32()},
)

_ = dataset.apply(
    ref=AddNumber,
    produces={"z": pa.int32()},
    arguments={"n": 1},
)
```

Here we are creating a pipeline that reads data from the `CreateData` component and then applies
the `AddNumber` component to it. The `produces` argument is used to define the schema of the output
of the component. This is used to validate the output of the component and to define the schema
of the next component in the pipeline.

Behind the scenes, Fondant will automatically package the component into a containerized component that
uses a base image with the current installed Fondant and python version.

## Installing additional requirements

If you want to install additional requirements for your component, you can do so by adding the 
package to the `extra_requires` argument of the `@lightweight_component` decorator. This will
install the package in the containerized component.

```python title="pipeline.py"
@lightweight_component(extra_requires=["numpy"])
```

## Changing the base image

If you want to change the base image of the containerized component, you can do so by adding the
`base_image` argument to the `@lightweight_component` decorator. This will use the specified base
image instead of the default one. Make sure you install Fondant in the base image or list it 
in the `extra_requires` argument.

```python title="pipeline.py"
@lightweight_component(base_image="python:3.8-slim")
```

## Optimizing loaded data
By default, Fondant will load all the data from the previous component into memory. You can 
restrict the columns that are loaded by specifying the columns to be loaded in the `consumes` argument
of the decorator. 
If we take the previous example, we can restrict the columns that are loaded by the `AddNumber` component
by specifying the `x` column in the `consumes` argument:

```python title="pipeline.py"
@lightweight_component(consumes={"x"})
class AddNumber(PandasTransformComponent):
    def __init__(self, n: int):
        self.n = n

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["z"] = dataframe["x"].map(lambda x: x + self.n)
        return dataframe
```

This will omit the `y` column from the loaded data, which can be useful if you are working with large
datasets and want to avoid loading unnecessary data.

If you want to publish your component to the Fondant Hub, you will need to convert 
it to containerized component. See the [containerized component guide](../components/custom_containerized_component.md) for more info.

**Note:** Python based components also support defining dynamic fields by default. See the [dynamic fields guide](../components/component_spec.md#dynamic-fields) for more info
on dynamic fields.
