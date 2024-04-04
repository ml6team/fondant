# Creating lightweight components

Lightweight components are a great way to implement custom data processing steps in your dataset workflows. 
They are easy to implement and can be reused across different datasets. If you want to 
build more complex components that require additional dependencies (e.g. GPU support), you can
also build a containerized component. See the [containerized component guide](../components/containerized_components.md) for more info.

To implement a lightweight component, you simply need to create a python script that implements 
the component logic. Here is an example of a dataset composed of two custom components,
one that creates a dataset and one that adds a number to a column of the dataset:

```python title="dataset.py"
from fondant.component import DaskLoadComponent, PandasTransformComponent
from fondant.dataset import lightweight_component
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa

@lightweight_component(produces={"x": pa.int32(), "y": pa.int32()})
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

@lightweight_component(produces={"z": pa.int32()})
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

To register those components to a dataset, we can use the `create` and `apply` method for the 
first and second component respectively:

```python title="datast.py"
from fondant.dataset import Dataset

dataset = Dataset.create(
    ref=CreateData,
    dataset_name="dummy-pipeline",
)
_ = dataset.apply(
    ref=AddNumber,
    arguments={"n": 1},
)
```

Here we are creating a dataset workflow that reads data from the `CreateData` component and then applies
the `AddNumber` component to it. The `produces` argument is used to define the schema of the output
of the component. This is used to validate the output of the component and to define the schema
of the next component in the dataset.

Behind the scenes, Fondant will automatically package the component into a containerized component that
uses a base image with the current installed Fondant and python version.

## Installing additional requirements

If you want to install additional requirements for your component, you can do so by adding the 
package to the `extra_requires` argument of the `@lightweight_component` decorator. This will
install the package in the containerized component.

```python title="dataset.py"
@lightweight_component(extra_requires=["numpy"])
```

Under the hood, we are injecting the source to a docker container. If you want to use additional 
dependencies, you have to make sure to import the libaries inside a function directly.

For example: 
```python title="dataset.py"
...
def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    # Your code here
    ...
```

## Changing the base image

If you want to change the base image of the containerized component, you can do so by adding the
`base_image` argument to the `@lightweight_component` decorator. This will use the specified base
image instead of the default one. Make sure you install Fondant in the base image or list it 
in the `extra_requires` argument.

```python title="dataset.py"
@lightweight_component(base_image="python:3.10-slim")
```

## Optimizing loaded data
By default, Fondant will load all the data from the previous component into memory. You can 
restrict the columns that are loaded by specifying the columns to be loaded in the `consumes` argument
of the decorator. 
If we take the previous example, we can restrict the columns that are loaded by the `AddNumber` component
by specifying the `x` column in the `consumes` argument:

```python title="dataset.py"
@lightweight_component(
    consumes={
    "x": pa.int32()
    }
)
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
it to containerized component. See the [containerized component guide](../components/containerized_components.md) for more info.

## Loading dynamic fields

You can also choose to load in dynamic fields by setting the `additionalProperties` argument to `True` in the `consumes` argument.   

This will allow you to define an arbitrary number of columns to be loaded when applying your component to the dataset.  

This can be useful in scenarios when we want to dynamically load in fields from a dataset. For example, if we want to aggregate results 
from multiple columns, we can define a component that loads in specific column from the previous component and then aggregates them.   

Starting  from the previous example where we now have a dataset with a `x`, `y` and `z` column, we can define a component that aggregates
the `x` and `z` columns into a new column `score`:

```python
import dask.dataframe as dd
from fondant.component import PandasTransformComponent
from fondant.dataset import lightweight_component

@lightweight_component(
    consumes={
    "additionalProperties": True
    },
    produces={"score": pa.int32()},
)
class AggregateResults(PandasTransformComponent):
    def __init__(self):
        pass

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        metrics = list(self.consumes.keys())
        agg = dataframe[metrics].mean()
        agg_df = agg.to_frame(name="score")

        return agg_df

_ = dataset.apply(
    ref=AggregateResults,
    consumes={"x": pa.int32(), "z": pa.int32()},
)
```

This will aggregate the `x` and `z` columns into a new column `score`.

The main difference between the `consumes` argument in the `@lightweight_component` decorator and the `consumes` argument in the `apply` method is that the former is used to define the
schema of the component and the latter is used to specify the input data that will be passed to the component.  
  
Specifying the `consumes` argument in the `apply`allows for more flexibility in defining the input schema of the component 
compared to the `consumes` argument in the `@lightweight_component` decorator which is used to define the schema of the component.

Refer to this [section](../components/component_spec.md#dynamic-fields) for more info
on dynamic fields.