# Express Components - Common

Manifest handling, dataset loading and writing are moderate-complexity recurring patterns across different express components.

To make implementing express components as lightweight as possible, Express provides a python package and base docker image that takes care of this heavy lifting, and makes it easy to implement many data transformations out of the box.


## 1. Concepts

### 1.a) DataManifests, ExpressDatasets and ExpressDatasetDrafts
A **DataManifest** is a json file that describes the location and contents of different data sources. It can be seen as the recipe for a dataset.

An **ExpressDataset** is a wrapper object around the manifest that implements the data access logic, and exposes methods to read data from specific data sources.

After transforming the input data (see below), an **ExpressDatasetDraft** creates a plan for an output dataset / manifest, by specifying which data sources to retain from the input, which to replace with locally created data, and which new data sources to create. At the end of a component run, local data will be uploaded and an output manifest will be created.

![Figure 1. Relation between different dataset concepts](diagrams/data-flow.png)

### 1.b) Transforms and Loaders
The most common type of component in Express is an **ExpressTransformComponent**, which takes an ExpressDataset and an optional dict of arguments as input and returns an ExpressDatasetDraft of transformed output data.

However, at the start of a new pipeline, you won't yet have any express datasets to transform. Instead, an express pipeline can use an **ExpressLoaderComponent** as entry-point, which only takes the optional dict of arguments to construct an ExpressDatasetDraft. For example, the arguments could specify an external data location and how to interpret it, after which a loader job can create a first ExpressDataset.


## 2. Usage

To implement your own Transform component, you'll need to take a dependency on the `express_components` package and subclass one of the TransformComponent base classes. 

### 2.a) General flow
The below example uses the PyArrow base classes.

#### I. Subclass one of the TransformComponent/LoaderComponent base classes and implement the transform method.

```python
from express_components.pandas_components import PandasTransformComponent, PandasDataset,
    PandasDatasetDraft


class MyFirstTransform(PandasTransformComponent):
    @classmethod
    def transform(cls, data: PandasDataset,
                  extra_args: Optional[Dict[str, Union[str, int, float, bool]]] = None) -> PandasDatasetDraft:
        # Reading data
        index: pd.Series = data.load_index()
        my_data: pd.DataFrame = data.load("my_data_source")
        
        # filter index 
        index = index.filter(items:<list of ids to filter>)
        # Transforming data
        my_data = my_data.apply(<transformation_function>)
       
         return data.extend() \
            .with_index(index) \
            .with_data_source("my_transformed_data_source", my_data)

```

#### II. Implement Docker entrypoint

```
if __name__ == '__main__':
    MyFirstTransform.run()
```

### 2.b) Taking a dependency on express_components
There are two ways to add `express_components` to your dependencies.

1. (Recommended) Build the `common` docker image, and have your component use this as a base image. This base image will include the `express_components` python package, and itself extends a PyTorch GPU image.
2. `express_components` can be installed as a standalone Python package into your custom images. See the Dockerfile of the `common` base image for an example implementation.

### 2.c) Pick an ExpressTransformerComponent / ExpressLoaderComponent base implementation to subclass

Different implementation mainly differ in how they expose the data, and what data manipulation capabilities are exposed at runtime.

**Available implementations**
1. Non-distributed pyarrow implementation.\
    - `express_components.pyarrow_components.{PyArrowTransformComponent, PyArrowLoaderComponent}`
    - Data is exposed as a `pyarrow.dataset.Scanner`. Depending on the use-case, consumers can do batch transforms, or collect data in-memory to a pyarrow `Table` or pandas `DataFrame`.

**Planned implementations**
1. Spark-based components and base image.


## 3. Adding additional Transform/Loader base implementations
If you want a different data manipulation runtime, use different data structures, or do other component-level bootstrapping across multiple jobs, you could add another base implementation for the `ExpressTransformComponent` / `ExpressLoaderComponent`.

A general overview of the different implementation levels can be seen in Figure 2. Additional base implementations work on the middle layer, and mainly affect data loading / writing logic.

![Figure 2. Express component class hierarchy](diagrams/class-hierarchy.png)

More specifically, you'll need to subclass the ExpressDatasetHandler mix-in and implement the abstract dataset reading / writing methods.

Look at `express_components/pyarrow_components.py` for an example implementation.