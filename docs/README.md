# Docs

This file contains general documentation on Express and KubeFlow on GCS.

## Express: built around manifests

Express provides helper functions and boilerplate to speed up the creation of KubeFlow pipelines and components.

### Helpers

The `express.helpers` module contains a few helper functions:

- `io_helpers.py`: general I/O helper functions.
- `kfp_helpers.py`: include helper functions for GPU logging when running a KFP component and parsing specific KFP input.
- `logger.py`: General logger module for event logging.
- `manifest_helpers.py`: Defines the structure of the data manifest that holds the location and contents of the different data sources.
- `parquet_helpers.py`: General helper functions for creating and updating the different parquet files that contain the index and data sources information (metadata, captions, ...) as well as various utilities (duplication removal, metadata and column retrieval, ...).
- `storage_helpers.py`: helper functions to interact with Google Cloud Storage (blob listing, I/O, ...)

Those helper functions can be used when creating components.

### Component base classes

The `express.components` module contains component base classes which you can overwrite to create components. Each of these component classes always have 2 variants; a loader and a transform version.

- Pandas: `PandasTransformComponent`, `PandasLoaderComponent`

## Using KubeFlow on GCP

### Connecting the Kubeflow Pipelines UI

There are two ways to connect to KFP UI, first make sure you autheticate to the GKE cluster
hosting KFP

```
gcloud container clusters get-credentials [CLUSTER_NAME} --zone=[ZONE]
   ```

1) Port-forwarding to access the kubernetes service

    ```
    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
    ```

   Open the Kubeflow Pipelines UI at <http://localhost:8080>


2) Using the IAP URL of KFP

    ```
    kubectl describe configmap inverse-proxy-config -n kubeflow | grep googleusercontent.com
    ```

### Pipelines access issues
In case you have any authentication issues when compile the pipeline or accessing kubeflow: 

* Make sure that you're authenticated with the right credentials and to the right project by running
`gcloud init` in your local terminal.  

* If you still run into issues, you can try running `gcloud auth application-default login` and
select the right account in the pop up page, this will make sure that your local application
temporarily uses your own user credentials for API access.  

* Finally, if the issues still persists. Make sure that the IP address you're trying to connect to 
is listed within the [authorized networks](https://github.com/creativefabrica/express-infrastructure/blob/master/terraform/modules/kfp/variables.tf)
of the GKE cluster where Kubeflow is deployed. 


## Component

Pipelines exist out of several components. These are stored in `mlpipelines/components`. Each component requires
the setup of a Dockerfile 

There are multiple steps for setting up the component:  

- Write the main script of your component within the `src` directory inside the component.  

- Define all the input and output parameters of your component in the `component.yaml` file  

- Specify the requirement packages of the component in the `requirements.txt` file and specify the build 
steps in the `Dockerfile`.  

- Finally, once you're done, you're ready to build the images and push it to the [Artifact Registry](https://cloud.google.com/artifact-registry). You can do so by running the script (`sh build_image.sh`). Make sure to enable [kaniko](https://cloud.google.com/build/docs/optimize-builds/kaniko-cache) for faster container build time.  

More information about the structure of a component can be found in this [section](example_components/example_component).

## Pipelines

Finally, once you have built all the separate components, you are ready to compile and upload the 
pipeline. The `upload.py` script contains functions to help with the compilation and upload of new pipelines.

[Here](pipelines/example_pipeline.py) is a simple example demonstrate a toy pipeline, you can see how parameters are passed between different steps of the pipeline.
To compile the pipeline, simply execute the file, the pipeline will be uploaded to kubeflow under a specific version
where it can be visualized and run. 

## Example of a typical pre-process, train and deploy pipeline

A typical workflow to automate with Kubeflow Pipelines is training and 
deployment of a Machine Learning model. 
Please check out the [example](https://github.com/kubeflow/examples/tree/master/financial_time_series#kubeflow-pipelines) 
that showcases this automated workflow with conditional deployment (based on 
accuracy threshold). The pipeline code is stored [here](https://github.com/kubeflow/examples/blob/master/financial_time_series/tensorflow_model/ml_pipeline.py).


## Iterate and develop quickly with Lightweight components

One of the primary goals of the lightweight components is to enable to iterate 
and develop quickly on a Kubeflow Pipeline, from a notebook environment. 
Next to that, the lightweight components can be useful for small steps in a pipeline 
that typically only require a few lines of Python code. Instead of having to 
containerize this small Python function, you can simply 'mount' it on top of 
an existing image.

To get some experience with these lightweight components, 
you can checkout 
some [sample notebooks](https://github.com/Svendegroote91/kfp_samples) 
that highlight their main functionality.

# Express Components - Common

Manifest handling, dataset loading and writing are moderate-complexity recurring patterns across different express components.

To make implementing express components as lightweight as possible, Express provides a python package and base docker image that takes care of this heavy lifting, and makes it easy to implement many data transformations out of the box.


## 1. Concepts

### 1.a) DataManifests, ExpressDatasets and ExpressDatasetDrafts
A **DataManifest** is a json file that describes the location and contents of different data sources. It can be seen as the recipe for a dataset.

An **ExpressDataset** is a wrapper object around the manifest that implements the data access logic, and exposes methods to read data from specific data sources.

After transforming the input data (see below), an **ExpressDatasetDraft** creates a plan for an output dataset / manifest, by specifying which data sources to retain from the input, which to replace with locally created data, and which new data sources to create. At the end of a component run, local data will be uploaded and an output manifest will be created.

![Figure 1. Relation between different dataset concepts](data-flow.png)

### 1.b) Transforms and Loaders
The most common type of component in Express is an **ExpressTransformComponent**, which takes an ExpressDataset and an optional dict of arguments as input and returns an ExpressDatasetDraft of transformed output data.

However, at the start of a new pipeline, you won't yet have any express datasets to transform. Instead, an express pipeline can use an **ExpressLoaderComponent** as entry-point, which only takes the optional dict of arguments to construct an ExpressDatasetDraft. For example, the arguments could specify an external data location and how to interpret it, after which a loader job can create a first ExpressDataset.


## 2. Usage

To implement your own Transform component, you'll need to take a dependency on the `express_components` package and subclass one of the TransformComponent base classes. 

### 2.a) General flow
The below example uses the Pandas base classes.

#### I. Subclass one of the TransformComponent/LoaderComponent base classes and implement the transform method.

```python
from express_components.pandas_components import PandasTransformComponent, PandasDataset, PandasDatasetDraft

class MyFirstTransform(PandasTransformComponent):
    @classmethod
    def transform(cls, data: PandasDataset, extra_args: Optional[Dict] = None) -> PandasDatasetDraft:
        
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
1. Non-distributed Pandas implementation.\
    - `express_components.pandas_components.{PandasTransformComponent, PandasLoaderComponent}`
    - Data is exposed as a Pandas `DataFrame`. Depending on the use-case, consumers can do batch transforms, or collect data in-memory to a Pandas `DataFrame`.

**Planned implementations**
1. Spark-based components and base image.


## 3. Adding additional Transform/Loader base implementations
If you want a different data manipulation runtime, use different data structures, or do other component-level bootstrapping across multiple jobs, you could add another base implementation for the `ExpressTransformComponent` / `ExpressLoaderComponent`.

A general overview of the different implementation levels can be seen in Figure 2. Additional base implementations work on the middle layer, and mainly affect data loading / writing logic.

![Figure 2. Express component class hierarchy](class-hierarchy.png)

More specifically, you'll need to subclass the ExpressDatasetHandler mix-in and implement the abstract dataset reading / writing methods.

Look at `express_components/pyarrow_components.py` for an example implementation.