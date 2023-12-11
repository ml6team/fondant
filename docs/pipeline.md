# Pipeline

A Fondant pipeline is a tool for building complex workflows by creating a Directed Acyclic Graph 
(DAG) of different components that need to be executed. With Fondant, you can use both reusable
components and custom components, and chain them into a pipeline.

## Composing a pipeline

Start by creating a `pipeline.py` file and adding the following code.

```python
from fondant.pipeline import Pipeline

pipeline = Pipeline(
    name="my-pipeline",
    base_path="./data",
)
```

We identify our pipeline with a name and provide a base path where the pipeline will store its 
data and artifacts.

The base path can be:

* **A remote cloud location (S3, GCS, Azure Blob storage)**:  
  For the **local runner**, make sure that your local credentials or service account have read/write
  access to the designated base path and that you provide them to the pipeline.   
  For the **Vertex**, **Sagemaker**, and **Kubeflow** runners, make sure that the service account 
  attached to those runners has read/write access.
* **A local directory**: only valid for the local runner, points to a local directory. This is
  useful for local development.

!!! note "IMPORTANT"

    Make sure the provided base_path already exists.

??? "View a detailed reference of the options accepted by the `Pipeline` class"

    ::: fondant.pipeline.Pipeline.__init__
        handler: python
        options:
            show_source: false

### Adding a load component

You can read data into your pipeline by using the `Pipeline.read()` method with a load component.

```python
dataset = pipeline.read(
    "load_from_parquet",
    arguments={
        "dataset_uri": "path/to/dataset",
        "n_rows_to_load": 100,
    },
)
```

??? "View a detailed reference of the `Pipeline.read()` method"

    ::: fondant.pipeline.Pipeline.read
        handler: python
        options:
            show_source: false

The read method does not execute your component yet, but adds the component to the pipeline 
graph. It returns a lazy `Dataset` instance which you can use to chain transform components.

### Adding transform components

```python
from fondant.pipeline import Resources

dataset = dataset.apply(
    "embed_text",
    resources=Resources(
        accelerator_number=1,
        accelerator_name="GPU",
    )
)
```

The `apply` method also returns a lazy `Dataset` which you can use to chain additional components.

The `apply` method also provides additional configuration options on how to execute the component. 
You can for instance provide a `Resources` definition to specify the hardware it should run on. 
In this case, we want to leverage a GPU to run our embedding model. Depending on the runner, you 
can choose the type of GPU as well.

[//]: # (TODO: Add section on Resources or a general API section)

??? "View a detailed reference of the `Dataset.apply()` method"

    ::: fondant.pipeline.pipeline.Dataset.apply
        handler: python
        options:
            show_source: false

### Adding a write component

The final step is to write our data to its destination.

```python
dataset = dataset.write(
    "write_to_hf_hub",
    arguments={
        "username": "user",
        "dataset_name": "dataset",
        "hf_token": "xxx",
    }
)
```

??? "View a detailed reference of the `Dataset.write()` method"

    ::: fondant.pipeline.pipeline.Dataset.write
        handler: python
        options:
            show_source: false

!!! note "IMPORTANT"  

    Currently Fondant supports linear DAGs with single dependencies. Support for non-linear DAGs 
    will be available in future releases.

[//]: # (TODO: Add info on mapping fields between components)

## Running a pipeline

Once all your components are added to your pipeline you can use different runners to run your
pipeline.

!!! note "IMPORTANT"  

    When using other runners you will need to make sure that your new environment has access to:

    - The base path of your pipeline (as mentioned above)
    - The images used in your pipeline (make sure you have access to the registries where the images are
    stored)

=== "Console"

    === "Local"
    
        ```bash
        fondant run local <pipeline_ref>
        ```
    === "Vertex"
    
        ```bash 
        fondant run vertex <pipeline_ref> \
         --project-id $PROJECT_ID \
         --project-region $PROJECT_REGION \
         --service-account $SERVICE_ACCOUNT
        ```
    === "SageMaker"
    
        ```bash
        fondant run sagemaker <pipeline_ref> \
         --role-arn <sagemaker_role_arn> 
        ```
    === "Kubeflow"
    
        ```bash
        fondant run kubeflow <pipeline_ref>
        ```

=== "Python"

    === "Local"
    
        ```python
        from fondant.pipeline.runner import DockerRunner

        runner = DockerRunner()
        runner.run(input=<pipeline_ref>)
        ```
    === "Vertex"
    
        ```python
        from fondant.pipeline.runner import VertexRunner

        runner = VertexRunner()
        runner.run(input=<pipeline_ref>)
        ```
    === "SageMaker"
    
        ```python
        from fondant.pipeline.runner import SageMakerRunner

        runner = SageMakerRunner()
        runner.run(input=<pipeline_ref>, pipeline_name=<pipeline-name> role_arn=<sagemaker_role_arn>)        
        ```
    === "KubeFlow"
    
        ```python
        from fondant.pipeline.runner import KubeFlowRunner

        runner = KubeFlowRunner(host=<kubeflow_host>)
        runner.run(input=<pipeline_ref>)        
        ```
    
  The pipeline ref can be a reference to the file containing your pipeline, a variable 
  containing your pipeline, or a factory function that will create your pipeline.
