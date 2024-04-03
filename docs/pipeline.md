# Dataset

Fondant helps you build datasets by providing a set of operations to load, transform, 
and write data. With Fondant, you can use both reusable components and custom components, 
and chain them to create datasets.

## Build a dataset

Start by creating a `dataset.py` file and adding the following code.

```python
from fondant.dataset import Dataset

dataset = Dataset.create(
    "load_from_parquet",
    arguments={
        "dataset_uri": "path/to/dataset",
        "n_rows_to_load": 100,
    },
    produces={
        "text": pa.string()
    },
    dataset_name="my_dataset"
)
```

This code initializes a `Dataset` instance with a load component. The load component reads data.

??? "View a detailed reference of the `Dataset.create()` method"

    ::: fondant.dataset.Dataset.read
        handler: python
        options:
            show_source: false

The create method does not execute your component yet, but adds the component to the execution 
graph. It returns a lazy `Dataset` instance which you can use to chain transform components.

### Adding transform components

```python
from fondant.dataset import Resources

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

    ::: fondant.dataset.dataset.Dataset.apply
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

    ::: fondant.dataset.dataset.Dataset.write
        handler: python
        options:
            show_source: false

[//]: # (TODO: Add info on mapping fields between components)

## Materialize the dataset

Once all your components are added to your dataset you can use different runners to materialize the 
your dataset.

!!! note "IMPORTANT"  

    When using other runners you will need to make sure that your new environment has access to:

    - The base path of your pipeline (as mentioned above)
    - The images used in your pipeline (make sure you have access to the registries where the images are
    stored)

=== "Console"

    === "Local"
    
        ```bash
        fondant run local <dataset_ref> --working_directory <path_to_working_directory>
        ```
    === "Vertex"
    
        ```bash 
        fondant run vertex <dataset_ref> \
         --project-id $PROJECT_ID \
         --project-region $PROJECT_REGION \
         --service-account $SERVICE_ACCOUNT \
         --working_directory <path_to_working_directory>
        ```
    === "SageMaker"
    
        ```bash
        fondant run sagemaker <dataset_ref> \
         --role-arn <sagemaker_role_arn> \
         --working_directory <path_to_working_directory>
        ```
    === "Kubeflow"
    
        ```bash
        fondant run kubeflow <dataset_ref> --working_directory <path_to_working_directory>
        ```

=== "Python"

    === "Local"
    
        ```python
        from fondant.dataset.runner import DockerRunner

        runner = DockerRunner()
        runner.run(input=<dataset_ref>, working_directory=<path_to_working_directory>)
        ```
    === "Vertex"
    
        ```python
        from fondant.dataset.runner import VertexRunner

        runner = VertexRunner()
        runner.run(input=<dataset_ref>, working_directory=<path_to_working_directory>)
        ```
    === "SageMaker"
    
        ```python
        from fondant.dataset.runner import SageMakerRunner

        runner = SageMakerRunner()
        runner.run(input=<dataset_ref>,role_arn=<sagemaker_role_arn>, 
                  working_directory=<path_to_working_directory>)        
        ```
    === "KubeFlow"
    
        ```python
        from fondant.dataset.runner import KubeFlowRunner

        runner = KubeFlowRunner(host=<kubeflow_host>)
        runner.run(input=<dataset_ref>)        
        ```
    
  The dataset ref can be a reference to the file containing your dataset, a variable 
  containing your dataset, or a factory function that will create your dataset.

  The working directory can be:
  - **A remote cloud location (S3, GCS, Azure Blob storage):**
    For the local runner, make sure that your local credentials or service account have read/write 
    access to the designated working directory and that you provide them to the dataset.
    For the Vertex, Sagemaker, and Kubeflow runners, make sure that the service account 
    attached to those runners has read/write access.
  - **A local directory:** only valid for the local runner, points to a local directory. 
    This is useful for local development.
