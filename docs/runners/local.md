### Local Runner

Leverages [docker compose](https://docs.docker.com/compose/). The local runner is mainly aimed
at helping you develop fondant datasets and components faster since it allows you to develop on
your local machine or a Virtual Machine. This enables you to quickly iterate on development. Once
you have a dataset developed, Switching to either the [Vertex](vertex.md) or [Kubeflow](kfp.md) runners 
offers many advantages such as the ability to assign specific hardware requirements, 
better monitoring and reproducibility.

In order to use the local runner, you need to have a recent version of [docker-compose](https://docs.docker.com/compose/install/) installed.

### Installing the local runner

Make sure that you have installed Docker compose on your system. You can find more information 
about this in the [installation](../guides/installation.md) guide.


### Running a dataset with the local runner

Fondant will create a default working directory (for intermediate artifacts) for the dataset in the current working directory called '/.artifacts'. You can override this by passing the `--working-dir` argument to the `run` command. Or by setting the `working_dir` argument in the `run` method of the `DockerRunner` class.

=== "Console"

    ```bash
    fondant run local <dataset_ref>
    ```
    The dataset ref is the path to the compiled dataset spec or a reference to a fondant dataset.

    If you want to use remote paths (GCS, S3, etc.) you can use pass the correct cloud credentials flag when invoking the runner.
    This will mount your default local cloud credentials to the containers. Make sure you are authenticated locally before running the work and
    that you have the correct permissions to access the folder (read/write/create). 

    === "GCP"
    
        ```bash
        fondant run local <dataset_ref> --auth-provider gcp
        ```

    === "AWS"
    
        ```bash
        fondant run local <dataset_ref> --auth-provider aws
        ```

    === "Azure"
    
        ```bash
        fondant run local <dataset_ref> --auth-provider azure
        ```

    You can also use the `--extra-volumes` argument to mount extra credentials or additional files.
    This volumes will be mounted to every component/service of the docker-compose spec.


=== "Python"

    ```python 
    from fondant.dataset.runner import DockerRunner

    runner = DockerRunner()
    runner.run(extra_volumes=<str_or_list_of_optional_extra_volumes_to_mount>)
    ```

    If you want to use remote paths (GCS, S3, etc.) you can use the authentification argument 
    while invoking the runner.

    === "GCP"
    
        ```python
        from fondant.dataset.runner import DockerRunner
        from fondant.core.schema import CloudCredentialsMount

        runner = DockerRunner()
        runner.run(auth_provider=CloudCredentialsMount.GCP)
        ```

    === "AWS"
    
        ```python
        from fondant.dataset.runner import DockerRunner
        from fondant.core.schema import CloudCredentialsMount

        runner = DockerRunner()
        runner.run(auth_provider=CloudCredentialsMount.AWS)
        ```

    === "Azure"
    
        ```python
        from fondant.dataset.runner import DockerRunner
        from fondant.core.schema import CloudCredentialsMount

        runner = DockerRunner()
        runner.run(auth_provider=CloudCredentialsMount.AZURE)
        ```

    This will mount your default local cloud credentials to the containers. Make sure you are authenticated locally before running the work and
    that you have the correct permissions to access the folder (read/write/create). 

    You can also use the  optional `extra_volumes` argument to mount extra credentials or
    additional files.
    This volumes will be mounted to every component/service of the docker-compose spec.


The Docker compiler will compile the dataset to a docker compose specification before running the workflow. 
This will start the containers and provide logs per component (service).

Components that are not located in the registry (local custom components) will be built on runtime. This allows for quicker iteration
during component development. 


#### Assigning custom resources to the execution

The local runner uses the computation resources (RAM, CPU) of the host machine. In case a GPU is available and is needed for a component,
it needs to be assigned explicitly. 

```python
from fondant.dataset import Resources

dataset = dataset.apply(  
    "...",  
    arguments={  
     ...,  
    },  
    resources=Resources(
        accelerator_number=1,
        accelerator_name="GPU",
    )
)
```