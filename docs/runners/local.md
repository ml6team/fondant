### Local Runner

Leverages [docker compose](https://docs.docker.com/compose/). The local runner is mainly aimed
at helping you develop fondant pipelines and components faster since it allows you to develop on
your local machine or a Virtual Machine. This enables you to quickly iterate on development. Once
you have a pipeline developed, Switching to either the [Vertex](vertex.md) or [Kubeflow](kfp.md) runners 
offers many advantages such as the ability to assign specific hardware requirements, 
better monitoring and pipeline reproducibility.

In order to use the local runner, you need to have a recent version of [docker-compose](https://docs.docker.com/compose/install/) installed.

### Installing the Local runner

Make sure that you have installed Docker compose on your system. You can find more information 
about this in the [installation](../guides/installation.md) guide.


=== "Console"

    ```bash
    fondant run local <pipeline_ref>
    ```

    If you want to use remote paths (GCS, S3, etc.) you can use the `--auth-gcp`, `--auth-aws` or `--auth-azure`.
    This will mount your default local cloud credentials to the pipeline. Make sure you are authenticated locally before running the pipeline and
    that you have the correct permissions to access the `base_path` of the pipeline (read/write/create). 
    
    You can also use the `--extra_volumes` argument to mount extra credentials or additional files.
    This volumes will be mounted to every component/service of the docker-compose spec.


=== "Python"

    ```python 
    from fondant.pipeline.compiler import DockerCompiler
    from fondant.pipeline.runner import DockerRunner
    
    EXTRA_VOLUMES = <str_or_list_of_optional_extra_volumes_to_mount>
    compiler = DockerCompiler(extra_volumnes=EXTRA_VOLUMES)
    compiler.compile(pipeline=<pipeline_object>)

    runner = DockerRunner()
    runner.run(input_spec=<path_to_compiled_spec>)
    ```

    If you want to use remote paths (GCS, S3, etc.) you can use pass the default local cloud credentials to the pipeline.

    === "GCP"
    
        ```python
        from fondant.pipeline.compiler import DockerCompiler
        from fondant.pipeline.runner import DockerRunner
        from fondant.core.schema import CloudCredentialsMount
        
        gcp_mount_dir = CloudCredentialsMount.GCP.value
        compiler = DockerCompiler(extra_volumnes=gcp_mount_dir)
        compiler.compile(pipeline=<pipeline_object>)

        runner = DockerRunner()
        runner.run(input_spec=<path_to_compiled_spec>)
        ```

    === "AWS"
    
        ```python
        from fondant.pipeline.compiler import DockerCompiler
        from fondant.pipeline.runner import DockerRunner
        from fondant.core.schema import CloudCredentialsMount
        
        aws_mount_dir = CloudCredentialsMount.AWS.value
        compiler = DockerCompiler(extra_volumnes=aws_mount_dir)
        compiler.compile(pipeline=<pipeline_object>)

        runner = DockerRunner()
        runner.run(input_spec=<path_to_compiled_spec>)
        ```

    === "Azure"
    
        ```python
        from fondant.pipeline.compiler import DockerCompiler
        from fondant.pipeline.runner import DockerRunner
        from fondant.core.schema import CloudCredentialsMount
        
        azure_mount_dir = CloudCredentialsMount.AZURE.value
        compiler = DockerCompiler(extra_volumnes=azure_mount_dir)
        compiler.compile(pipeline=<pipeline_object>)

        runner = DockerRunner()
        runner.run(input_spec=<path_to_compiled_spec>)
        ```

    This will mount your default local cloud credentials to the pipeline. Make sure you are authenticated locally before running the pipeline and
    that you have the correct permissions to access the `base_path` of the pipeline (read/write/create). 

    You can also use the  optional `extra_volumes` argument to mount extra credentials or
    additional files.
    This volumes will be mounted to every component/service of the docker-compose spec.


**NOTE:**  The pipeline ref is the path to the compiled pipeline spec or a reference to a fondant pipeline in which case a Docker compiler will compile the pipeline
to a docker compose specification before running the pipeline. This will start the pipeline and provide logs per component (service).

Components that are not located in the registry (local custom components) will be built on runtime. This allows for quicker iteration
during component development. 

The local runner will try to check if the `base_path` of the pipeline is a local or remote storage. If it's local, the `base_path` will be mounted as a bind volume on every service/component.


#### Assigning custom resources to the pipeline

The local runner uses the computation resources (RAM, CPU) of the host machine. In case a GPU is available and is needed for a component,
it needs to be assigned explicitly. 

```python
from fondant.pipeline.pipeline import ComponentOp, Resources

component = ComponentOp(  
    component_dir="...",  
    arguments={  
     ...,  
    },  
    resources=Resources(
        accelerator_number=1,
        accelerator_name="GPU",
    )
)
```