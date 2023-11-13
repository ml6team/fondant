# Pipeline

A Fondant pipeline is a tool for building complex workflows by creating a Directed Acyclic Graph (
DAG) of different components that need to be executed. With Fondant, you can use both reusable
components and custom components to construct your pipeline. In order to build a pipeline you
register components on it with dependencies (if any) and Fondant will construct the graph
automatically.

## Composing a pipeline

To build a pipeline, you need to define a set of component operations called `ComponentOp`. A
component operation encapsulates the specifications of the component and its runtime configuration.

The component specifications include the location of the Docker image in a registry.

The runtime configuration consists of the component's arguments and the definition of node pools,
resources and custom partitioning specification.
For example, if a component requires GPU for model inference, you can specify the necessary GPU
resources in the runtime configuration.

Here is an example of how to build a pipeline using Fondant:

```python
from fondant.pipeline import ComponentOp, Pipeline, Resources

pipeline = Pipeline(pipeline_name="example pipeline", base_path="fs://bucket")

load_from_hub_op = ComponentOp.from_registry(
    name="load_from_hf_hub",
    arguments={"dataset_name": "lambdalabs/pokemon-blip-captions"},
)

caption_images_op = ComponentOp(
    component_dir="components/captioning_component",
    arguments={
        "model_id": "Salesforce/blip-image-captioning-base",
        "batch_size": 2,
        "max_new_tokens": 50,
    },
    resources=Resources(
        accelerator_number=1,
        accelerator_name="GPU",
    )
)

pipeline.add_op(load_from_hub_op)
pipeline.add_op(caption_images_op, dependencies=load_from_hub_op)
```

In the example above, we first define our pipeline by providing a name as an identifier and a base
path where the pipeline run artifacts will be stored.

The base path can be:

* **A remote cloud location (S3, GCS, Azure Blob storage)**: valid across all runners.
  For the **local runner**, make sure that your local credentials or service account have read/write
  access to the
  designated base path and that they are mounted. <br>
  For the **Vertex** and **Kubeflow** runners, make sure that the service account attached to those
  runners has read/write access.
* **A local directory**: only valid for the local runner, points to a local directory. This is
  useful for local development.

Next, we define two operations: `load_from_hub_op`, which is a based from a reusable component
loaded from the Fondant registry, and `caption_images_op`, which is a custom component defined by
you. We add these components to the pipeline using the `.add_op()` method and specify the
dependencies between components to build the DAG.

!!! note "IMPORTANT"  

    Currently Fondant supports linear DAGs with single dependencies. Support for non-linear DAGs will be
    available in future releases.

## Compiling a pipeline

Once all your components are added to your pipeline you can use different compilers to run your
pipeline:

!!! note "IMPORTANT"  

    When using other runners you will need to make sure that your new environment has access to:

    - The base path of your pipeline (as mentioned above)
    - The images used in your pipeline (make sure you have access to the registries where the images are
    stored)

=== "Console"
    
    === "Local"
    
        ```bash
        fondant compile local <pipeline_ref>
        ```
    === "Vertex"
    
        ```bash
        fondant compile vertex <pipeline_ref>
        ```

    === "Kubeflow"
    
        ```bash
        fondant compile kfp <pipeline_ref>
        ```

    The pipeline ref is reference to a fondant pipeline (e.g. `pipeline.py`) where a pipeline instance
    exists (see above).
    This will produce a pipeline spec file associated with a given runner.
   
=== "Python"

    === "Local"
    
        ```python
        from fondant.pipeline.compiler import DockerCompiler
        from fondant.pipeline.runner import DockerRunner
        
        EXTRA_VOLUMES = <str_or_list_of_optional_extra_volumes_to_mount>
        compiler = DockerCompiler(extra_volumes=EXTRA_VOLUMES)
        compiler.compile(pipeline=<pipeline_object>)
    
        runner = DockerRunner()
        runner.run(input_spec=<path_to_compiled_spec>)
        ```  

    === "Vertex"
    
        ```python
        from fondant.pipeline.compiler import VertexCompiler

        compiler = VertexCompiler()
        compiler.compile(pipeline=<pipeline_object>)
        ```

    === "KubeFlow"
    
        ```python
        from fondant.pipeline.compiler import KubeFlowCompiler

        compiler = KubeFlowCompiler()
        compiler.compile(pipeline=<pipeline_object>)
        ```

## Running a pipeline

=== "Console"

    === "Local"
    
        ```bash
        fondant run local <pipeline_ref>
        ```
    === "Vertex Runner"
    
        ```bash 
        fondant run vertex <pipeline_ref> \
         --project-id $PROJECT_ID \
         --project-region $PROJECT_REGION \
         --service-account $SERVICE_ACCOUNT
        ```

    === "Kubeflow"
    
        ```bash
        fondant run kubeflow <pipeline_ref>
        ```
    Here, the pipeline ref can be either be a path to a compiled pipeline spec or a reference to fondant
    pipeline (e.g. `pipeline.py`) in which case
    the pipeline will first be compiled to the corresponding runner specification before running the
    pipeline.

=== "Python"

    === "Local"
    
        ```python
        from fondant.pipeline.runner import DockerRunner

        runner = DockerRunner()
        runner.run(input_spec=<path_to_compiled_spec>)
        ```
    === "Vertex"
    
        ```python
        from fondant.pipeline.runner import VertexRunner

        runner = VertexRunner()
        runner.run(input_spec=<path_to_compiled_spec>)
        ```

    === "KubeFlow"
    
        ```python
        from fondant.pipeline.runner import KubeFlowRunner

        runner = KubeFlowRunner(host=<kubeflow_host>)
        runner.run(input_spec=<path_to_compiled_spec>)        
        ```
    
    Where the path to the compiled spec is the path to the compiled pipeline spec file produced
    by the compiler.

