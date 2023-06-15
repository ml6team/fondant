# Pipeline 

A Fondant pipeline is a tool for building complex workflows by creating a Directed Acyclic Graph (DAG) of different components that need to be executed. With Fondant, you can use both reusable components and custom components to construct your pipeline. In order to build a pipeline you register components on it with dependencies (if any) and Fondant will construct the graph automatically.

## Composing a pipeline 

To build a pipeline, you need to define a set of component operations called `ComponentOp`. A component operation encapsulates the specifications of the component and its runtime configuration.

The component specifications include the location of the Docker image in a registry.

The runtime configuration consists of the component's arguments and the definition of node pools and resources. For example, if a component requires GPU for model inference, you can specify the necessary GPU resources in the runtime configuration.

Here is an example of how to build a pipeline using Fondant:
```python
from fondant.pipeline import ComponentOp, Pipeline, Client

def build_pipeline():
    pipeline = Pipeline(pipeline_name="example pipeline", base_path="fs://bucket")

    load_from_hub_op = ComponentOp.from_registry(
        name="load_from_hf_hub",
        arguments={"dataset_name": "lambdalabs/pokemon-blip-captions"},
    )
    pipeline.add_op(load_from_hub_op)

    caption_images_op = ComponentOp(  
        component_spec_path="components/captioning_component/fondant_component.yaml",  
        arguments={  
            "model_id": "Salesforce/blip-image-captioning-base",  
            "batch_size": 2,  
            "max_new_tokens": 50,  
        },  
        number_of_gpus=1,  
        node_pool_name="model-inference-pool",  
    )
    pipeline.add_op(caption_images_op, dependencies=load_from_hub_op)
    return pipeline

```

In the example above, we first define our pipeline by providing a name as an identifier and a base path where the pipeline run artifacts will be stored. The base path can be a remote cloud location or a local directory, which is useful for local development.

Next, we define two operations: `load_from_hub_op`, which is a based from a reusable component loaded from the Fondant registry, and `caption_images_op`, which is a custom component defined by you. We add these components to the pipeline using the `.add_op()` method and specify the dependencies between components to build the DAG.


!!! note "IMPORTANT"
    Currently Fondant supports linear DAGs with single dependencies. Support for non-linear DAGs will be available in future releases.



## Compiling a pipeline

Once all your components are added to your pipeline you can use different compilers to run your pipeline:

### Kubeflow
TODO: update this once kubeflow compiler is implemented

~~Once the pipeline is built, you need to initialize the client with the kubeflow host path (more info about the host path can be found in the [infrastructure documentation](https://github.com/ml6team/fondant/blob/main/docs/infrastructure.md))
and use it to compile and run the pipeline with the `compile_and_run()` method. This performs static checking to ensure that all required arguments are provided to the components and that the required input data subsets are available. If the checks pass, a URL will be provided, allowing you to visualize and monitor the execution of your pipeline.~~

### Docker-Compose

The DockerCompiler will take your pipeline and create a docker-compose.yml file where every component is added as a service with the correct dependencies by leveraging the `depends_on` functionality and the `service_completed_successfully` status. See the basic example below:

```yaml
version: '3.8'
services:
  component_1:
    command: ["python", "main.py"]
    image: component_1:latest
  component_2:
    command: ["python", "main.py"]
    image: component_2:latest
    depends_on:
      component_1:
        condition: service_completed_successfully
  component_3:
    command: ["python", "main.py"]
    depends_on:
      component_2:
        condition: service_completed_successfully
    image: component_3:latest
```

In order to compile your pipeline to a `docker-compose` spec you need to import the `DockerCompiler`

```python
from fondant.compiler import DockerCompiler

compiler = DockerCompiler()
compiler.compile(pipeline=pipeline)
```

The DockerCompiler will try to see if the `base_path` of the pipeline is local or remote. If local the `base_path` will be mounted as a bind volume on every service/component.


#### Running a Docker compiled pipeline

Navigate to the folder where your docker compose is located and run (you need to have [docker-compose](https://docs.docker.com/compose/install/) installed)
```bash
docker compose up
```

This will start the pipeline and provide logs per component(service)