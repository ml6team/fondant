# Pipeline

A Fondant pipeline is a tool for building complex workflows by creating a Directed Acyclic Graph (DAG) of different components that need to be executed. With Fondant, you can use both reusable components and custom components to construct your pipeline. Fondant extends the functionality of Kubeflow and provides additional features to simplify pipeline development.


## Building a pipleine 

To build a pipeline, you need to define a set of component operations called `ComponentOp`. A component operation encapsulates the specifications of the component and its runtime configuration.

The component specifications include the location of the Docker image in the artifact registry. It is important to ensure that your Kubeflow service account has access to the image registry if it is not public.

The runtime configuration consists of the component's arguments and the definition of node pools and resources. For example, if a component requires GPU for model inference, you can specify the necessary GPU resources in the runtime configuration.

Here is an example of how to build and compile a pipeline using Fondant:
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
	pipeline.add_op(custom_op, dependencies=load_from_hub_op)
	return pipeline

if __name__ == "__main__":
    client = Client(host="https://kfp-host.com/")
    pipeline = build_pipeline()
    client.compile_and_run(pipeline=pipeline)
```

In the example above, we first define our pipeline by providing a name as an identifier and a base path where the pipeline run artifacts will be stored. The base path can be a remote cloud location or a local directory, which is useful for local development.

Next, we define two operations: `load_from_hub_op`, which is a based from a reusable component loaded from the Fondant registry, and `caption_images_op`, which is a custom component defined by you. We add these components to the pipeline using the `.add_op()` method and specify the dependencies between components to build the DAG.

Please note that currently Fondant supports linear DAGs with single dependencies. Support for non-linear DAGs will be available in future releases.

Once the pipeline is built, you can execute it by calling the `compile_and_run()` method. This performs static checking to ensure that all required arguments are provided to the components and that the required input data subsets are available. If the checks pass, a URL will be provided, allowing you to visualize and monitor the execution of your pipeline.
