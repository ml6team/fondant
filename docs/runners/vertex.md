### Vertex Runner

Uses Google
cloud's [Vertex AI pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction) to
help you
orchestrate your Fondant pipelines in a serverless manner. This makes it easy to scale up your
pipelines without worrying about infrastructure
deployment.

Vertex AI pipelines leverages Kubeflow pipelines under the hood. The Vertex compiler will take your
pipeline and compile it to a Kubeflow pipeline spec.
This spec can be used to run your pipeline on Vertex.

### Installing the Vertex runner

Make sure to install Fondant with the Vertex runner extra.

```bash
pip install fondant[vertex]
```

### Running a pipeline with Vertex

You will first need to make sure that your Google Cloud environment is properly setup. More
info [here](https://codelabs.developers.google.com/vertex-pipelines-intro#2)

=== "Console"
    
    ```bash 
    fondant run vertex <pipeline_ref> \
     --project-id $PROJECT_ID \
     --region $PROJECT_REGION \
     --service-account $SERVICE_ACCOUNT
    ```
    
    The pipeline ref is reference to a fondant pipeline (e.g. `pipeline.py`) where a pipeline instance
    exists.


=== "Python"
    
    ```python
    from fondant.pipeline.compiler import VertexCompiler
    from fondant.pipeline.runner import VertexRunner
    
    project_id = <the_gcp_project_id>
    project_region = <the_region_where_the_pipeline_will_run>
    service_account = <the_service_account_to_run_the_pipeline_with>

    compiler= VertexCompiler()
    compiler.compile(pipeline=<pipeline_object>)

    runner = VertexRunner(
        project_id=project_id,
        region=project_region,
        service_account=service_account)
    )
    runner.run(input_spec=<path_to_compiled_spec>)
    ```


Once your pipeline is running you can monitor it using the Vertex UI.

#### Assigning custom resources to the pipeline

The computation resources needs to be assigned explicitly, Vertex will then randomly attempt to
allocate
a machine that fits the resources. The GPU name needs to be assigned explicitly. Check
this [link](https://github.com/googleapis/python-aiplatform/blob/main/google/cloud/aiplatform_v1/types/accelerator_type.py)
for a list of available GPU resources. Make sure to check that the chosen GPU is available in the
region where the pipeline will be run.

```python
from fondant.pipeline import Resources

dataset = dataset.apply(
    "...",
    arguments={
        ...,
    },
    resources=Resources(
        accelerator_number=1,
        accelerator_name="NVIDIA_TESLA_K80",
        memory_limit="512M",
        cpu_limit="4",
    )
)
```
