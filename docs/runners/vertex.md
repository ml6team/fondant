### Vertex Runner

Uses Google
cloud's [Vertex AI pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction) to
help you
orchestrate your Fondant workflows in a serverless manner. This makes it easy to scale up your
pipelines without worrying about infrastructure
deployment.

Vertex AI pipelines leverages Kubeflow pipelines under the hood. The Vertex compiler will take your
Fondant workflow and compile it to a Kubeflow pipeline spec.
This spec can be used to run your workflow on Vertex.

### Installing the Vertex runner

Make sure to install Fondant with the Vertex runner extra.

```bash
pip install fondant[vertex]
```

### Running a dataset with Vertex

You will first need to make sure that your Google Cloud environment is properly setup. More
info [here](https://codelabs.developers.google.com/vertex-pipelines-intro#2)

=== "Console"
    
    ```bash 
    fondant run vertex <dataset_ref> \
     --working-dir $GCP_BUCKET \ 
     --project-id $PROJECT_ID \
     --region $PROJECT_REGION \
     --service-account $SERVICE_ACCOUNT
    ```

    The dataset ref is reference to a fondant dataset (e.g. `pipeline.py`) where a dataset instance
    exists.


=== "Python"
    
    ```python
    from fondant.dataset.compiler import VertexCompiler
    from fondant.dataset.runner import VertexRunner
    
    project_id = <the_gcp_project_id>
    project_region = <vertex_region>
    service_account = <the_service_account_to_be_used_by_vertex>

    compiler= VertexCompiler()
    compiler.compile(dataset=<dataset_object>)

    runner = VertexRunner(
        project_id=project_id,
        region=project_region,
        service_account=service_account)
    )
    runner.run(input_spec=<path_to_compiled_spec>, working_dir=<working_dir>)
    ```


Once your pipeline is running you can monitor it using the Vertex UI.

#### Assigning custom resources to the vertex pipeline

The computation resources needs to be assigned explicitly, Vertex will then randomly attempt to
allocate
a machine that fits the resources. The GPU name needs to be assigned explicitly. Check
this [link](https://github.com/googleapis/python-aiplatform/blob/main/google/cloud/aiplatform_v1/types/accelerator_type.py)
for a list of available GPU resources. Make sure to check that the chosen GPU is available in the
region where the pipeline will be run.

```python
from fondant.dataset import Resources

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
