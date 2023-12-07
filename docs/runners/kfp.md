### Kubeflow Runner

Leverages [Kubeflow pipelines](https://www.kubeflow.org/docs/components/pipelines/v1/introduction/)
on any Kubernetes cluster.
All Fondant needs is a url pointing to the Kubeflow pipeline host and an Object Storage provider (
S3, GCS, etc) to store data produced in the pipeline between steps.
We have compiled some references and created some scripts
to [get you started](https://fondant.readthedocs.io/en/latest/infrastructure) with setting up the
required infrastructure.

### Installing the Kubeflow runner

Make sure to install Fondant with the Kubeflow runner extra.

```bash
pip install fondant[kfp]
```

### Running a pipeline with Kubeflow

You will need a Kubeflow cluster to run your pipeline on and specify the host of that cluster. More
info on setting up a Kubeflow pipelines deployment and the host path can be found in
the [kubeflow infrastructure documentation](kfp_infrastructure.md).

=== "Console"
    
    ```bash 
    fondant run kubeflow <pipeline_ref> \
     --host $KUBEFLOW_HOST
    ```
    
    The pipeline ref is reference to a fondant pipeline (e.g. `pipeline.py`) where a pipeline instance
    exists.


=== "Python"
    
    ```python
    from fondant.pipeline.compiler import KubeFlowCompiler
    from fondant.pipeline.runner import KubeFlowRunner
    
    compiler= KubeFlowCompiler()
    compiler.compile(pipeline=<pipeline_object>)

    runner = KubeFlowRunner(host=<kubeflow_host>)
    runner.run(input_spec=<path_to_compiled_spec>)
    ```

Once your pipeline is running you can monitor it using the Kubeflow UI.

#### Assigning custom resources to the pipeline

Each component can optionally be constrained to run on particular node(s) using `node_pool_label`
and `node_pool_name`. You can find these under the Kubernetes labels of your cluster.
You can use the default node label provided by Kubernetes or attach your own. Note that the value of
these labels is cloud provider specific. Make sure to assign a GPU if required, the specified node
needs to
have an available GPU.

```python
from fondant.pipeline.pipeline import Resources

dataset = dataset.apply(
    "...",
    arguments={
        ...,
    },
    resources=Resources(
        accelerator_number=1,
        accelerator_name="GPU",
        node_pool_label="node_pool",
        node_pool_name="n2-standard-128-pool",
    )
```