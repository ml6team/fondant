## Using KubeFlow on GCP

This file includes some tips and tricks to set up and deploy a KubeFlow pipeline on GCS.

### Connecting the Kubeflow Pipelines UI

There are two ways to connect to KFP UI, first make sure you autheticate to the GKE cluster
hosting KFP:

```
gcloud container clusters get-credentials [CLUSTER_NAME} --zone=[ZONE]
```

1) Port-forwarding to access the kubernetes service

    ```
    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
    ```

   Open the Kubeflow Pipelines UI at <http://localhost:8080>


2) Using the IAP URL of KFP

    ```
    kubectl describe configmap inverse-proxy-config -n kubeflow | grep googleusercontent.com
    ```

### Pipelines access issues
In case you have any authentication issues when compile the pipeline or accessing kubeflow: 

* Make sure that you're authenticated with the right credentials and to the right project by running
`gcloud init` in your local terminal.  

* If you still run into issues, you can try running `gcloud auth application-default login` and
select the right account in the pop up page, this will make sure that your local application
temporarily uses your own user credentials for API access.  

* Finally, if the issues still persists. Make sure that the IP address you're trying to connect to 
is listed within the [authorized networks](https://github.com/creativefabrica/fondant-infrastructure/blob/master/terraform/modules/kfp/variables.tf)
of the GKE cluster where Kubeflow is deployed. 


## Implementing components

Pipelines exist out of several components. These are stored in a `components` folder. Each component requires
the setup of a Dockerfile. One typically has the following files for each component:

- a src/main.py script that implements the main logic
- component.yaml that defines the input and output parameters
- Dockerfile
- build_image.sh
- requirements.txt 

To implement these components faster, it's recommended to leverage the boilerplate components available in the `fondant.components` module and include `fondant` in the `requirements.txt` file of the component. Regarding the `component.yaml` file, one typically defines a `components.config` file at the pipeline root level. This file defines the root of the directory where artifacts get placed (Docker images). Each component's `component.yaml` file then defines the actual path to the Docker image.

When these files are implemented, you're ready to build the images and push it to the [Artifact Registry](https://cloud.google.com/artifact-registry). You can do so by running the script (`sh build_image.sh`). Make sure to enable [kaniko](https://cloud.google.com/build/docs/optimize-builds/kaniko-cache) for faster container build time. One can run the following command locally:

```
gcloud config set builds/use_kaniko True
```

This ensures that, when building a new image, only files that are changed will get rebuild. See [this link](https://cloud.google.com/build/docs/optimize-builds/kaniko-cache#configuring_the_cache_expiration_time) for more details.

## Pipelines

Next, once you have built all the separate components, you are ready to define the pipeline. A pipeline is defined in a script (like `my_pipeline.py`). In this script, one can leverage the `@dsl.pipeline` annotator like so:

python
```
@dsl.pipeline(
    name='XGBoost Trainer',
    description='A trainer that does end-to-end distributed training for XGBoost models.'
)
def xgb_train_pipeline():
   return -1
```

Next, make sure to include the following in the main pipeline script:

```
from fondant.kfp_utils import compile_and_upload_pipeline

if __name__ == '__main__':
    compile_and_upload_pipeline(pipeline=example_pipeline,
                                host=KubeflowConfig.HOST,
                                env=KubeflowConfig.ENV)
```

To compile the pipeline, simply execute the file. The pipeline will be uploaded to KubeFlow under a specific version
where it can be visualized and run. To run the pipeline, you can go to the KubeFlow UI and click "run".

### Create configurations

For each component, a `Config` class can be created which defines a configuration (hyperparameters) for each component. Typically, also a `GeneralConfig` is defined which includes general information like the cloud project ID.

The config classes are then imported in the pipeline, and pass to the op functions, which in turn pass them to the component scripts.

## Example of a typical pre-process, train and deploy pipeline

A typical workflow to automate with Kubeflow Pipelines is training and 
deployment of a Machine Learning model. 
Please check out the [example](https://github.com/kubeflow/examples/tree/master/financial_time_series#kubeflow-pipelines) 
that showcases this automated workflow with conditional deployment (based on 
accuracy threshold). The pipeline code is stored [here](https://github.com/kubeflow/examples/blob/master/financial_time_series/tensorflow_model/ml_pipeline.py).


## Iterate and develop quickly with Lightweight components

One of the primary goals of the lightweight components is to enable to iterate 
and develop quickly on a Kubeflow Pipeline, from a notebook environment. 
Next to that, the lightweight components can be useful for small steps in a pipeline 
that typically only require a few lines of Python code. Instead of having to 
containerize this small Python function, you can simply 'mount' it on top of 
an existing image.

To get some experience with these lightweight components,you can checkout some [sample notebooks](https://github.com/Svendegroote91/kfp_samples) that highlight their main functionality.