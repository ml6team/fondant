## Using KubeFlow on GCP

### Connecting the Kubeflow Pipelines UI

There are two ways to connect to KFP UI, first make sure you autheticate to the GKE cluster
hosting KFP

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
is listed within the [authorized networks](https://github.com/creativefabrica/express-infrastructure/blob/master/terraform/modules/kfp/variables.tf)
of the GKE cluster where Kubeflow is deployed. 


## Component

Pipelines exist out of several components. These are stored in `mlpipelines/components`. Each component requires
the setup of a Dockerfile 

There are multiple steps for setting up the component:  

- Write the main script of your component within the `src` directory inside the component.  

- Define all the input and output parameters of your component in the `component.yaml` file  

- Specify the requirement packages of the component in the `requirements.txt` file and specify the build 
steps in the `Dockerfile`.  

- Finally, once you're done, you're ready to build the images and push it to the [Artifact Registry](https://cloud.google.com/artifact-registry). You can do so by running the script (`sh build_image.sh`). Make sure to enable [kaniko](https://cloud.google.com/build/docs/optimize-builds/kaniko-cache) for faster container build time.  

More information about the structure of a component can be found in this [section](example_components/example_component).

## Pipelines

Finally, once you have built all the separate components, you are ready to compile and upload the 
pipeline. The `upload.py` script contains functions to help with the compilation and upload of new pipelines.

[Here](pipelines/example_pipeline.py) is a simple example demonstrate a toy pipeline, you can see how parameters are passed between different steps of the pipeline.
To compile the pipeline, simply execute the file, the pipeline will be uploaded to kubeflow under a specific version
where it can be visualized and run. 

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