# Kubeflow Pipelines

This section contains information for getting started with Kubeflow 
Pipelines and documents best practices on how to use Kubeflow Pipelines.

## Kubeflow Pipelines context
Please checkout this [page](https://www.kubeflow.org/docs/components/pipelines/v1/concepts/) to learn about important concept of kubeflow.

## Layout

```
project root
+-- mlpipelines
    +-- components
    +-- pipelines
        +-- config <contains general and pipeline specific input configuration>
        +-- helpers
            +-- upload.py <helper functions to compile and upload the pipelines>
```

## Connecting the Kubeflow Pipelines UI

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

- Finally, once you're done, you're ready to build the images and push it to the [Artifact Registry](https://cloud.google.com/artifact-registry),
you can do so by running script (`sh build_image.sh`). Make sure to enable
[kaniko](https://cloud.google.com/build/docs/optimize-builds/kaniko-cache) for faster container build time.  


More information about the structure of a component can be found in this [section](example_components/example_component).
## Pipelines

Finally, once you have built all the separate components, you are ready to compile and upload the 
pipeline. The `upload.py` contains functions to help with the compilation and upload of new pipelines.

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

To get some experience with these lightweight components, 
you can checkout 
some [sample notebooks](https://github.com/Svendegroote91/kfp_samples) 
that highlight their main functionality.

## Re-usable and pre-made components

With Kubeflow Pipelines it's possible to re-use components across different 
pipelines. To see an example, checkout this [sample notebook](https://github.com/Svendegroote91/kfp_samples/blob/master/Reusable%20Components%20101.ipynb) 
that showcases how you can save and load a component.

The [AI Hub](https://aihub.cloud.google.com) also contains Kubeflow Pipelines 
components that you can 
easily re-use. Some interesting examples:

- [Gather training data by querying BigQuery](https://aihub.cloud.google.com/p/products%2F4700cd7e-2826-4ce9-a1ad-33f4a5bf7433)
- [Bigquery to TFRecords converter](https://aihub.cloud.google.com/p/products%2F28a006d0-c833-4c68-98ff-37358eeb7726)
- [Executing an Apache Beam Python job in Cloud Dataflow](https://aihub.cloud.google.com/p/products%2F44999f4a-1668-4d42-a4e3-1269a8786840)
- [Submitting a Cloud ML training job as a pipeline step](https://aihub.cloud.google.com/p/products%2Ffbe29250-9b67-4dfb-8900-d6ce41cdb85a)
- [Deploying a trained model to Cloud Machine Learning Engine](https://aihub.cloud.google.com/p/products%2F7a08de6c-3864-4ccf-8151-4119e1b4e890)
- [Batch predicting using Cloud Machine Learning Engine](https://aihub.cloud.google.com/p/products%2F3d5d2340-0eb2-4b03-aecc-ae34f6105822)
