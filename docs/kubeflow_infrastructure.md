# Setting up kubeflow 

## Introduction
In order to run Fondant on [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/v1/introduction/), we'll need:

- A kubernetes cluster
- Kubeflow pipelines installed on the cluster
- A registry to store custom component images like (docker hub, Github Container Registry, etc)

This can be on any kubernetes cluster, if you don't have access to a setup like this or you feel uncomfortable to setup your own we have provided some basic scripts to get you started on GCP or on a small scale locally.

!!! note "IMPORTANT"
    - These script serve just a kickstart to help you setup Kubeflow for running Fondant, these are not production ready environments.
    - Spinning up a cluster on a cloud vendor will incur a cost.
    - You should never run a script without inspecting it so please familiarize yourself with the commands defined in the Makefiles and adapt it to your own needs.

## If you already have a kubernetes cluster
 
If you already have setup a kubernetes cluster and you have configured kubectl you can install kubeflow pipelines following this [guide](https://www.kubeflow.org/docs/components/pipelines/v1/installation/standalone-deployment/#deploying-kubeflow-pipelines)


## Kubeflow on AWS

There are multiple guides on how to setup kubeflow pipelines on AWS:

- [official kubeflow distribution](https://awslabs.github.io/kubeflow-manifests/)
- [Kubeflow Pipelines on AWS](https://docs.aws.amazon.com/sagemaker/latest/dg/kubernetes-sagemaker-components-install.html)
- [deployment guide by kubeflow](https://awslabs.github.io/kubeflow-manifests/docs/deployment/)

Fondant needs the host url of kubeflow pipelines which you can [fetch](https://docs.aws.amazon.com/sagemaker/latest/dg/kubernetes-sagemaker-components-install.html#:~:text=.-,Access%20the%20KFP%20UI%20(Kubeflow%20Dashboard),-The%20Kubeflow%20Pipelines) (depending on your setup).

The BASE_PATH can be an [S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html)


## Kubeflow on Google Cloud

There are several ways to get up and running with kubeflow pipelines on Google Cloud.

- [On the GCP marketplace](https://console.cloud.google.com/marketplace/details/google-cloud-ai-platform/kubeflow-pipelines?project=thematic-lore-290312)
- [How to do a standalone deployment of kubeflow pipelines on GKE](https://www.kubeflow.org/docs/components/pipelines/v1/installation/standalone-deployment/)
- [Customizable deployments through overlays](https://www.kubeflow.org/docs/components/pipelines/v1/installation/standalone-deployment/#customizing-kubeflow-pipelines)

### OR you can use the scripts we provide to get a simple setup going

1. If you don't already have a google cloud project ready you can follow this [guide](https://v1-5-branch.kubeflow.org/docs/distributions/gke/deploy/project-setup/) to set one up, you will need to have set up billing.

2. Make sure you have the [gcloud cli](https://cloud.google.com/sdk/docs/install) installed (and it is the latest version) and that you have it configured to use your project by using `gcloud init`.

3. Setup [Default compute Region and Zone](https://cloud.google.com/compute/docs/gcloud-compute#default-region-zone)

3. Install [kubectl](https://kubernetes.io/docs/tasks/tools/) 

4. Run gcp.mk Makefile (located in the `scripts/` folder) which will do the following:
- Setup all gcp services needed 
- Start a GKE cluster
- Create a google storage bucket for data artifact storage
- Authenticate the local machine 
- Install kubeflow pipelines on the cluster

To run the complete makefile use (note this might take some time to complete):
```
make -f gcp.mk
```
Or run specific steps:
```
make -f gcp.mk authenticate-gcp-cluster
```

### Getting the variables for your pipeline

Running the following command:
```
make -f gcp.mk kubeflow-ui
```
Will print out the BASE_PATH and HOST which you can use to configure your pipeline. The HOST url will also allow you to use the kubeflow ui when opened in a browser.

### In order to delete the setup:
```
make -f gcp.mk delete
```

### More Information

- [Official documentation on cluster creation](https://cloud.google.com/kubernetes-engine/docs/how-to/creating-a-zonal-cluster)
- [Provision a GKE cluster with terraform](https://developer.hashicorp.com/terraform/tutorials/kubernetes/gke)
- [Use kubespray to setup a cluster](https://github.com/kubernetes-sigs/kubespray)

### More Information
- [Standalone deployments](https://www.kubeflow.org/docs/components/pipelines/v1/installation/standalone-deployment/)
- [Other local cluster installations](https://www.kubeflow.org/docs/components/pipelines/v1/installation/localcluster-deployment/)
- [Authenticating google cloud resources like storage and artifact registry](https://minikube.sigs.k8s.io/docs/handbook/addons/gcp-auth/)




