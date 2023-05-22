# 
Fondant is built on top of Kubeflow Pipelines, so to run it we'll need:

- A kubernetes cluster
- (OPTIONAL) A nodepool with GPU's available
- Kubeflow pipelines installed on the cluster

This can be on any kubernetes cluster, if you don't have access to a setup like this or you feel uncomfortable to setup your own we have provided some basic scripts to get you started on GCP or AWS.

IMPORTANT: These script serve just a kickstart to help you start using fondant fast, these are not production ready environments.
IMPORTANT 2: Spinning up a cluster on a cloud vendor will incur a cost.
IMPORTANT 3: You should never run a script without inspecting it so please familiarize  yourself with the commands defined in the Makefiles and adapt it to your own needs.

## If you already have a kubernetes cluster

If you already have setup a kubernetes cluster and you have configured kubectl you can install kubeflow pipelines following this [guide](https://www.kubeflow.org/docs/components/pipelines/v1/installation/standalone-deployment/#deploying-kubeflow-pipelines)

## Fondant on Google Cloud

There are several ways to get up and running with kubeflow pipelines on Google Cloud.
- [On the GCP marketplace](https://console.cloud.google.com/marketplace/details/google-cloud-ai-platform/kubeflow-pipelines?project=thematic-lore-290312)
- [How to do a standalone deployment of kubeflow pipelines on GKE](https://www.kubeflow.org/docs/components/pipelines/v1/installation/standalone-deployment/)
- [Customizable deployments through overlays](https://www.kubeflow.org/docs/components/pipelines/v1/installation/standalone-deployment/#customizing-kubeflow-pipelines)

### OR you can use the scripts we provide to get a simple setup going

1. If you don't already have a google cloud project ready you can follow this [guide](https://v1-5-branch.kubeflow.org/docs/distributions/gke/deploy/project-setup/) to set one up, you will need to have set up billing.

2. Make sure you have the [gcloud cli](https://cloud.google.com/sdk/docs/install) installed (and it is the latest version) and that you have it configured to use your project by using `gcloud init`.

3. Setup [Default compute Region and Zone](https://cloud.google.com/compute/docs/gcloud-compute#default-region-zone)

3. Install [kubectl](https://kubernetes.io/docs/tasks/tools/) 

4. Run Makefile which will do the following:
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
- [use kubespray to setup a cluster](https://github.com/kubernetes-sigs/kubespray)


## Fondant on AWS

https://awslabs.github.io/kubeflow-manifests/docs/deployment/

## Fondant on a local machine

### Using Minikube

1. Install Minikube following [this official installation guide](https://minikube.sigs.k8s.io/docs/start/)

2. Make sure you have docker [installed](https://docs.docker.com/desktop/) and the deamon is running

2. Run the makefile which will do the following:
- start a minikube cluster with 4 cpu's and 8gb of ram
- install kubeflow



is only for development
microk8s

- you can run components individually by 

```cd /your_component_folder```

and then calling the python code by running:

```python main.py --metadata '{"run_id":"YOUR RUN ID", "base_path": "/SOME_PATH"}' --output_manifest_path /SOME_PATH/manifest/manifest.txt```