# 

In order to run fondant properly we'll need a couple of things:

- A kubernetes cluster
- (OPTIONAL) A nodepool with GPU's available
- Kubeflow pipelines installed on the cluster

This can be on any kubernetes cluster, if you don't have access to a setup like this or you feel uncomfortable to setup your own we have provided some basic scripts to get you started on GCP or AWS.

IMPORTANT: These script serve just a kickstart to help you start using fondant fast, these are not production ready environments.
IMPORTANT 2: Spinning up a cluster on a cloud vendor will incur a cost.
IMPORTANT 3: You should never run a script without inspecting it so please familiarize  yourself with the commands defined in the Makefiles and adapt it to your own needs.

## If you already have a kubernetes cluster

If you already have setup a kubernetes cluster and you have configured kubectl you can install kubeflow pipelines following this [guide](https://www.kubeflow.org/docs/components/pipelines/v1/installation/standalone-deployment/#deploying-kubeflow-pipelines)

## Fondant on GCP

1. If you don't already have a google cloud project ready you can follow this [guide](https://v1-5-branch.kubeflow.org/docs/distributions/gke/deploy/project-setup/) to set one up, you will need to have set up billing.

2. Make sure you have the [gcloud cli](https://cloud.google.com/sdk/docs/install) installed (and it is the latest version) and that you have it configured to use your project by using `gcloud init`.

3. Setup [Default compute Region and Zone](https://cloud.google.com/compute/docs/gcloud-compute#default-region-zone)

3. Install [kubectl](https://kubernetes.io/docs/tasks/tools/) 

4. Run Makefile which will do the following:
- Setup all gcp services needed 
- Start a GKE cluster
- Authenticate the local machine 
- Install kubeflow pipelines on the cluster

### More Information
- [How to do a standalone deployment of kubeflow pipelines on GKE](https://www.kubeflow.org/docs/components/pipelines/v1/installation/standalone-deployment/)
- [Official documentation on cluster creation](https://cloud.google.com/kubernetes-engine/docs/how-to/creating-a-zonal-cluster)
- [Provision a GKE cluster with terraform](https://developer.hashicorp.com/terraform/tutorials/kubernetes/gke)
- [use kubespray to setup a cluster](https://github.com/kubernetes-sigs/kubespray)


## Fondant on AWS

https://awslabs.github.io/kubeflow-manifests/docs/deployment/

## Fondant on a local machine

is only for development
microk8s