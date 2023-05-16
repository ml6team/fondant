CLUSTER_NAME = "fondant-cluster"

KUBEFLOW_PIPELINE_VERSION = 1.8.5


gcp: setup-gcp-services provision-gcp-cluster authenticate-gcp-cluster install-kfp

setup-gcp-services:
	gcloud services enable \
	compute.googleapis.com \
	container.googleapis.com \
	iam.googleapis.com \
	servicemanagement.googleapis.com \
	cloudresourcemanager.googleapis.com \
	ml.googleapis.com \
	iap.googleapis.com \
	sqladmin.googleapis.com \
	meshconfig.googleapis.com \
	krmapihosting.googleapis.com

provision-gcp-cluster:
	gcloud container clusters create ${CLUSTER_NAME} \
     --machine-type "e2-standard-2" \
     --scopes "cloud-platform"

authenticate-gcp-cluster:
	gcloud container clusters get-credentials ${CLUSTER_NAME} \

install-kfp:
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${KUBEFLOW_PIPELINE_VERSION}"
	kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=${KUBEFLOW_PIPELINE_VERSION}"

delete:
	gcloud container clusters delete ${CLUSTER_NAME} 
