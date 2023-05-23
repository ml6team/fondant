LOCATION = "europe-west1"
CLUSTER_NAME = "fondant-cluster"
GCP_PROJECT_ID = ${shell gcloud config list --format 'value(core.project)'}
BUCKET_NAME = gs://$(GCP_PROJECT_ID)_kfp_artifacts
KUBEFLOW_PIPELINE_VERSION = 1.8.5


gcp: setup-gcp-services provision-gcp-cluster create-artifact-bucket authenticate-gcp-cluster variables install-kfp kubeflow-ui

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
	krmapihosting.googleapis.com \
	artifactregistry.googleapis.com

provision-gcp-cluster:
	gcloud container clusters create ${CLUSTER_NAME} \
     --machine-type "e2-standard-2" \
     --scopes "cloud-platform"

create-artifact-bucket:
	gcloud storage buckets create ${BUCKET_NAME}

create-artifact-repo:
	gcloud artifacts repositories create fondant-images \
		--repository-format=docker \
		--location=${LOCATION} \
		--description="Repository for hosting images used by fondant pipelines"

authenticate-gcp-cluster:
	gcloud container clusters get-credentials ${CLUSTER_NAME} \

install-kfp:
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${KUBEFLOW_PIPELINE_VERSION}"
	kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=${KUBEFLOW_PIPELINE_VERSION}"

kubeflow-ui:
	$(info Use these variables to configure the fondant pipeline:) \
	$(info BASE_PATH = "${BUCKET_NAME}/fondant_artifacts") \
    $(info HOST = "https://$(shell kubectl describe configmap inverse-proxy-config -n kubeflow | grep googleusercontent.com)")

delete:
	gcloud storage rm --recursive ${BUCKET_NAME}
	gcloud container clusters delete ${CLUSTER_NAME} 