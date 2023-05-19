KUBEFLOW_PIPELINE_VERSION = 1.8.5

all: start-cluster install-kfp

start-cluster:
	minikube start  --memory 8192 --cpus 4 --driver=docker

install-kfp:
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${KUBEFLOW_PIPELINE_VERSION}"
	kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=${KUBEFLOW_PIPELINE_VERSION}" && \
	kubectl rollout status -n kubeflow deployment/ml-pipeline -w