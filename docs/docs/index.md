# Welcome to Fondant

Fondant helps you create pipelines for training foundation models.

## Fondant internals

The central object in Fondant is the manifest, which ensures that data is being read and written efficiently.

Fondant is built on top of [KubeFlow pipelines](https://github.com/kubeflow/pipelines), which are machine learning workflows that run on Kubernetes.
