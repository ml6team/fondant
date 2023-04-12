# Welcome to Fondant

Fondant helps you create data processing pipelines to train foundation models, like Stable Diffusion or GPT language models.

Fondant offers a set of reusable components, such as:
- image filtering
- language filtering
- deduplication
- image expansion based on LAION retrieval

and so on, to help you train foundation models faster. 

## Fondant internals

The central object in Fondant is the manifest, which ensures that data is being read and written efficiently.

Fondant is built on top of [KubeFlow pipelines](https://github.com/kubeflow/pipelines), which are machine learning workflows that run on Kubernetes.
