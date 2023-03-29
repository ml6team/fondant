# dataset_loader_component

### Description

This is the first component of the pipeline, it loads an image dataset from the 🤗 [hub](https://huggingface.co/) and creates the initial manifest. This manifest includes 3 data sources: images, captions and image metadata.

### **Inputs/Outputs**
The component accepts a `dataset_name` as input which refers to a dataset on the 🤗 hub.

The component creates the first manifest as output that will be updated by subsequent components when new data sources
are added/filtered.  

See [`component.yaml`](component.yaml) for a more detailed description on all the input/output parameters.