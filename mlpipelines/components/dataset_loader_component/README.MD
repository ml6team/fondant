# dataset_loader_component

### Description

This is the first component of the pipeline, it loads in an image dataset from a specific 
[Google Cloud Storage (GCS)](https://cloud.google.com/storage/docs) path and creates a
parquet files with different metadata (image path, format, size, ...).

### **Inputs/Outputs**
The component accepts a `source-dataset-bucket` as input with a reference to the blob `source-dataset-blob`
where the image dataset is located.  

The component created the first data manifest that will be updated in subsequent components when new dataset sources
are added/filtered.  

See [`component.yaml`](component.yaml) for a more detailed description on all the input/output parameters. 

### **Practical considerations**

* The main accepted formats of the input images are either `png`, `jpg` or `svg`. Eventually, all the formats will be converted to 
`jgp` in a subsequent component since it is more suitable for machine learning inference and training.
* Make sure that your images are located in one directory on GCS and not spread across different directories. 