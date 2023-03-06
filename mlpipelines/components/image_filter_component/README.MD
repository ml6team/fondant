# dataset_filter_component

### Description

This component is used to filter the images based on the metadata
attributes to only keep images of a certain size and format.

### **Inputs/Outputs**
The component accepts as input different filtering parameters of image size and formats to keep. The filtering
is done against the `dataset.parquet` which contains the relevant metadata used for filtering. 
The **index.parquet** of the dataset source is then updated to include only the images that passed the filter. 

See [`component.yaml`](component.yaml) for a more detailed description on all the input/output parameters. 

### **Practical considerations**

* The reason for only allowing `png` and `svg` (as well as `jpg`) is that the [`image_conversion_component`](../image_conversion_component)
currently only supports converting those formats to `jpg`. If you have a different data format that you want to include,
make sure to add a method to convert to `jpg` in the `image_conversion_component` and add that format to the `image_formats` so that 
images of those formats won't get filtered. 