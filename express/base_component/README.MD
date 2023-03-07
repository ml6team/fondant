# base_component

### Description

This component contains common **system dependencies**  and **helper functions** that will be re-used across the different component. It 
will act as base component for the other components to build open (checkout this [link](https://docs.docker.com/build/building/multi-stage/#:~:text=Use%20multi%2Dstage%20builds%F0%9F%94%97)
for more info on multi-stage builds).

### **System Dependencies**
System dependencies include pytorch Cuda dependencies and the Google Cloud SDK which are needed to run
machine learning models and read/write to Google Cloud Storage respectively. 

### **Helper functions**
Consists of the following files:

1. **io_helpers.py**: general I/O helper functions.
2. **kfp_helpers.py**: include helper functions for GPU logging when running a KFP component and parsing specific KFP input.
3. **logger.py**: General logger module for event logging. 
4. **manifest_helpers.py**: Defines the structure of the data manifest that holds the location and contents of the different data sources. 
5. **parquet_helpers.py**: General helper functions for creating and updating the different parquet files that contain the index and data sources information (metadata, captions, ...) as well
as various utilities (duplication removal, metadata and column retrieval, ...). 
6. **storage_helpers.py**: helper functions to interact with Google Cloud Storage (blob listing, I/O, ...)

### **Practical considerations**

* Make sure to build this component first since the other components will fetch it during multi-stage buildings.
* Any changes to this component will require you to rebuild this component and the other components in order for the changes to take place. 