# Load from files

## Description
This component is based on the `DaskLoadComponent` and is used to load dataset from files within a directory. 
It allows you to load datasets which
- Have files within a local data directory 
- Have compressed files like .zip, gzip, tar or tar.gz within the data directory
- Are hosted on remote locations like AWS S3 bucket, Azure's Blob storage or GCP's cloud storage

And returns a dataframe with two columns
- file_filename(containing the file name in string format)
- file_content (containing the respective file content in bytes format)

Here is an illustration of how to use this component in your pipeline
on a local directory with zip files

```python
from fondant.pipeline import Pipeline, ComponentOp

my_pipeline = Pipeline(
    pipeline_name="my_pipeline",
    base_path="./",  # TODO: update this
    pipeline_description="This is my pipeline",
)

load_from_files = ComponentOp(
    component_dir="components/load_from_files",
    arguments={
        "directory_uri": "./data.zip", # change this to your
                                       # directory_uri, remote or local
    },
    output_partition_size="10MB",
)

my_pipeline.add_op(load_from_files, dependencies=[])
```