

## Setting up the environment

### Installing fondant

We suggest that you use a [virtual environment](https://docs.python.org/3/library/venv.html) for you project. Fondant supports Python >=3.8.

Fondant can be installed using pip:

```bash
pip install fondant
```

For the latest development version, you might want to install from source instead:
```bash
pip install git+https://github.com/ml6team/fondant.git
```
You can validate the installation of fondant by running its root CLI command:
```bash
fondant --help
```

## Your first pipeline

Create a `pipeline.py` file in the root of your project and add the following code:

```Python
from fondant.pipeline import Pipeline, ComponentOp

my_pipeline = Pipeline(
    pipeline_name='my_pipeline',
    base_path='/home/username/my_pipeline',
    pipeline_description='This is my pipeline',
)
```

This is all you need to initialize a fondant pipeline:

- **pipeline_name**: A name to reference your pipeline.
- **base_path**: The base path of your pipeline, this will be used to store artifacts: the data between steps and the pipeline manifests. This base_path can be a local path or a cloud storage path (e.g. s3://my_bucket/artifacts, or gs://my_bucket/artifacts).
- **pipeline_description**: A description of your pipeline.

## Adding components

Now that we have a pipeline, we can add components to it. Components are the building blocks of your pipeline. They are the individual steps that will be executed in your pipeline. There are 2 main types of components:

- reusable components: These are components that are already created by the community and can be easily used in your pipeline. You can find a list of reusable components [here](https://github.com/ml6team/fondant/tree/main/components). They often have arguments that you can set to configure them for your use case.

- custom components: These are the components you create to solve your use case. A custom component can be easily created by adding a `fondant_component.yaml`, `dockerfile` and `main.py` file to your component subdirectory. This file contains the metadata of your component. You can find more information about the `fondant_component.yaml` file [here](https://github.com/ml6team/fondant/blob/main/docs/component_spec.md)

Let's add a reusable component to our pipeline. We will use the `load_from_hf_hub` component to read data from huggingface. Add the following code to your `pipeline.py` file:

```Python
load_from_hf_hub = ComponentOp.from_registry(
    name='load_from_hf_hub',
    component_spec_path='components/load_from_hf_hub/component_spec.yml',
    arguments={
        'dataset_name': 'huggan/pokemon',
        'n_rows_to_load': 100,
        'column_name_mapping': {
            'image': 'images_raw',
        },
        "image_column_names": ["image"],
        
    }
)

my_pipeline.add_op(load_from_hf_hub, dependencies=[])
```

Two things are happening here:
1. We created a ComponentOp from the registry. This is a reusable component, we pass it arguments needed to configure it

- **dataset_name**: The name of the dataset on huggingface hub, here we load a [dataset with pokemon images](https://huggingface.co/datasets/huggan/pokemon)
- **n_rows_to_load**: The number of rows to load from the dataset. This is useful for testing your pipeline on a small scale.
- **column_name_mapping**: A mapping of the columns in the dataset to the columns in fondant. Here we map the `image` column in the dataset to the `images_raw` subset_column in fondant.
- **image_column_names**: A list of the original image column names in the dataset. This is used to format the image from the huggingface hub format to a byte string.


2. We add our created componentOp to the pipeline using the `add_op` method. This component has no dependencies since it is the first component in our pipeline.


Next create a file `load_from_hf_hub/component_spec.yml` with the following content:

```yaml
name: Load from hub
description: Component that loads a dataset from the hub
image: ghcr.io/ml6team/load_from_hf_hub:latest

produces:
  images: # subset name
    fields:
      raw: # field name
        type: binary # field type


args:
  dataset_name:
    description: Name of dataset on the hub
    type: str
  column_name_mapping:
    description: Mapping of the consumed hub dataset to fondant column names
    type: dict
  image_column_names:
    description: Optional argument, a list containing the original image column names in case the 
      dataset on the hub contains them. Used to format the image from HF hub format to a byte string.
    type: list
    default: None
  n_rows_to_load:
    description: Optional argument that defines the number of rows to load. Useful for testing pipeline runs on a small scale
    type: int
    default: None
```

This is the component spec of the component we have just added to our pipelines, the only thing we have altered is the `produces` section. We have defined what subsets, fields and types this component produces.

Your project should look like this now:
```
.
├── components
│   └── load_from_hf_hub
│       └── component_spec.yml
└── pipeline.py
```

We now have a fully functional Fondant pipeline, it does not have much functionality yet, but it is a good starting point to build upon. We can already try running this limited example in order to validate our setup.

## Running your pipeline

A Fondant pipeline needs to be compiled before it can be ran. This means translating the user friendly Fondant pipeline definition into something that can be executed by a runner.

There are currently 2 runners available:
- Local runner: This runner runs the pipeline locally on your machine. This is useful for testing your pipeline. We leverage Docker Compose to compile and run the pipeline locally.
- Kubeflow runner: This runner runs the pipeline on a Kubeflow cluster. This is useful for running your pipeline in production on full data.

Fondant has a feature rich CLI that helps you with these steps, let's start by compiling our pipeline for the local runner:

```bash
fondant compile pipeline:my_pipeline --local
```

We call the fondant CLI to compile our pipeline, we pass a reference to our pipeline using the import_string syntax `<module>:<instance>`. We also pass the `--local` flag to indicate we want to compile our pipeline for the local runner.
Running this command will create a `docker-compose.yml` file with the compiled pipeline definition. Feel free to inspect this file but changing it is not needed.

Note that if you use a local `base_path` in your pipeline declaration that this path will be mounted in the docker containers. This means that the data will be stored locally on your machine. If you use a cloud storage path, the data will be stored in the cloud.

Now that we have compiled our pipeline, we can run it:

```bash
fondant run docker-compose.yml --local
```

You should see the image used by the component being pulled and a container being created that downloads the dataset from huggingface hub. This container will be removed after the pipeline has finished running. But the data should be stored as parquet files in the `base_path` you defined when creating the pipeline.

You can combine the compile and run commands into one command by referencing the fondant pipeline directly with the run command:

```bash
fondant run pipeline:my_pipeline --local
```

Now your pipeline will be converted into a `docker-compose.yml` file, compiled and ran in one go.


## Adding a custom component

Let's expand our pipeline by adding a custom component:

#TODO