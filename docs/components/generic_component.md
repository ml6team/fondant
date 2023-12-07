[//]: # (TODO: Update and reinclude)

# Generic components

Fondant provides a set of reusable generic components that facilitate loading and writing
datasets to/from different platforms.

We currently have components that interface with the following platforms:

- Hugging Face Hub ([Read](https://github.com/ml6team/fondant/tree/main/components/load_from_hf_hub)/[Write](https://github.com/ml6team/fondant/tree/main/components/write_to_hf_hub)).

To integrate a generic Read/Write component into your pipeline, you only need to modify the
component specification and define the custom required/optional arguments.

## Using Generic components

Each Fondant component is defined by a specification which describes its interface. This
specification is represented by a single `fondant_component.yaml` file. See the [component
specification page](../components/component_spec.md) for info on how to write the specification for your component.

### Load component

To use a Load component, you need to modify the subset of data **produced** by the component.
These subsets define the fields that will be read from the source dataset.

For example, let's consider the [`load_from_hf_hub`](<(https://github.com/ml6team/fondant/tree/main/components/load_from_hf_hub/fondant_component.yaml)>)
Suppose we are interested in reading two columns, width and height, from a given input dataset:

| width<br/>(int32) | height<br/>(int32) |
| ----------------- | ------------------ |
| Value             | Value              |

The component specification can be modified as follows

```yaml
name: Load from hub
description: Component that loads a dataset from the hub
image: fndnt/load_from_hf_hub:latest

consumes:
  images:
    fields:
      width:
        type: int32
      height:
        type: int32

args:
  dataset_name:
    description: Name of dataset on the hub
    type: str
  column_name_mapping:
    description: Mapping of the consumed hub dataset to fondant column names
    type: dict
  image_column_names:
    description:
      Optional argument, a list containing the original image column names in case the
      dataset on the hub contains them. Used to format the image from HF hub format to a byte string.
    type: list
    default: None
  n_rows_to_load:
    description: Optional argument that defines the number of rows to load. Useful for testing pipeline runs on a small scale
    type: int
    default: None
  index_column:
    description: Column to set index to in the load component, if not specified a default globally unique index will be set
    type: str
    default: None
```

Here are a few things to note:

- The original fields are mapped to a valid
  [subset](../components/component_spec.md#subsets) using the `column_name_mapping` dictionary:

```python
column_name_mapping = {
  "width":"images_width",
  "height":"images_height"
}
```

This mapping changes the names of the original dataset fields to match the component specification,
enabling their use in subsequent pipeline steps.

- The specification includes pre-defined arguments, some of which are required (e.g., `dataset_name`),
  while others are optional but necessary in certain scenarios (e.g., `image_column_names`).

### Write component

To use a Write component, you need to modify the subset of data **consumed** by the component.
These subsets define the fields that will be written in the final dataset.

For example, let's consider the dataset that was loaded by the previous component, which currently has the following schema:

| images_width<br/>(int32) | images_height<br/>(int32) |
| ------------------------ | ------------------------- |
| Value                    | Value                     |

If we want to write this dataset to a Hugging Face Hub location, we can use the [`write_to_hf_hub`](<(https://github.com/ml6team/fondant/tree/main/components/write_to_hf_hub/fondant_component.yaml)>)

```yaml
name: Write to hub
description: Component that writes a dataset to the hub
image: fndnt/write_to_hf_hub:latest

consumes:
  images:
    fields:
      width:
        type: int32
      height:
        type: int32
args:
  hf_token:
    description: The hugging face token used to write to the hub
    type: str
  username:
    description: The username under which to upload the dataset
    type: str
  dataset_name:
    description: The name of the dataset to upload
    type: str
  image_column_names:
    description: A list containing the image column names. Used to format to image to HF hub format
    type: list
    default: None
  column_name_mapping:
    description: Mapping of the consumed fondant column names to the written hub column names
    type: dict
    default: None
```

Note that the `column_name_mapping` is optional and can be used to change the name of the columns
before writing them to their final destination. For example, if you want to have the same column names as
the original dataset, you could set the `column_name_mapping` argument as follows

```python
column_name_mapping = {
  "images_width":"width",
  "images_height":"height"
}
```

For a practical example of using and adapting load/write components, refer to the
[stable_diffusion_finetuning](https://github.com/ml6team/fondant/blob/main/examples/pipelines/finetune_stable_diffusion/pipeline.py) example.

Feel free to explore the Fondant documentation for more information on these components and their usage.
