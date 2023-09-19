# Creative common license dataset

## Overview

We present a sample pipeline that demonstrates how to effectively utilize a creative
commons image dataset within a fondant pipeline. This dataset comprises images from diverse sources
and is available in various data formats. In this illustrative example, our objective is to refine
the dataset to exclusively include PNG files.

We with the initialization of the image dataset sourced from HuggingFace.
Afterwards, we narrow down the dataset contents to exclusively encompass PNG files. Finally, we
proceed with the downloading of these carefully selected images. Accomplishing these tasks
necessitates the use of both pre-built reusable components (HuggingFace dataset loading and image
downloading) and the creation of a customized component tailored to filter images based on their
data type.

To better understand the instructions provided, we highly recommend familiarizing yourself with the
fundamental concepts of Fondant. To do so, you can take a look at
our [getting started](https://fondant.readthedocs.io/en/stable/getting_started) page.

### File structure

The directory structure is organized as follows:

```
.
├── components
│ ├── download_images
│ ├── filter_file_type
│ └── load_from_hf_hub
└── pipeline.py
```

Within this folder, you will find the following items:

- pipeline.py: This file defines the core pipeline and serves as the starting point for the pipeline
  execution.
- components: This directory contains three distinct components.

The load_from_hub and download_images components are reusable components, while the filter_file_type
component serves as an illustrative example of a custom component.

- The load_from_hf_hub component is used to initialise the dataset from huggingface hub.
- The filter_file_type component is used to filter the images for a specific file type, e.g. PNG
- The download_images component downloads each image, based on the provided image_url and stores the
  result to the dataset

## Running the sample pipeline and explore the data

Accoridingly, the getting started documenation, we can run the pipeline by using the `LocalRunner`
as follow:

```bash
fondant run pipeline --local
```

After the pipeline is succeeded you can explore the data by using the fondant data explorer:

```bash
fondant explore --base_path ./data-dir
```

### Customize the base pipeline

Customizing the pipeline is an easy process. To make adjustments, you can simply edit the
component arguments defined in `pipeline.py`. As an example, if your intention is to filter for JPEG
images, you can substitute `image/png` with `image/jpeg`.

```python
filter_mime_type = ComponentOp(
    component_dir="components/filter_file_type",
    arguments={
        "mime_type": "image/jpeg"
    },
    cache=False,
    cluster_type="default"
)
```

Expanding upon the concept of custom component implementation, you have the flexibility to create
additional custom components. For example, you can design custom components to filter out NSFW (Not
Safe For Work) content or to identify and exclude images containing watermarks.

