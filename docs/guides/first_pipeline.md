# Getting started

Check out the [installation guide](installation.md) to get started with Fondant.

For demonstration purposes, we provide sample pipelines in the Fondant GitHub repository. A great starting point is the pipeline that loads and filters creative commons images. To follow along with the upcoming instructions, you can clone the [repository](https://github.com/ml6team/fondant) and navigate to the `examples/pipelines/filter-cc-25m` folder.

Clone the Fondant GitHub repository

```
git clone https://github.com/ml6team/fondant.git
```

Make sure that Docker Compose is running, navigate to `fondant/examples/pipelines/filter-cc-25m`, and initiate the pipeline by executing:

```
fondant run local pipeline.py
```

Note: For local testing purposes, the pipeline will only download the first 10,000 images. If you want to download the full dataset, you will need to modify the component arguments in the pipeline.py file, specifically the following part:

```python
load_from_hf_hub = ComponentOp(
    component_dir="components/load_from_hf_hub",
    arguments={
        "dataset_name": "fondant-ai/fondant-cc-25m",
        "column_name_mapping": load_component_column_mapping,
        "n_rows_to_load": <HERE INSERT THE NUMBER OF IMAGES YOU WANT TO DOWNLOAD>
    },
)
```

To visually inspect the results quickly, you can use:

```
fondant explore --base_path ./data
```

### Custom pipelines

Fondant enables you to leverage existing reusable components and integrate them with custom components. To delve deeper into creating your own pipelines, please explore our [guide](guides/build_a_simple_pipeline.md). There, you will gain insights into components, various component types, and how to effectively utilise them.
