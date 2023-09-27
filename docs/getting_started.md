# Getting started

!!! note

    To execute the pipeline locally, you must have docker compose, Python >=3.8 and Git 
    installed on your system.

!!! note

    For Apple M1/M2 ship users: - Make sure that Docker uses linux/amd64 platform and not 
    arm64. - In Docker Dashboards’ Settings<Features in development, make sure to uncheck Use containerid for pulling and storing images.

For demonstration purposes, we provide sample pipelines in the Fondant GitHub repository. A great starting point is the pipeline that loads and filters creative commons images. To follow along with the upcoming instructions, you can clone the [repository](https://github.com/ml6team/fondant) and navigate to the `examples/pipelines/filter-cc-25m` folder.

This pipeline loads an image dataset and reduces the dataset to png files. For more details on how you can build this pipeline from scratch, check out our [guide](/docs/guides/build_a_simple_pipeline.md). 

Install Fondant by running:
```
pip install fondant
```

Clone the Fondant GitHub repository
```
git clone https://github.com/ml6team/fondant.git
```
Make sure that Docker Compose is running, navigate to `fondant/examples/pipelines/filter-cc-25m`, and initiate the pipeline by executing:
```
fondant run pipeline --local
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
Fondant enables you to leverage existing reusable components and integrate them with custom components. To delve deeper into creating your own pipelines, please explore our [guide](/docs/guides/build_a_simple_pipeline.md). There, you will gain insights into components, various component types, and how to effectively utilise them.