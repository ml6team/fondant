# Getting started

Have a look at this page to learn how to run your first Fondant pipeline. It provides instructions for installing, executing a sample pipeline, and visually exploring the pipeline results using Fondant on your local machine.

## Prerequisite
In this example, we will utilise Fondant's LocalRunner, which leverages docker compose for the pipeline execution. Therefore, it's important to ensure that docker compose is correctly installed.

## Installation
We suggest that you use a virtual environment for your project. Fondant supports Python >=3.8.
To install Fondant via Pip, run:

```
pip install fondant[pipelines]
```

You can validate the installation of fondant by running its root CLI command:

```
fondant --help
```

## Demo
For demonstration purposes, we provide sample pipelines in the Fondant GitHub repository. A great starting point is the pipeline that loads and filters creative commons images. To follow along with the upcoming instructions, you can clone the [repository](https://github.com/ml6team/fondant) and navigate to the `examples/pipelines/filter-cc-25m` folder.

This pipeline loads an image dataset and reduces the dataset to png files. For more details on how you can build this pipeline from scratch, check out our [Tutorial](tuturials/tutorial_1.md). 

## Running the sample pipeline and explore the data
After navigating to the pipeline directory, we can run the pipeline by using the LocalRunner as follow:
```
fondant run pipeline --local
```

The sample pipeline will run and execute three steps, which you can monitor in the logs. It will load data from HuggingFace, filter out images, and then download those images. The pipeline results will be saved to parquet files. If you wish to visually explore the results, you can use the data explorer.
The following command will start the data explorer:
```
fondant explore --base_path ./data
```

### Custom pipelines
Fondant enables you to leverage existing reusable components and integrate them with custom components. To delve deeper into creating your own pipelines, please explore our [Tutorial](tuturials/tutorial_1.md). There, you will gain insights into components, various component types, and how to effectively utilise them.
