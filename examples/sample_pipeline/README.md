# Sample pipeline

This example is a simple sample pipeline which uses two reusable components
(load_from_parquet, chunk_text), and a custom dummy component. The custom dummy component only
returns the received dataframe. 

The pipeline can be executed with the Fondant cli:

```bash
fondant run local pipeline.py --extra-volumes ./data:/data
```

> We are mounting the data folder using `--extra-volumes` to ensure that the component 
> can load the sample data and write to the folder.

The automated integration test will use the `run.sh` script. 