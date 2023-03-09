# Single node data loading

This folder contains an example script showcasing data loading with Express on a single node. Make sure to install express
with the `poetry install` command from the root of the repository.

# Usage

## Pandas

Run with the following command:

```
python3 data_loading_pandas.py --extra-args '{"project_id": "storied-landing-366912"}' \
 --output-manifest <local_path> \
  --metadata-args '{"run_id":"test","component_name":"test_component", \
  "artifact_bucket":"storied-landing-366912-kfp-output"}'
```

## Hugging Face Datasets

Run the following command to load a Hugging Face Dataset from the hub and upload it to a Google Cloud Storage bucket:

```
python3 data_loading_hf_datasets.py --extra-args '{"project_id": "soy-audio-379412"}' \
 --output-manifest <local_path> \
  --metadata-args '{"run_id":"test","component_name":"test_component", \
  "artifact_bucket":"soy-audio-379412_kfp-artifacts"}'
```