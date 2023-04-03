## Hugging Face Datasets

## Usage

Run the following command to load a Hugging Face Dataset from the [hub](https://huggingface.co/) and upload it to a Google Cloud Storage bucket:

```
python3 data_loading_hf_datasets.py \
--args '{"project_id": "soy-audio-379412", "run_id":"test","component_name":"test_component", \
  "artifact_bucket":"soy-audio-379412_kfp-artifacts"}' \
--output-manifest <local_path>
```