## Hugging Face Datasets

## Usage

Run the following command to load a Hugging Face Dataset from the [hub](https://huggingface.co/) and upload it to a Google Cloud Storage bucket:

```
python3 data_loading_pandas.py
 --extra-args '{"project_id": "soy-audio-379412"}'
 --metadata-args '{"run_id":"test","component_name":"test_component","artifact_bucket":"soy-audio-379412_kfp-artifacts"}' --output-manifest captions.text 
 --storage-args '{"cloud_env":"GCP","mount_buckets":["soy-audio-379412_kfp-artifacts"],"mount_dir":"./test_mount","mount_args":["implicit-dirs"],"mount_kwargs":null}' 
```