# Pandas

## Usage

Run the following command to upload a Pandas dataframe to a Google Cloud Storage bucket:

```
python3 data_loading_pandas.py --extra-args '{"project_id": "storied-landing-366912"}' \
 --output-manifest <local_path> \
  --metadata-args '{"run_id":"test","component_name":"test_component", \
  "artifact_bucket":"storied-landing-366912-kfp-output"}'
```