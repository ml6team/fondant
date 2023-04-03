# Pandas

## Usage

Run the following command to upload a Pandas dataframe to a Google Cloud Storage bucket:

```
--args '{"project_id": "soy-audio-379412", "run_id":"test","component_name":"test_component", \
  "artifact_bucket":"soy-audio-379412_kfp-artifacts"}' \
--output-manifest <local_path>
```