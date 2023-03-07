# Single node data loading

This folder contains an example script showcasing data loading with Express. Make sure to install express with
the `setup.py develop` command.

# Usage

Run with the following commad:

```
python3 data_loading.py --extra-args '{"project_id": "storied-landing-366912"}' \
 --output-manifest <local_path> \
  --metadata-args '{"run_id":"test","component_name":"test_component", \
  "artifact_bucket":"storied-landing-366912-kfp-output"}
```