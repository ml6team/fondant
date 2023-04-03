This component can be run as follows:

```
python3 main.py 
--args '{"dataset_name": "lambdalabs/pokemon-blip-captions", "run_id":"test",
"component_name":"test_component","artifact_bucket":"artifact_name"}' \
--output-manifest <local_path> 
```

To build the Docker image, simply run:

```
sh build_image.sh
```