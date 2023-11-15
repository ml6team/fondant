fondant execute main --metadata '{"base_path": "s3://s3-fondant-artifacts", "pipeline_name": "cool-pipeline", "run_id": "cool-pipeline-20231113135733", "component_id": "load_from_hub", "cache_key": "9b6b2cf8be0b5acd235d7ca7d0ef9f36"}' --output_manifest_path s3://s3-fondant-artifacts/cool-pipeline/cool-pipeline-20231113135733/load_from_hub/manifest.json --dataset_name 'fondant-ai/fondant-cc-25m' --column_name_mapping '{"alt_text": "images_alt+text", "image_url": "images_url", "license_location": "images_license+location", "license_type": "images_license+type", "webpage_url": "images_webpage+url", "surt_url": "images_surt+url", "top_level_domain": "images_top+level+domain"}' --n_rows_to_load '1000' --input_partition_rows '100' --cache 'True' --cluster_type 'default' --component_spec '{"name": "Load from hub", "description": "Component that loads a dataset from the hub", "image": "281086077386.dkr.ecr.us-east-1.amazonaws.com/load_from_hf_hub:main", "produces": {"images": {"fields": {"alt+text": {"type": "string"}, "url": {"type": "string"}, "license+location": {"type": "string"}, "license+type": {"type": "string"}, "webpage+url": {"type": "string"}, "surt+url": {"type": "string"}, "top+level+domain": {"type": "string"}}}}, "args": {"dataset_name": {"description": "Name of dataset on the hub", "type": "str"}, "column_name_mapping": {"description": "Mapping of the consumed hub dataset to fondant column names", "type": "dict", "default": {}}, "image_column_names": {"description": "Optional argument, a list containing the original image column names in case the dataset on the hub contains them. Used to format the image from HF hub format to a byte string.", "type": "list", "default": []}, "n_rows_to_load": {"description": "Optional argument that defines the number of rows to load. Useful for testing pipeline runs on a small scale", "type": "int", "default": "None"}, "index_column": {"description": "Column to set index to in the load component, if not specified a default globally unique index will be set", "type": "str", "default": "None"}}}'