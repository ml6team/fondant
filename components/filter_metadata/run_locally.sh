export BASE_PATH="gs://boreal-array-387713_kfp-artifacts/custom_artifact"
export METADATA="{\"run_id\":\"dummy\", \"base_path\":\"$BASE_PATH\"}"

pushd src
python main.py \
 --metadata "$METADATA" \
 --input_manifest_path  ../../comments_filtering/.output_manifest.json \
 --output_manifest_path ../.output_manifest.json
 # TODO: Add arguments
 popd
