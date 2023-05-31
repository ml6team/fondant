export BASE_PATH="$PWD/fondant_artifacts_starcoder"
export METADATA="{'run_id':'test-demo', 'base_path':'${BASE_PATH}'}"

# Using sed to replace single quotes with double quotes
METADATA=$(echo "$METADATA" | sed "s/'/\"/g")

mkdir ${BASE_PATH}

cd components/load_from_hub_stack/src
python main.py \
 --metadata "$METADATA" \
 --output_manifest_path ${BASE_PATH}/manifest/load_from_hub_stack/manifest.txt \
 --dataset_name "ml6team/the-stack-smol-python"

cd ../..
cd filter_metadata/src
python main.py \
 --metadata "$METADATA" \
 --input_manifest_path  ${BASE_PATH}/manifest/load_from_hub_stack/manifest.txt \
 --output_manifest_path ${BASE_PATH}/manifest/filter_metadata/manifest.txt \
 --avg_line_length_threshold 10 \
 --max_line_length_threshold 100 \
 --alphanum_fraction_threshold 0.25

cd ../..
cd filter_comments/src
python main.py \
 --metadata "$METADATA" \
 --input_manifest_path  ${BASE_PATH}/manifest/filter_metadata/manifest.txt \
 --output_manifest_path ${BASE_PATH}/manifest/filter_comments/manifest.txt \
 --min_comments_ratio 0.1 \
 --max_comments_ratio 0.9

cd ../..
cd pii_redaction/src
python main.py \
 --metadata "$METADATA" \
 --input_manifest_path  ${BASE_PATH}/manifest/filter_comments/manifest.txt \
 --output_manifest_path ${BASE_PATH}/manifest/pii_redaction/manifest.txt