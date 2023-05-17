cd components/generate_prompts/src
python main.py --metadata '{"run_id":"test_niels", "base_path":"/Users/nielsrogge/Documents/fondant_artifacts"}' --output_manifest_path /Users/nielsrogge/Documents/fondant_artifacts/manifest/generate_prompts/manifest.txt

cd ..
cd ..
cd prompt_based_laion_retrieval/src
python main.py --metadata '{"run_id":"test_niels", "base_path":"/Users/nielsrogge/Documents/fondant_artifacts"}' --output_manifest_path /Users/nielsrogge/Documents/fondant_artifacts/manifest/laion_retrieval/manifest.txt --input_manifest_path /Users/nielsrogge/Documents/fondant_artifacts/manifest/generate_prompts/manifest.txt --num_images 2 --aesthetic_score 9 --aesthetic_weight 0.5

cd ..
cd ..
cd download_images/src
python main.py --metadata '{"run_id":"test_niels", "base_path":"/Users/nielsrogge/Documents/fondant_artifacts"}' --output_manifest_path /Users/nielsrogge/Documents/fondant_artifacts/manifest/download_images/manifest.txt --input_manifest_path /Users/nielsrogge/Documents/fondant_artifacts/manifest/laion_retrieval/manifest.txt --timeout 10 --retries 0 --image_size 512 --resize_mode "center_crop" --resize_only_if_bigger False --min_image_size 0 --max_aspect_ratio 2.5

cd ..
cd ..
cd caption_images/src
python main.py --metadata '{"run_id":"test_niels", "base_path":"/Users/nielsrogge/Documents/fondant_artifacts"}' --output_manifest_path /Users/nielsrogge/Documents/fondant_artifacts/manifest/caption_images/manifest.txt --input_manifest_path /Users/nielsrogge/Documents/fondant_artifacts/manifest/download_images/manifest.txt --model_id "microsoft/git-base-coco" --batch_size 2 --max_new_tokens 50