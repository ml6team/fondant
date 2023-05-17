cd components/generate_prompts/src
python main.py --metadata '{"run_id":"test_niels", "base_path":"/Users/nielsrogge/Documents/fondant_artifacts"}' --output_manifest_path /Users/nielsrogge/Documents/fondant_artifacts/manifest/generate_prompts/manifest.txt

cd ..
cd ..
cd prompt_based_laion_retrieval/src
python main.py --metadata '{"run_id":"test_niels", "base_path":"/Users/nielsrogge/Documents/fondant_artifacts"}' --output_manifest_path /Users/nielsrogge/Documents/fondant_artifacts/manifest/laion_retrieval/manifest.txt --input_manifest_path /Users/nielsrogge/Documents/fondant_artifacts/manifest/generate_prompts/manifest.txt --num_images 2 --aesthetic_score 9 --aesthetic_weight 0.5
