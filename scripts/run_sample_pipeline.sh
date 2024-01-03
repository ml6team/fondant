#!/bin/bash
# This script executes the sample pipeline in the example folder, checks the correct execution and
# cleans up the directory again

cleanup() {
  # Create a temporary directory
  artifact_directory="./examples/sample_pipeline_test/.artifacts"
  # Check if the temporary directory exists before attempting to remove it
  if [ -d "$artifact_directory" ]; then
    rm -r "$artifact_directory"
    echo "Temporary directory removed"
  fi
}

trap cleanup EXIT

echo "Temporary directory created: $artifact_directory"
fondant run local examples/sample_pipeline.py