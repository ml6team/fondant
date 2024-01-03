#!/bin/bash
# This script executes the sample pipeline in the example folder, checks the correct execution and
# cleans up the directory again
set -e

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

poetry run fondant run local examples/sample_pipeline_test/sample_pipeline.py

# Expect that .artifacts was created and isn't empty
if [ -d "./examples/sample_pipeline_test/.artifacts" ]; then
    if [ "$(ls -A .artifacts)" ]; then
        echo "Sample pipeline executed successfully."
        exit 0
    fi
fi

echo "Sample pipeline execution failed."
exit 1
