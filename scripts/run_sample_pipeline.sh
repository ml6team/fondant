#!/bin/bash
# This script executes the sample pipeline in the example folder, checks the correct execution and
# cleans up the directory again
set -e

mkdir -p ./examples/sample_pipeline_test/.artifacts

cleanup() {
  rv=$?

  # Create a temporary directory
  artifact_directory="./examples/sample_pipeline_test/.artifacts"
  # Check if the temporary directory exists before attempting to remove it
  if [ -d "$artifact_directory" ]; then
    sudo rm -r "$artifact_directory"
    echo "Temporary directory removed"
  fi

  exit $rv
}

trap cleanup EXIT

resolved_path=$(readlink -f "examples/sample_pipeline_test/data")

poetry run fondant run local examples/sample_pipeline_test/sample_pipeline.py \
  --extra-volumes $resolved_path:/data

if [ "$(ls -A ./examples/sample_pipeline_test/.artifacts)" ]; then
    echo "Sample pipeline executed successfully."
    exit 0
fi


echo "Sample pipeline execution failed."
exit 1
