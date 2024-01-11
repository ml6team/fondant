#!/bin/bash
# This script executes the sample pipeline in the example folder, checks the correct execution and
# cleans up the directory again
set -e
GIT_HASH=$1


# Setup teardown
cleanup() {
  rv=$?

  # Try to remove .artifact folder
  artifact_directory="./.artifacts"

  if [ -d "$artifact_directory" ]; then
    # Directory exists, remove it
    rm -rf "$artifact_directory"
  fi

  exit $rv
}

trap cleanup EXIT

# Bind local data directory to pipeline
data_dir=$(readlink -f "data")

# Run pipeline
poetry run fondant run local pipeline.py \
  --extra-volumes $data_dir:/data --build-arg FONDANT_VERSION=$GIT_HASH
