#!/bin/bash

source ../docker_build.config

# Get the flag, if it was passed
if [[ "$1" == "--build-dir" ]]; then
  build_dir=true
else
  build_dir=false
fi

# Get the component directory
component_dir=$(pwd)/"components"

# Loop through all subdirectories
for dir in $component_dir/*/; do
  cd "$dir"
  BASENAME=${dir%/}
  BASENAME=${BASENAME##*/}
  # Build all images or one image depending on the passed argument
  if [[ $build_dir == true && "$BASENAME" == "$2" ]] || [[ $build_dir == false ]]; then
    full_image_name=ghcr.io/${NAMESPACE}/${BASENAME}:${IMAGE_TAG}
    echo $full_image_name
    docker build -t "$full_image_name" \
     --build-arg COMMIT_SHA=$(git rev-parse HEAD) \
     --build-arg GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD) \
     --build-arg BUILD_TIMESTAMP=$(date '+%F_%H:%M:%S') \
     --label org.opencontainers.image.source=https://github.com/${NAMESPACE}/${REPO_NAME} \
     .
    docker push "$full_image_name"
  fi
  cd "$component_dir"
done
