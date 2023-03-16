#!/bin/bash

source ../components.config

IMAGE_TAG="latest"

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
  basename=${dir%/}
  basename=${basename##*/}
  # Build all images or one image depending on the passed argument
  if [[ $build_dir == true && "$basename" == "$2" ]] || [[ $build_dir == false ]]; then
    image_name="components/${PWD##*/}"
    full_image_name=${PROJECT_ARTIFACT_PATH}/${image_name}:${IMAGE_TAG}
    echo $full_image_name
    #TODO: replace with docker build after deciding on container locations
    gcloud builds submit --machine-type n1-highcpu-32 . -t "$full_image_name"
  fi
  cd "$component_dir"
done
