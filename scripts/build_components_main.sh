#!/bin/bash
set -e

function usage {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -t, --tag <value>        Tag to add to image, repeatable
                                   The first tag is set in the component specifications"
  echo "  -h, --help               Display this help message"
}

# Parse the arguments
while [[ "$#" -gt 0 ]]; do case $1 in
  -t|--tag) tags+=("$2"); shift;;
  -h|--help) usage; exit;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

# Check for required argument
if [ -z "${tags}" ]; then
  echo "Error: tag parameter is required"
  usage
  exit 1
fi

# Set github repo information
namespace="ml6team"
repo="fondant"

# Get the component directory
scripts_dir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
root_dir=$(dirname "$scripts_dir")
component_dir=$root_dir/"components"

# Loop through all subdirectories
for dir in "$component_dir"/*/; do
  pushd "$dir"

  BASENAME=${dir%/}
  BASENAME=${BASENAME##*/}

  echo "Tagging image with following tags:"
  for tag in "${tags[@]}"; do
    full_image_name=ghcr.io/${namespace}/${BASENAME}:${tag}
    echo "$full_image_name"
    full_image_names+=(full_image_name)
  done
  # Prevent this from mistakenly being used below
  unset full_image_name

  echo "Updating the image version in the fondant_component.yaml with:"
  echo "${full_image_names[0]}"
  sed -i "s|^image: .*|image: ${full_image_names[0]}|" fondant_component.yaml

  echo "Freezing Fondant dependency version to:"
  echo "${tags[0]}"
  sed -i "s|^fondant.*|fondant==${tags[0]}|" requirements.txt

  tag_args=()
  for tag in "${full_image_names[@]}" ; do
      tag_args+=(-t "$tag")
  done

  echo "Caching from/to:"
  cache_name=ghcr.io/${namespace}/${BASENAME}:build-cache

  docker build --push "${tag_args[@]}" \
   --cache-to type=registry,ref=${cache_name} \
   --cache-from type=registry,ref=${cache_name} \
   --label org.opencontainers.image.source=https://github.com/${namespace}/${repo} \
   .

  popd
done