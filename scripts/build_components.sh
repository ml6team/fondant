#!/bin/bash
set -e

function usage {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -t, --tag <value>        Set the tag (default: latest)"
  echo "  -h, --help               Display this help message"
}

# Parse the arguments
while [[ "$#" -gt 0 ]]; do case $1 in
  -t|--tag) tag="$2"; shift;;
  -h|--help) usage; exit;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

# Check for required argument
if [ -z "${tag}" ]; then
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
  full_image_name=ghcr.io/${namespace}/${BASENAME}:${tag}
  echo "$full_image_name"

  # Update the image version in the fondant_component.yaml
  sed -i "s|^image: .*|image: $full_image_name|" fondant_component.yaml

  # Update the fondant requirement to the version being built
  sed -i "s|^fondant.*|fondant==$tag|" requirements.txt

  docker build -t "$full_image_name" \
   --label org.opencontainers.image.source=https://github.com/${namespace}/${repo} \
   .

  docker push "$full_image_name"

  popd
done