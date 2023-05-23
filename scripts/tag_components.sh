#!/bin/bash
set -e

function usage {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -o, --old-tag <value>    Old tag to of image to retag"
  echo "  -n, --new-tag <value>    New tag to tag image with"
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

# Get the component directory
scripts_dir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
root_dir=$(dirname "$scripts_dir")
component_dir=$root_dir/"components"

# Loop through all subdirectories
for dir in "$component_dir"/*/; do
  pushd "$dir"

  BASENAME=${dir%/}
  BASENAME=${BASENAME##*/}
  old_image_name=ghcr.io/${namespace}/${BASENAME}:${new_tag}
  new_image_name=ghcr.io/${namespace}/${BASENAME}:${new_tag}
  echo "$old_image_name"
  echo "$new_image_name"

  docker pull "$old_image_name"
  docker tag "$old_image_name" "$new_image_name"
  docker push "$new_image_name"

  popd
done
