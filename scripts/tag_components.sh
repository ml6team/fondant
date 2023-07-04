#!/bin/bash
set -e

function usage {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -o, --old-tag <value>    Old tag to of image to retag"
  echo "  -n, --new-tag <value>    New tag to tag image with"
  echo "  -h, --help               Display this help message"
  echo "  -co, --component <value>  Set the component name. Pass the component folder name to build
  a certain components or 'all' to build all components in the components directory (default: all)"
}

# Parse the arguments
while [[ "$#" -gt 0 ]]; do case $1 in
  -o|--old-tag) old_tag="$2"; shift;;
  -n|--new-tag) new_tag="$2"; shift;;
  -co|--component) component="$2"; shift;;
  -h|--help) usage; exit;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

# Check for required argument
if [ -z "${old_tag}" ] || [ -z "${new_tag}" ]; then
  echo "Error: missing parameter"
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
  if [[ "$BASENAME" == "${component}" ]] || [[ "${component}" == "all" ]]; then

    old_image_name=ghcr.io/${namespace}/${BASENAME}:${old_tag}
    new_image_name=ghcr.io/${namespace}/${BASENAME}:${new_tag}
    echo "$old_image_name"
    echo "$new_image_name"

    docker pull "$old_image_name"
    docker tag "$old_image_name" "$new_image_name"
    docker push "$new_image_name"

    popd
  fi
done
