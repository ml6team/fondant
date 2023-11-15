#!/bin/bash
set -e

function usage {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -t,  --tag <value>                 Tag to add to image, repeatable
                                             The first tag is set in the component specifications"
  echo "  -c,  --cache <value>               Use registry caching when building the components (default:false)"
  echo "  -d,  --components-dir <value>      Directory containing components to build as subdirectories.
                                             The path should be relative to the root directory (default:components)"
  echo "  -n, --namespace <value>            The namespace for the built images, should match the github organization (default: ml6team)"
  echo "  -co, --component <value>           Specific component to build. Pass the component subdirectory name(s) to build
                                             certain component(s) or 'all' to build all components in the components
                                             directory (default: all)"
  echo "  -r,  --repo <value>                Set the repo (default: fondant)"
  echo "  -h,  --help                        Display this help message"
}

# Parse the arguments
while [[ "$#" -gt 0 ]]; do case $1 in
  -n |--namespace) namespace="$2"; shift;;
  -d |--components-dir ) components_dir="$2"; shift;;
  -r |--repo) repo="$2"; shift;;
  -t |--tag) tags+=("$2"); shift;;
  -co|--component) components+=("$2"); shift;;
  -c |--cache) caching=true;;
  -h |--help) usage; exit;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

# Check for required argument
if [ -z "${tags}" ]; then
  echo "Error: tag parameter is required"
  usage
  exit 1
fi

# Set default values for optional arguments if not passed
component="${components:-all}"
components_dir="${components_dir:-components}"
namespace="${namespace:-ml6team}"
repo="${repo:-fondant}"

# Get the component directory
scripts_dir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
# root_dir=$(dirname "$scripts_dir")
components_dir=$scripts_dir/${components_dir}


# Determine the components to build
if [[ "${component}" == "all" ]]; then
  components_to_build=("$components_dir"/*/);
else
  for component in "${components[@]}"; do
    components_to_build+=("$components_dir/${component}/")
  done
fi

# Loop through all subdirectories
for dir in "${components_to_build[@]}"; do
  pushd "$dir"
  BASENAME=${dir%/}
  BASENAME=${BASENAME##*/}

  full_image_names=()
  echo "Tagging image with following tags:"
  for tag in "${tags[@]}"; do
    full_image_name=${namespace}/${BASENAME}:${tag}
    echo "$full_image_name"
    full_image_names+=("$full_image_name")
  done

  # Prevent this from mistakenly being used below
  unset full_image_name

  echo "Updating the image version in the fondant_component.yaml with:"
  echo "${full_image_names[0]}"
  sed -i -e "s|^image: .*|image: ${full_image_names[0]}|" fondant_component.yaml

  args=()

  # Add argument for each tag
  for tag in "${full_image_names[@]}" ; do
    args+=(-t "$tag")
  done

  # Add cache arguments if caching is enabled
  if [ "$caching" = true ] ; then

    cache_name=ghcr.io/${namespace}/${BASENAME}:build-cache
    echo "Caching from/to ${cache_name}"
    args+=(--cache-to "type=registry,ref=${cache_name}")
    args+=(--cache-from "type=registry,ref=${cache_name}")
  fi

  echo "Freezing Fondant dependency version to ${tags[0]}"
  docker build --push "${args[@]}" \
   --build-arg="FONDANT_VERSION=${tags[0]}" \
   .

  popd

done
