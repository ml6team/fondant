#!/bin/bash

function usage {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -c, --component <value>  Set the component name. Pass the component folder name to build a certain components or 'all' to build all components in the current directory (required)"
  echo "  -n, --namespace <value>  Set the namespace (default: ml6team)"
  echo "  -r, --repo <value>       Set the repo (default: fondant)"
  echo "  -t, --tag <value>        Set the tag (default: latest)"
  echo "  -h, --help               Display this help message"
}

# Parse the arguments
while [[ "$#" -gt 0 ]]; do case $1 in
  -n|--namespace) namespace="$2"; shift;;
  -r|--repo) repo="$2"; shift;;
  -t|--tag) tag="$2"; shift;;
  -c|--component) component="$2"; shift;;
  -h|--help) usage; exit;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

# Check for required argument
if [ -z "${component}" ]; then
  echo "Error: component parameter is required"
  usage
  exit 1
fi

# Set default values for optional arguments if not passed
[ -n "${namespace-}" ] || namespace="ml6team"
[ -n "${repo-}" ] || repo="fondant"
[ -n "${tag-}" ] || tag="latest"

# Get the component directory
component_dir=$(pwd)/"components"

# Loop through all subdirectories
for dir in $component_dir/*/; do
  cd "$dir"
  BASENAME=${dir%/}
  BASENAME=${BASENAME##*/}
  # Build all images or one image depending on the passed argument
  if [[ "$BASENAME" == "${component}" ]] || [[ "${component}" == "all" ]]; then
    full_image_name=ghcr.io/${namespace}/${BASENAME}:${tag}
    echo $full_image_name
    docker build -t "$full_image_name" \
     --build-arg COMMIT_SHA=$(git rev-parse HEAD) \
     --build-arg GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD) \
     --build-arg BUILD_TIMESTAMP=$(date '+%F_%H:%M:%S') \
     --label org.opencontainers.image.source=https://github.com/${namespace}/${repo} \
     --platform=linux/arm64 \
     .
    docker push "$full_image_name"
  fi
  cd "$component_dir"
done
