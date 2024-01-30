#!/bin/bash
set -e

function usage {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -t, --tag <value>        Tag to add to image, repeatable"
  echo "  -h, --help               Display this help message"
}

# Parse the arguments
while [[ "$#" -gt 0 ]]; do case $1 in
  -t |--tag) tags+=("$2"); shift;;
  -h|--help) usage; exit;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

# Check for required argument
if [ -z "${tags}" ]; then
  echo "Error: tag parameter is required"
  usage
  exit 1
fi

# Supported Python versions
python_versions=("3.8" "3.9" "3.10" "3.11")


for python_version in "${python_versions[@]}"; do
    BASENAME=fondant

    image_tags=()
    echo "Tagging image with following tags:"
    for tag in "${tags[@]}"; do
      image_tags+=("${tag}-py${python_version}")
    done

    full_image_names=()

    # create repo if not exists
    aws ecr-public describe-repositories --region us-east-1 --repository-names ${BASENAME} || aws ecr-public create-repository --region us-east-1 --repository-name ${BASENAME}

    for image_tag in "${image_tags[@]}"; do
      full_image_names+=("public.ecr.aws/fndnt/${BASENAME}:${image_tag}")
      full_image_names+=("fndnt/${BASENAME}:${image_tag}")
    done

    # Add argument for each tag
    for image_name in "${full_image_names[@]}" ; do
      args+=(-t "$image_name")
    done

    for element in "${args[@]}"; do
      echo "$element"
    done

    # Build docker images and push to docker hub
    docker build --push "${args[@]}" \
    --build-arg="PYTHON_VERSION=${python_version}" \
    --build-arg="FONDANT_VERSION=${tag}" \
    -f "images/Dockerfile" \
    .
done
