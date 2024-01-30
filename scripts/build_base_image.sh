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
  -t|--tag) tags+=("$2"); shift;;
  -h|--help) usage; exit;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

# Supported Python versions
python_versions=("3.8" "3.9" "3.10" "3.11")

for tag in "${tags[@]}"; do
  for python_version in "${python_versions[@]}"; do
    BASENAME=fondant
    IMAGE_TAG=${tag}-py${python_version}
    full_image_names=()

    # create repo if not exists
    aws ecr-public describe-repositories --region us-east-1 --repository-names ${BASENAME} || aws ecr-public create-repository --region us-east-1 --repository-name ${BASENAME}
    full_image_names+=("public.ecr.aws/fndnt/${BASENAME}:${IMAGE_TAG}")
    full_image_names+=("fndnt/${BASENAME}:${IMAGE_TAG}")

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
    --build-arg="FONDANT_VERSION=${tags[0]}" \
    -f "images/Dockerfile" \
    .
  done
done
