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
  echo "  -r,  --registry <value>            The docker registry prefix to use (default: null for DockerHub)"
  echo "  -n,  --namespace <value>           The DockerHub namespace for the built images (default: fndnt)"
  echo "  -co, --component <value>           Specific component to build. Pass the component subdirectory name(s) to build
                                             certain component(s) or 'all' to build all components in the components
                                             directory (default: all)"
  echo "  -r,  --repo <value>                Set the repo (default: ml6team/fondant)"
  echo "  -h,  --help                        Display this help message"
}

# Parse the arguments
while [[ "$#" -gt 0 ]]; do case $1 in
  -r |--registry) registry="$2"; shift;;
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
namespace="${namespace:-fndnt}"
repo="${repo:-ml6team/fondant}"

# Get the component directory
scripts_dir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
root_dir=$(dirname "$scripts_dir")
components_dir=$root_dir/${components_dir}

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
    if [ -n "${registry}" ] ; then
      full_image_name=${registry}/${full_image_name}
    fi
    echo "$full_image_name"
    full_image_names+=("$full_image_name")
  done

  echo "Updating the image version in the fondant_component.yaml with:"
  echo "${full_image_names[0]}"
  sed -i -e "s|^image: .*|image: ${full_image_names[0]}|" fondant_component.yaml

  # create repo if not exists
  aws ecr-public describe-repositories --region us-east-1 --repository-names ${BASENAME} || aws ecr-public create-repository --region us-east-1 --repository-name ${BASENAME}
  full_image_names+=("public.ecr.aws/fndnt/${BASENAME}:${tag}")

  args=()

  # Add argument for each tag
  for tag in "${full_image_names[@]}" ; do
    args+=(-t "$tag")
  done

  # Add cache arguments if caching is enabled
  if [ "$caching" = true ] ; then

    cache_name=${namespace}/${BASENAME}:build-cache
    if [ -n "${registry}" ] ; then
      cache_name=${registry}/${cache_name}
    fi
    echo "Caching from/to ${cache_name}"
    args+=(--cache-to "type=registry,ref=${cache_name}")
    args+=(--cache-from "type=registry,ref=${cache_name}")
  fi


  echo "Freezing Fondant dependency version to ${tags[0]}"
  docker build --push "${args[@]}" \
   --build-arg="FONDANT_VERSION=${tags[0]}" \
   --label org.opencontainers.image.source=https://github.com/${repo}/components/{BASENAME} \
   .

  docker pushrm ${full_image_name} | echo "
  README was not pushed.

  \`docker pushrm\` might not be installed.

  To install, run:
  \`wget https://github.com/christian-korneck/docker-pushrm/releases/download/v1.9.0/docker-pushrm_linux_amd64 -O /usr/libexec/docker/cli-plugins/docker-pushrm\`
  \`chmod +x /usr/libexec/docker/cli-plugins/docker-pushrm\`
  And validate by running:
  \`docker pushrm --help\`
  "

  popd

done
