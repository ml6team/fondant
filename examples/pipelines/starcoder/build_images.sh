
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