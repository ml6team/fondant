#!/bin/bash

DEFAULT_REGISTRY="ghcr.io/ml6team/data_explorer"
DEFAULT_TAG="latest"
DEFAULT_PORT="8501"

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --source|-s)
    SOURCE="$2"
    shift # past argument
    shift # past value
    ;;
    --registry|-r)
    REGISTRY="$2"
    shift # past argument
    shift # past value
    ;;
    --tag|-t)
    TAG="$2"
    shift # past argument
    shift # past value
    ;;
    --port|-p)
    PORT="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    echo "Unknown option: $1"
    exit 1
    ;;
esac
done

if [ -z "$SOURCE" ]
then
    echo "Please provide a source directory with the --source or -s option."
    exit 1
fi

if [ -z "$REGISTRY" ]
then
    REGISTRY="$DEFAULT_REGISTRY"
fi

if [ -z "$TAG" ]
then
    TAG="$DEFAULT_TAG"
fi

if [ -z "$PORT" ]
then
    PORT="$DEFAULT_PORT"
fi

echo "Running image from registry: $REGISTRY with tag: $TAG on port: $PORT"
echo "Access the explorer at http://localhost:$PORT"
docker run -p "$PORT":8501 --mount type=bind,source="$SOURCE",target="/artifacts" "$REGISTRY":"$TAG"