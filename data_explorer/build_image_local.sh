#!/bin/bash

# Build the Docker image
docker build -t data_explorer .

# Tag the image for the local registry
docker tag data_explorer localhost:5000/data_explorer:latest

# Push the image to the local registry
docker push localhost:5000/data_explorer:latest