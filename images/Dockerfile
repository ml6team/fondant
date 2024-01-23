ARG PYTHON_VERSION
FROM --platform=linux/amd64 python:${PYTHON_VERSION}-slim

# System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y

# Install Fondant
# This is split from other requirements to leverage caching
ARG FONDANT_VERSION=main
RUN pip3 install fondant[component,aws,azure,gcp]@git+https://github.com/ml6team/fondant@${FONDANT_VERSION}

