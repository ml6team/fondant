# Start with a base Ubuntu image
FROM ubuntu:latest

# Install necessary packages
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    fuse \
    git \
    libfuse-dev \
    libssl-dev \
    pkg-config \
    wget

# Install GCSFuse
RUN echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" > /etc/apt/sources.list.d/gcsfuse.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y gcsfuse

## Install S3FS
RUN echo "deb http://deb.debian.org/debian buster-backports main" | tee /etc/apt/sources.list.d/backports.list && \
    apt-get update && \
    apt-get -t buster-backports install -y s3fs

## Install BlobFuse
RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y blobfuse

## Clean up
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the entrypoint
ENTRYPOINT ["/bin/bash"]