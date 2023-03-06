FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install curl -y

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# install requirements
RUN pip3 install --upgrade pip
COPY requirements.txt /
RUN pip3 install -r /requirements.txt

# Set the working directory to the source folder
WORKDIR /src

# Copy over src-files of the component
COPY helpers /src/helpers
