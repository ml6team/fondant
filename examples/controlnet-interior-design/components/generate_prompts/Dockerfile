FROM --platform=linux/amd64 python:3.8-slim

## System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git curl -y

# Installing the package
RUN mkdir -p /usr/local/gcloud \
&& tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
&& /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# install requirements
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy over src-files of the component
COPY src /src

# Set the working directory to the source folder
WORKDIR /src

ENTRYPOINT ["python", "main.py"]