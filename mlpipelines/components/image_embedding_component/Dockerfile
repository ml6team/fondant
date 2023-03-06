FROM europe-west1-docker.pkg.dev/storied-landing-366912/storied-landing-366912-default-repository/mlpipelines/kubeflow/components/base_component:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV COMMIT_SHA_CLIP=d50d76daa670286dd6cacf3bcd80b5e4823fc8e1

## System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git wget -y

# Set the working directory to the source folder
WORKDIR /src

## Download model
# (Download Link:https://github.com/openai/CLIP/issues/199)
RUN wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt

# Install CLIP
RUN pip3 install git+https://github.com/openai/CLIP.git@${COMMIT_SHA_CLIP}

# Install packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY src /src

ENTRYPOINT ["python", "main.py"]