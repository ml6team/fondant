FROM europe-west1-docker.pkg.dev/storied-landing-366912/storied-landing-366912-default-repository/mlpipelines/kubeflow/components/base_component:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV COMMIT_SHA_BLIP=3a29b7410476bf5f2ba0955827390eb6ea1f4f9d

## System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git wget -y

# Set the working directory to the source folder
WORKDIR /src

## Download blip model
RUN wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth

# Clone BLIP repo (no package version available)
RUN git clone https://github.com/salesforce/BLIP.git && \
    cd BLIP && \
    git reset --hard ${COMMIT_SHA_BLIP}

# Install packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY src /src

ENTRYPOINT ["python", "main.py"]