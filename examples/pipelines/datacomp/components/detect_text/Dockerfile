FROM --platform=linux/amd64 pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive

## System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y && \
    apt-get install gcc -y && \
    apt-get install g++ -y && \
    apt-get install ffmpeg libsm6 libxext6  -y

# install requirements
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# Install Fondant
# This is split from other requirements to leverage caching
ARG FONDANT_VERSION=main
RUN pip3 install fondant[aws,azure,gcp]@git+https://github.com/ml6team/fondant@${FONDANT_VERSION}

# Set the working directory to the component folder
WORKDIR /component/src

# Copy over src-files
COPY src/ .

RUN sh ./compile.sh

ENTRYPOINT ["fondant", "execute", "main"]