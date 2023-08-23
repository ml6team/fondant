FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

## System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y

# install requirements
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install -U openmim
RUN mim install mmengine
RUN mim install mmcv
RUN mim install mmdet
RUN mim install mmocr

# Install Fondant
# This is split from other requirements to leverage caching
ARG FONDANT_VERSION=main
RUN pip3 install fondant[aws,azure,gcp]@git+https://github.com/ml6team/fondant@${FONDANT_VERSION}

# Set the working directory to the component folder
WORKDIR /component/src

# Copy over src-files
COPY src/ .

ENTRYPOINT ["fondant", "execute", "main"]