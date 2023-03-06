FROM europe-west1-docker.pkg.dev/storied-landing-366912/storied-landing-366912-default-repository/mlpipelines/kubeflow/components/base_component:latest

ARG DEBIAN_FRONTEND=noninteractive

## System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git curl wget -y

# Set the working directory to the source folder
WORKDIR /src

# Install packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt
# Specifc to A100 GPU https://github.com/pytorch/pytorch/issues/57922
# (installations included here due to error in pylint when trying to find links from pytorch.org)
RUN pip3 install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

COPY src /src

ENTRYPOINT ["python", "main.py"]