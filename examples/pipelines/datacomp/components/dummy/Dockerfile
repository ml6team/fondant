FROM --platform=linux/amd64 python:3.8-slim

# System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y

# Install requirements
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Fondant
# This is split from other requirements to leverage caching
ARG FONDANT_VERSION=cc418db0964ee202ebf2d4fd3aaee15604604244
RUN pip3 install fondant[aws,azure,gcp]@git+https://github.com/ml6team/fondant@${FONDANT_VERSION}

# Set the working directory to the component folder
WORKDIR /component/src

# Copy over src-files
COPY src/ .

ENTRYPOINT ["python", "main.py"]