FROM --platform=linux/amd64 python:3.8-slim

## System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y

# install requirements
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the working directory to the compoent folder
WORKDIR /component/src

# Copy over src-files and spec of the component
COPY src/ .
COPY fondant_component.yaml ../

ENTRYPOINT ["python", "main.py"]