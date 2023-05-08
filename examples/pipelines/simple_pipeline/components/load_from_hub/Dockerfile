FROM --platform=linux/amd64 python:3.8-slim

## System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y

# install requirements
COPY requirements.txt /
COPY fondant_component.yaml /
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy over src-files of the component
COPY src /src

# Set the working directory to the source folder
WORKDIR /src

ENTRYPOINT ["python", "main.py"]