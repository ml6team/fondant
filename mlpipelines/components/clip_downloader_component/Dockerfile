FROM europe-west1-docker.pkg.dev/storied-landing-366912/storied-landing-366912-default-repository/mlpipelines/kubeflow/components/base_component:latest

ARG DEBIAN_FRONTEND=noninteractive

# add required packages from universe repository
# Install opencv dependices
RUN apt-get update \
    && apt-get install ffmpeg libsm6 libxext6 libopencv-dev -y

# Set the working directory to the source folder
WORKDIR /src

# Copy over src-files of the component
COPY requirements.txt .

# Install packages
RUN pip3 install -r requirements.txt

COPY src /src

ENTRYPOINT ["python", "main.py"]