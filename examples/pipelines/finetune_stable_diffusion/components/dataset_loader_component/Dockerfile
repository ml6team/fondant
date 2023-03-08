FROM europe-west1-docker.pkg.dev/storied-landing-366912/storied-landing-366912-default-repository/mlpipelines/kubeflow/components/base_component:latest

# Set the working directory to the source folder
WORKDIR /src

# Install packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy over src-files of the component
COPY src /src

ENTRYPOINT ["python", "main.py"]