FROM --platform=linux/amd64 python:3.8-slim as base

## System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y

RUN pip3 install fondant[aws,azure,gcp]@git+https://github.com/mrchtr/fondant

# install requirements
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy over src-files
COPY src/ src/

# Run component tests
FROM base as test
RUN pip3 install pytest pandas # TODO add pytest to package setup
ENV PYTHONPATH "${PYTHONPATH}:./src"
COPY tests/ tests/
RUN ["python", "-m", "pytest", "tests/"]

FROM base
# Set the working directory to the component folder
WORKDIR /src
ENTRYPOINT ["python", "main.py"]