FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# install requirements
RUN pip3 install --upgrade pip

# Install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install curl -y

# Install gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Install local sources
RUN mkdir temp_src
COPY . temp_src
RUN pip3 install ./temp_src
RUN rm -rf temp_src

ENTRYPOINT ["echo", "This base component should not be called directly.", \
            "Received arguments:"]