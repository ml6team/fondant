FROM python:3.8-slim

ENV DEBIAN_FRONTEND=noninteractive

## System dependencies
# Install system dependencies
RUN set -e; \
    apt-get update -y && apt-get install curl gnupg2 -y \
    lsb-release; \
    gcsFuseRepo=gcsfuse-`lsb_release -c -s`; \
    echo "deb http://packages.cloud.google.com/apt $gcsFuseRepo main" | \
    tee /etc/apt/sources.list.d/gcsfuse.list; \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key add -; \
    apt-get update; \
    apt-get install -y gcsfuse \
    && apt-get clean

ENV APP_HOME /app
WORKDIR $APP_HOME

CMD ["/bin/bash"]

