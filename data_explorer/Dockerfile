FROM --platform=linux/amd64 python:3.8-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

## System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y

COPY ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY ./app /app

WORKDIR /app


ENTRYPOINT [ "streamlit", "run", "/app/main.py", "--server.headless", "true", \
 "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false", \
 "--server.port=8501", "--server.address=0.0.0.0", "--"]