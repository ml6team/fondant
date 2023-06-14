
## How to use?

Run locally with

```bash
bash run_locally.sh
```


Run UI container with docker. If you want to keep the docker image locally, this step requires a local docker registry.
Also note that the mounted volume path referring to the local filesystem needs to changed to your folder of choice.


Setting up local registry
```
docker run -d -p 5000:5000 --restart=always --name registry registry:2
```


Building the docker image locally
```
bash build_image_local.sh
```


Setting up the data_explorer docker container with docker-compose
```
docker-compose up -f docker-compose.isolated.yaml
```


Run UI container with the simple StarCoder pipeline (in local registry)
```
docker-compose up -f docker-compose.pipeline.yaml
```
