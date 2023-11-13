## Installing Fondant

Install Fondant by running:

```bash
pip install fondant
```

Fondant also includes extra dependencies for specific runners, storage integrations and publishing components to registries.

### Runner specific dependencies

For Kubeflow runner:
```bash
pip install fondant[kfp]
```

For SageMaker runner:
```bash
pip install fondant[SageMaker]
```

For Vertex runner:
```bash
pip install fondant[Vertex]
```

### Storage integration dependencies

For google cloud storage (GCS):
```bash
pip install fondant[gcp]
```

For s3 storage:
```bash
pip install fondant[aws]
```

For Azure storage:
```bash
pip install fondant[azure]
```

### Publishing components dependencies

For publishing components to registries:
```bash
pip install fondant[docker]
```

Check out the [guide](../components/publishing_components.md) on publishing components to registries.

### Runner specific dependencies


## Docker installation

To execute pipelines locally, you must
have [Docker compose](https://docs.docker.com/compose/install/) and Python >=3.8
installed on your system. We only support Docker compose V2 and above. If you have an older version of
Docker compose, please upgrade to the latest version as described in the [Docker compose migration doc](https://docs.docker.com/compose/migrate/).

!!! note "IMPORTANT"
    For **Apple M1/M2 ship users**: <br>
    
    - Make sure that Docker uses linux/amd64 platform and not arm64. <br>

        ```bash
        export DOCKER_DEFAULT_PLATFORM=linux/amd64
        ```

    - In Docker Desktop Dashboards’ `Settings -> Features in development`, make sure to
      uncheck `Use containerid for pulling and storing images`.

!!! note "IMPORTANT"
    For **Windows** users: <br>
    
    - We recommend installing and running docker on [WSL](https://learn.microsoft.com/en-us/windows/wsl/about). 
    Check out this [guide](https://docs.docker.com/desktop/wsl/) for more info on the installation. <br>.