## Installing Fondant

Install Fondant by running:

```bash
pip install fondant
```

Fondant also includes extra dependencies for specific runners, storage integrations and publishing components to registries.

### Runner specific dependencies

=== "Kubeflow"

    ```bash
    pip install fondant[kfp]
    ```

=== "Vertex"

    ```python
    pip install fondant[Vertex]
    ```

=== "SageMaker"

    ```python
    pip install fondant[SageMaker]
    ```

### Storage integration dependencies

=== "Google Cloud Storage (GCS)"

    ```bash
    pip install fondant[gcp]
    ```

=== "Amazon S3"

    ```python
    pip install fondant[aws]
    ```

=== "Azure Blob Storage"

    ```python
    pip install fondant[azure]
    ```


### Publishing components dependencies

For publishing components to registries check out the [guide](../components/publishing_components.md) on publishing components to registries.

### Runner specific dependencies


## Docker installation

To execute pipelines locally, you must
have [Docker compose](https://docs.docker.com/compose/install/) and Python >=3.9
installed on your system. We only support Docker compose V2 and above. If you have an older version of
Docker compose, please upgrade to the latest version as described in the [Docker compose migration doc](https://docs.docker.com/compose/migrate/).

!!! note "IMPORTANT"
    For **Apple M1/M2 ship users**: <br>
    
    - There is no support for `linux/arm64` based images (yet). `linux/amd64` images will be used by default.

    - In Docker Desktop Dashboards’ `Settings -> Features in development`, make sure to
      uncheck `Use containerid for pulling and storing images`.

!!! note "IMPORTANT"
    For **Windows** users: <br>
    
    - We recommend installing and running docker on [WSL](https://learn.microsoft.com/en-us/windows/wsl/about). 
    Check out this [guide](https://docs.docker.com/desktop/wsl/) for more info on the installation. <br>.