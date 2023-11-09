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

Check out the guide on publishing components to registries [here](../components/publishing_components.md).

### Runner specific dependencies


## Docker installation

To execute pipelines locally, you must
have [Docker compose](https://docs.docker.com/compose/install/) and Python >=3.8
installed on your system.

#### TODO: Modify/extend considerations for Docker-compose/desktop

For Apple M1/M2 ship users: <br>

- Make sure that Docker uses linux/amd64 platform and not arm64. <br>
- In Docker Dashboards’ Settings<Features in development, make sure to
  uncheck `Use containerid for pulling and storing images`.

If you are creating documentation to demonstrate the installation of Fondant's extra dependencies for specific runners, you should make it clear that each runner should be installed separately. Here's an example of how you can represent it in your documentation:

```markdown
Fondant includes extra dependencies for specific runners. To install these dependencies, use the following commands:

For Vertex runner:
```bash
pip install fondant[Vertex]
```



Replace the runner name within the brackets with the specific runner you wish to install.
```

This way, users can clearly understand that each runner needs to be installed separately using the specified commands.