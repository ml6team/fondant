## Installing Fondant

Install Fondant by running:

```
pip install fondant
```

## Docker installation

To execute pipelines locally, you must
have [Docker compose](https://docs.docker.com/compose/install/) and Python >=3.8
installed on your system.

#### TODO: Modify/extend considerations for Docker-compose/desktop

For Apple M1/M2 ship users: <br>

- Make sure that Docker uses linux/amd64 platform and not arm64. <br>
- In Docker Dashboards’ Settings<Features in development, make sure to
  uncheck `Use containerid for pulling and storing images`.
