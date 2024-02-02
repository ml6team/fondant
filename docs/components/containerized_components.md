# Creating containerized components

Fondant makes it easy to build data preparation pipelines leveraging reusable components. Fondant
provides a lot
of [components out of the box](https://fondant.ai/en/latest/components/hub/), but you can also
define your own containerized components.

Containerized components are useful when you want to share the components within your organization
or community.
If you don't need your component to be shareable, we recommend starting 
with a simpler [lightweight components](../components/lightweight_components.md) instead.

To make sure containerized components are reusable, they should implement a single logical data
processing
step (like captioning images or removing Personal Identifiable Information [PII] from text).
If a component grows too large, consider splitting it into multiple separate components each
tackling one logical part.

To implement a containerized component, a couple of files need to be defined:

- [Fondant component specification](#fondant-component-specification)
- [`main.py` script in a `src` folder](#mainpy-script)
- [Dockerfile](#dockerfile)
- [requirements.txt](#requirementstxt)

## Fondant component specification

Each containerized Fondant component  is defined by a specification which describes its interface. This
specification is represented by a single `fondant_component.yaml` file. See the [component
specification page](../components/component_spec.md) for info on how to write the specification for
your component.

## Main.py script

The component script should be implemented in a `main.py` script in a folder called `src`.
Refer to the [main.py script](../components/components.md) section for more info on how to implement the
script.

Note that the `main.py` script can be split up into several Python scripts in case it would become
prohibitively long. See the
[prompt based LAION retrieval component](https://github.com/ml6team/fondant/tree/main/components/retrieve_laion_by_prompt/src)
as an example: the CLIP client itself is defined in a separate script called `clip_client`,
which is then imported in the `main.py` script.

## Dockerfile

The `Dockerfile` defines how to build the component into a Docker image. An example Dockerfile is
defined below.

```bash
FROM --platform=linux/amd64 python:3.8-slim

# install requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the working directory to the component folder
WORKDIR /component/src

# Copy over src-files and spec of the component
COPY src/ .

ENTRYPOINT ["fondant", "execute", "main"]
```

## Requirements.txt

A `requirements.txt` file lists the Python dependencies of the component. Note that any Fondant
component will always have `Fondant[component]` as the minimum requirement. It's important to also 
pin the version of each dependency to make sure the component remains working as expected. Below is 
an example of a component that relies on several Python libraries such as Pillow, PyTorch and
Transformers.

```
fondant[component]
Pillow==10.0.1
torch==2.0.1
transformers==4.29.2
```

Refer to this [section](publishing_components.md) to find out how to build and publish your components to use them in 
your own pipelines.


