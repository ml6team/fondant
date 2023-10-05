# Components

Fondant makes it easy to build data preparation pipelines leveraging reusable components. Fondant
provides a lot of components out of the box
([overview](https://github.com/ml6team/fondant/tree/main/components)), but you can also define your
own custom components.

## The anatomy of a component

A component is completely defined by its [component specification](../components/component_spec.md) and a 
docker image. The specification defines the docker image fondant should run to execute the 
component, which data it consumes and produces, and which arguments it takes.

## Component types

We can distinguish three different types of components:

- **Reusable components** can be used out of the box and can be loaded from the fondant 
  component registry
- **Custom components** are completely defined and implemented by the user
- **Generic components** leverage a reusable implementation, but require a custom component 
  specification

### Reusable components

Reusable components are completely defined and implemented by fondant. You can easily add them 
to your pipeline by creating an operation using `ComponentOp.from_registry()`.

```python
from fondant.pipeline import ComponentOp

component_op = ComponentOp.from_registry(
  name="reusable_component",
  arguments={
    "arg": "value"
  }
)
```

??? "fondant.pipeline.ComponentOp.from_registry"

    ::: fondant.pipeline.ComponentOp.from_registry
        handler: python
        options:
          show_source: false

You can find an overview of the reusable components offered by fondant 
[here](https://github.com/ml6team/fondant/tree/main/components). Check their 
`fondant_component.yaml` file for information on which arguments they accept and which data they 
consume and produce.

### Custom components

To define your own custom component, you can build your code into a docker image and write an 
accompanying component specification that refers to it.

A typical file structure for a custom component looks like this:
```
|- components
|  |- custom_component
|     |- src
|     |  |- main.py
|     |- Dockerfile
|     |- fondant_component.yaml
|- pipeline.py
```

The `Dockerfile` is used to build the code into a docker image, which is then referred to in the 
`fondant_component.yaml`. 

```yaml title="components/custom_component/fondant_component.yaml"
name: Custom component
description: This is a custom component
image: custom_component:latest
```

You can add a custom component to your pipeline by creating a `ComponentOp` and passing in the path 
to the directory containing your `fondant_component.yaml`.

```python title="pipeline.py"
from fondant.pipeline import ComponentOp

component_op = ComponentOp(
  component_dir="components/custom_component",
  arguments={
    "arg": "value"
  }
)
```

??? "fondant.pipeline.ComponentOp"

    ::: fondant.pipeline.ComponentOp
        handler: python
        options:
          members: []
          show_source: false

See our [best practices on creating a custom component](../components/custom_component.md).

### Generic components

A generic component is a component leveraging a reusable docker image, but requiring a custom 
`fondant_component.yaml` specification.

Since a generic component only requires a custom `fondant_component.yaml`, its file structure 
looks like this:
```
|- components
|  |- generic_component
|     |- fondant_component.yaml
|- pipeline.py
```

The `fondant_component.yaml` refers to the reusable image it leverages:

```yaml title="components/generic_component/fondant_component.yaml"
name: Generic component
description: This is a generic component
image: reusable_component:latest
```

You can add a generic component to your pipeline by creating a `ComponentOp` and passing in the path
to the directory containing your custom `fondant_component.yaml`.

```python title="pipeline.py"
from fondant.pipeline import ComponentOp

component_op = ComponentOp(
  component_dir="components/generic_component",
  arguments={
    "arg": "value"
  }
)
```

??? "fondant.pipeline.ComponentOp"

    ::: fondant.pipeline.ComponentOp
        handler: python
        options:
          members: []
          show_source: false

An example of a generic component is the 
[`load_from_hf_hub`](https://github.com/ml6team/fondant/tree/main/components/load_from_hf_hub) 
components. It can read any dataset from the HuggingFace hub, but it requires the user to define 
the schema of the produced dataset in a custom `fondant_component.yaml` specification.
