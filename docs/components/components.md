# Components

Fondant makes it easy to build data preparation pipelines leveraging reusable components. Fondant
provides a lot of components out of the box
([overview](hub.md)), but you can also define your
own custom components.

## The anatomy of a component

A component is completely defined by its [component specification](../components/component_spec.md) and a 
docker image. The specification defines the docker image fondant should run to execute the 
component, which data it consumes and produces, and which arguments it takes.

## Component types

We can distinguish two different types of components:

- **Reusable components** can be used out of the box and can be loaded from the fondant 
  component registry
- **Custom components** are completely defined and implemented by the user

### Reusable components

Reusable components are out of the box components from the Fondant hub that you can easily add 
to your pipeline:

```python

dataset = dataset.apply(
  "reusable_component",
  arguments={
    "arg": "value"
  }
)
```

You can find an overview of the available reusable components on the
[Fondant hub](https://github.com/ml6team/fondant/tree/main/components). Check their 
documentation for information on which arguments they accept and which data they consume and 
produce.

[//]: # (TODO: Add info on "generic" components)

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
|     |- requirements.txt
|- pipeline.py
```

The `Dockerfile` is used to build the code into a docker image, which is then referred to in the 
`fondant_component.yaml`. 

```yaml title="components/custom_component/fondant_component.yaml"
name: Custom component
description: This is a custom component
image: custom_component:latest
```

You can add a custom component to your pipeline by passing in the path to the directory containing 
your `fondant_component.yaml`.

```python title="pipeline.py"

dataset = dataset.apply(
  component_dir="components/custom_component",
  arguments={
    "arg": "value"
  }
)
```

See our [best practices on creating a custom component](../components/custom_component.md).
