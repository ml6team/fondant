# Component specification

Each Fondant component is defined by a component specification which describes its interface. 
The component specification is used for a couple of things:

- To define which input data Fondant should provide to the component, and which output data it should 
  write to storage.
- To validate compatibility with other components.
- To execute the component with the correct parameters.

The component specification should be defined by the author of the component.

## Contents

A component spec(ification) consists of the following sections:

```yaml
name: ...
description: ...
image: ...

consumes:
  ...

produces:
  ...

args:
  ...
```

### Metadata

The metadata tracks metadata about the component, such as its name, description, and the URL of the [Docker](https://www.docker.com/) image used to run it.

```yaml
name: Example component
description: This is an example component
image: example_component:latest
...
```

### Consumes & produces

The `consumes` and `produces` sections describe which data the component consumes and produces. 
The specification below for instance defines a component that creates an embedding from an 
image-caption combination.

```yaml
...
consumes:
  images:
    fields:
      data:
        type: binary
  captions:
    fields:
      text:
        type: utf8

produces:
  embeddings:
    fields:
      data:
        type: array
        items:
          type: float32
...
```

The `consumes` and `produces` sections follow the schema below:

```yaml
consumes/produces:
  <subset>:
    fields:
      <field>:
        type: <type>
    additionalFields: true
  additionalSubsets: true
```

#### Subsets

A component consumes or produces `subsets` which match the `subsets` from 
[the manifest](manifest.md). 
- Only those subsets defined in the `consumes` section of the 
component specification are read and passed to the component implementation.
- Only those subsets defined in the `produces` section of the component specification are 
  written to storage.

#### Fields

Each subset defines a list of `fields`, which again match those from the manifest.
- Only those fields defined in the `consumes` section of the component specification are read 
  and passed to the component implementation.
- Only those fields defined in the `produces` section of the component specification are written 
  to storage

Each field defines the expected data type, which should match the 
[types defined by Fondant](https://github.com/ml6team/fondant/blob/main/fondant/schema.py#L13), 
which mostly match the [Arrow data types](https://arrow.apache.org/docs/python/api/datatypes.html).

#### AdditionalSubsets & additionalFields

The schema also defines the `additionalSubsets` and `additionalFields` keywords, which can be 
used to define which additional data should be passed on from the input to the output. They both 
default to `true`, which means that by default untouched data is passed on to the next component.

- If `additionalSubsets` is `false` in the `consumes` section, all subsets not specified in the 
  component specification's `consumes` will be dropped.
- If `additionalSubsets` is `false` in the `produces` section, all subsets not specified in the 
  component specification's `produces` section will be dropped, including consumed subsets.
- If `additionalFields` is `false` for a subset in the `consumes` section, all fields not 
  specified will be dropped.
- If `additionalFields` is `false` for a subset in the `produces` section, all fields not 
  specified will be dropped, including consumed fields.

Please check the [examples](#examples) below to build a better understanding.

### Args

The `args` section describes which arguments the component takes. Each argument is defined by a 
`description` and a `type`, which should be one of the builtin Python types. Additionally, you can 
pass an optional argument in the optional `default` field of the arguments. This can be useful in case
your function has many arguments that could be set to default and that don't need to be explicitly defined when
initializing your component. 

_Note:_ default iterable arguments such as `dict` and `list` have to be passed as a string 
(e.g. `'{"foo":1, "bar":2}`, `'["foo","bar]'`)
```yaml
args:
  custom_argument:
    description: A custom argument
    type: str
  default_argument:
    description: A default argument
    type: str
    default: bar
``` 

These arguments are passed in when the component is instantiated. Notice that we are not passing the
default argument specified above. You could override the default value in the component spec by passing
it as an argument to the component.
```python
from fondant.pipeline import ComponentOp

custom_op = ComponentOp(
    component_spec_path="components/custom_component/fondant_component.yaml",
    arguments={
        "custom_argument": "foo"
    },
)
```

Afterwards, we pass all keyword arguments to the `transform()` method of the component.

```python
from fondant.component import TransformComponent

class ExampleComponent(TransformComponent):

    def transform(self, dataframe, *, custom_argument, default_argument):
        """Implement your custom logic in this single method
        
        Args:
            dataframe: A Dask dataframe containing the data
            custom_argument: An argument passed to the component
            default_argument: A default argument passed to the components
        """
```

## Examples

Each component specification defines how the input manifest will be transformed into the output 
manifest. The following examples show how the component specification works:

### Example 1: defaults

Even though only a single `subset` and `field` are defined in both `consumes` and `produces`, 
all data is passed along since `additionalSubsets` and `additionalFields` default to `true`.

<table>
<tr>
<th width="500px">Input manifest</th>
<th width="500px">Component spec</th>
<th width="500px">Output manifest</th>
</tr>
<tr>
<td>

```json
{
  "subsets": {
    "images": {
      "location": "...",
      "fields": {
        "width": {
          "type": "int32"
        },
        "height": {
          "type": "int32"
        },
        "data": {
          "type": "binary"
        }
      }
    },
    "captions": {
      "location": "...",
      "fields": {
        "data": {
          "type": "binary"
        }
      }
    }
  }
}
```

</td>
<td>


```yaml
consumes:
  images:
    fields:
      data:
        type: binary

produces:
  embeddings:
    fields:
      data:
        type: array
        items:
          type: float32
```

</td>
<td>

```json
{
  "subsets": {
    "images": {
      "location": "...",
      "fields": {
        "width": {
          "type": "int32"
        },
        "height": {
          "type": "int32"
        },
        "data": {
          "type": "binary"
        }
      }
    },
    "captions": {
      "location": "...",
      "fields": {
        "data": {
          "type": "binary"
        }
      }
    },
    "embeddings": {
      "location": "...",
      "fields": {
        "data": {
          "type": "binary"
        }
      }
    }
  }
}
```

</td>
</tr>
</table>

### Example 2: `additionalSubsets: false` in `consumes`

When changing `additionalSubsets` in `consumes` to `false`, the unused `captions` subset is 
dropped.

<table>
<tr>
<th width="500px">Input manifest</th>
<th width="500px">Component spec</th>
<th width="500px">Output manifest</th>
</tr>
<tr>
<td>

```json
{
  "subsets": {
    "images": {
      "location": "...",
      "fields": {
        "width": {
          "type": "int32"
        },
        "height": {
          "type": "int32"
        },
        "data": {
          "type": "binary"
        }
      }
    },
    "captions": {
      "location": "...",
      "fields": {
        "data": {
          "type": "binary"
        }
      }
    }
  }
}
```

</td>
<td>


```yaml
consumes:
  images:
    fields:
      data:
        type: binary
  additionalSubsets: false

produces:
  embeddings:
    fields:
      data:
        type: array
        items:
          type: float32
```

</td>
<td>

```json
{
  "subsets": {
    "images": {
      "location": "...",
      "fields": {
        "width": {
          "type": "int32"
        },
        "height": {
          "type": "int32"
        },
        "data": {
          "type": "binary"
        }
      }
    },
    "embeddings": {
      "location": "...",
      "fields": {
        "data": {
          "type": "binary"
        }
      }
    }
  }
}
```

</td>
</tr>
</table>

### Example 3: `additionalFields: false` in `consumes`

When changing `additionalFields` in the consumed images subset to `false`, the unused fields of 
the images subset are dropped as well. 

<table>
<tr>
<th width="500px">Input manifest</th>
<th width="500px">Component spec</th>
<th width="500px">Output manifest</th>
</tr>
<tr>
<td>

```json
{
  "subsets": {
    "images": {
      "location": "...",
      "fields": {
        "width": {
          "type": "int32"
        },
        "height": {
          "type": "int32"
        },
        "data": {
          "type": "binary"
        }
      }
    },
    "captions": {
      "location": "...",
      "fields": {
        "data": {
          "type": "binary"
        }
      }
    }
  }
}
```

</td>
<td>


```yaml
consumes:
  images:
    fields:
      data:
        type: binary
    additionalFields: false
  additionalSubsets: false

produces:
  embeddings:
    fields:
      data:
        type: array
        items:
          type: float32
```

</td>
<td>

```json
{
  "subsets": {
    "images": {
      "location": "...",
      "fields": {
        "data": {
          "type": "binary"
        }
      }
    },
    "embeddings": {
      "location": "...",
      "fields": {
        "data": {
          "type": "binary"
        }
      }
    }
  }
}
```

</td>
</tr>
</table>

### Example 4 `additionalSubsets: false` in `produces`

When changing `additionalSubsets` in `produces` to `false`, both the unused `captions` subset
and the consumed `images` subsets are dropped.

<table>
<tr>
<th width="500px">Input manifest</th>
<th width="500px">Component spec</th>
<th width="500px">Output manifest</th>
</tr>
<tr>
<td>

```json
{
  "subsets": {
    "images": {
      "location": "...",
      "fields": {
        "width": {
          "type": "int32"
        },
        "height": {
          "type": "int32"
        },
        "data": {
          "type": "binary"
        }
      }
    },
    "captions": {
      "location": "...",
      "fields": {
        "data": {
          "type": "binary"
        }
      }
    }
  }
}
```

</td>
<td>


```yaml
consumes:
  images:
    fields:
      data:
        type: binary

produces:
  embeddings:
    fields:
      data:
        type: array
        items:
          type: float32
  additionalSubsets: false
```

</td>
<td>

```json
{
  "subsets": {
    "embeddings": {
      "location": "...",
      "fields": {
        "data": {
          "type": "binary"
        }
      }
    }
  }
}
```

</td>
</tr>
</table>

### Example 5: overwriting subsets

Finally, when we define a subset both in `consumes` and `produces`, the produced fields 
overwrite the consumed ones. Others are passed on according to the `additionalFields` flag.

<table>
<tr>
<th width="500px">Input manifest</th>
<th width="500px">Component spec</th>
<th width="500px">Output manifest</th>
</tr>
<tr>
<td>

```json
{
  "subsets": {
    "images": {
      "location": "...",
      "fields": {
        "width": {
          "type": "int32"
        },
        "height": {
          "type": "int32"
        },
        "data": {
          "type": "binary"
        }
      }
    },
    "captions": {
      "location": "...",
      "fields": {
        "data": {
          "type": "binary"
        }
      }
    }
  }
}
```

</td>
<td>


```yaml
consumes:
  images:
    fields:
      data:
        type: binary

produces:
  images:
    fields:
      data:
        type: string
  additionalSubsets: false
```

</td>
<td>

```json
{
  "subsets": {
    "images": {
      "location": "...",
      "fields": {
        "width": {
          "type": "int32"
        },
        "height": {
          "type": "int32"
        },
        "data": {
          "type": "string"
        }
      }
    }
  }
}
```

</td>
</tr>
</table>
