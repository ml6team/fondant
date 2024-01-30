# {{ name }}

<a id="{{ component_id }}#description"></a>
## Description
{{ description }}

<a id="{{ component_id }}#inputs_outputs"></a>
## Inputs / outputs 

<a id="{{ component_id }}#consumes"></a>
### Consumes 
{% if consumes %}
**This component consumes:**

{% for field_name, field in consumes.items() %}
- {{ field.name }}: {{ field.type.value }}
{% endfor %}
{% endif %}

{% if is_consumes_generic %}
**This component can consume additional fields**
- <field_name>: <dataset_field_name>
This defines a mapping to update the fields consumed by the operation as defined in the component spec.
The keys are the names of the fields to be received by the component, while the values are 
the name of the field to map from the input dataset

See the usage example below on how to define a field name for additional fields.

{% endif %}

{% if not is_consumes_generic and not consumes%}
**This component does not consume data.**
{% endif %}


<a id="{{ component_id }}#produces"></a>  
### Produces 
{% if produces %}
**This component produces:**

{% for field_name, field in produces.items() %}
- {{ field.name }}: {{ field.type.value }}
{% endfor %}
{% endif %}

{% if is_produces_generic %}
**This component can produce additional fields**
- <field_name>: <field_schema>
This defines a mapping to update the fields produced by the operation as defined in the component spec.
The keys are the names of the fields to be produced by the component, while the values are 
the type of the field that should be used to write the output dataset.
{% endif %}

{% if not is_produces_generic and not produces%}
**This component does not produce data.**
{% endif %}

<a id="{{ component_id }}#arguments"></a>
## Arguments

{% if arguments %}
The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
{% for argument in arguments %}
| {{ argument.name }} | {{ argument.type.__name__ }} | {{ argument.description.replace("\n", "") }} | {{ argument.default or "/" }} |
{% endfor %}
{% else %}
This component takes no arguments.
{% endif %}

<a id="{{ component_id }}#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

{% if "Data loading" in tags %}
dataset = pipeline.read(
{% else %}
dataset = pipeline.read(...)

{% if "Data writing" not in tags %}
dataset = dataset.apply(
{% else %}
dataset = dataset.apply(...)

dataset.write(
{% endif %}
{% endif %}
    "{{ id }}",
    arguments={
        # Add arguments
{% for argument in arguments %}
{% if argument.default %}
        # "{{ argument.name }}": {{ '\"' + argument.default + '\"' if argument.default is string else argument.default }},
{% else %}
        # "{{ argument.name }}": {{ argument.type() }},
{% endif %}
{% endfor %}
    },
{% if is_consumes_generic %}
    consumes={
         <field_name>: <dataset_field_name>,
         ..., # Add fields
     },
{% endif %}
{% if is_produces_generic %}
    produces={
         <field_name>: <field_schema>,
         ..., # Add fields
    },
{% endif %}
)
```

{% if tests %}
<a id="{{ component_id }}#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
{% endif %}