# {{ name }}

## Description
{{ description }}

## Inputs / outputs

### Consumes
{% if consumes %}
**This component consumes:**

{% for field_name, field in consumes.items() %}
- {{ field.name }}: {{ field.type.value }}
{% endfor %}
{% else %}
_**This component does not consume specific data.**_
{% endif %}

{% if is_consumes_generic %}
**This component consumes generic data**
- <field_name>: <mapped_field_name>
{% endif %}


### Produces

{% if produces %}
**This component produces:**

{% for field_name, field in produces.items() %}
- {{ field.name }}: {{ field.type.value }}
{% endfor %}
{% else %}
_**This component does not produce specific data.**_
{% endif %}

{% if is_produces_generic %}
**This component produces generic data**
- <field_name>: <field_schema>
{% endif %}

## Arguments

{% if arguments %}
The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
{% for argument in arguments %}
| {{ argument.name }} | {{ argument.type }} | {{ argument.description.replace("\n", "") }} | {{ argument.default or "/" }} |
{% endfor %}
{% else %}
This component takes no arguments.
{% endif %}

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
        # "{{ argument.name }}": {{ (argument.type|eval)() }},
{% endif %}
{% endfor %}
    },
{% if is_consumes_generic %}
    consumes={
         <field_name>: <mapped_field_name>,
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
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
{% endif %}
