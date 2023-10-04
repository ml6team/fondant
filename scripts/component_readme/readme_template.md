# {{ name }}

### Description
{{ description }}

### Inputs/Outputs

**The component comsumes:**
{% for subset_name, subset in consumes.items() %}
- {{ subset_name }}
{% for field in subset.fields.values() %}
  - {{ field.name }}: {{ field.type.value }}
{% endfor %}
{% endfor %}

**The component produces:**
{% for subset_name, subset in produces.items() %}
- {{ subset_name }}
{% for field in subset.fields.values() %}
  - {{ field.name }}: {{ field.type.value }}
{% endfor %}
{% endfor %}

### Arguments

{% if arguments %}
The component takes the following arguments to alter its behavior:

| argument | type | description |
| -------- | ---- | ----------- |
{% for argument in arguments %}
| {{ argument.name }} | {{ argument.type }} | {{ argument.description }} |
{% endfor %}
{% else %}
This component takes no arguments.
{% endif %}

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


{{ id }}_op = ComponentOp.from_registry(
    name="{{ id }}",
    arguments={
        # Add arguments
{% for argument in arguments %}
{% if argument.default %}
        "{{ argument.name }}": {{ argument.default }},
{% endif %}
{% endfor %}
    }
)
pipeline.add_op({{ name }}_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
