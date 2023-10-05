# {{ name }}

### Description
{{ description }}

### Inputs / outputs

{% if consumes %}
**This component consumes:**
{% for subset_name, subset in consumes.items() %}
- {{ subset_name }}
{% for field in subset.fields.values() %}
  - {{ field.name }}: {{ field.type.value }}
{% endfor %}
{% endfor %}
{% else %}
**This component consumes no data.**
{% endif %}

{% if produces %}
**This component produces:**
{% for subset_name, subset in produces.items() %}
- {{ subset_name }}
{% for field in subset.fields.values() %}
  - {{ field.name }}: {{ field.type.value }}
{% endfor %}
{% endfor %}
{% else %}
**This component produces no data.**
{% endif %}

### Arguments

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
        # "{{ argument.name }}": {{ '\"' + argument.default + '\"' if argument.default is string else argument.default }},
{% else %}
        # "{{ argument.name }}": {{ (argument.type|eval)() }},
{% endif %}
{% endfor %}
    }
)
pipeline.add_op({{ id }}_op, dependencies=[...])  #Add previous component as dependency
```

{% if tests %}
### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
{% endif %}
