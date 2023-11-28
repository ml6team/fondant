# {{ name }}

### Description
{{ description }}

### Inputs / outputs

{% if consumes %}
**This component consumes:**

{% for field_name, field in consumes.items() %}
- {{ field.name }}: {{ field.type.value }}
{% endfor %}
{% else %}
**This component consumes no data.**
{% endif %}

{% if produces %}
**This component produces:**

{% for field_name, field in produces.items() %}
- {{ field.name }}: {{ field.type.value }}
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
    }
)
```

{% if tests %}
### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
{% endif %}
