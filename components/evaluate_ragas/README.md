# Evalute ragas

## Description {: #description_evalute_ragas}
Component that evaluates the retriever using RAGAS

## Inputs / outputs  {: #inputs_outputs_evalute_ragas}

### Consumes  {: #consumes_evalute_ragas}
**This component consumes:**

- question: string
- retrieved_chunks: list<item: string>





### Produces {: #produces_evalute_ragas}

**This component can produce additional fields**
- <field_name>: <field_schema>
This defines a mapping to update the fields produced by the operation as defined in the component spec.
The keys are the names of the fields to be produced by the component, while the values are 
the type of the field that should be used to write the output dataset.


## Arguments {: #arguments_evalute_ragas}

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| module | str | Module from which the LLM is imported. Defaults to langchain.llms | langchain.llms |
| llm_name | str | Name of the selected llm | / |
| llm_kwargs | dict | Arguments of the selected llm | / |

## Usage {: #usage_evalute_ragas}

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "evaluate_ragas",
    arguments={
        # Add arguments
        # "module": "langchain.llms",
        # "llm_name": ,
        # "llm_kwargs": {},
    },
    produces={
         <field_name>: <field_schema>,
         ..., # Add fields
    },
)
```

## Testing {: #testing_evalute_ragas}

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
