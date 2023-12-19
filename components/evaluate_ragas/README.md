# retriever_eval_ragas

## Description
Component that evaluates the retriever using RAGAS

## Inputs / outputs

### Consumes
**This component consumes:**

- question: string
- retrieved_chunks: list<item: string>



### Produces

_**This component does not produce specific data.**_

**This component produces generic data**
- <field_name>: <field_schema>

## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| module | str | Module from which the LLM is imported. Defaults to langchain.llms | langchain.llms |
| llm_name | str | Name of the selected llm | / |
| llm_kwargs | dict | Arguments of the selected llm | / |

## Usage

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

## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
