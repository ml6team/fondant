# Evalute ragas

<a id="evalute_ragas#description"></a>
## Description
Component that evaluates the retriever using RAGAS

<a id="evalute_ragas#inputs_outputs"></a>
## Inputs / outputs 

<a id="evalute_ragas#consumes"></a>
### Consumes 
**This component consumes:**

- question: string
- retrieved_chunks: list<item: string>




<a id="evalute_ragas#produces"></a>  
### Produces 

**This component can produce additional fields**
- <field_name>: <field_schema>
This defines a mapping to update the fields produced by the operation as defined in the component spec.
The keys are the names of the fields to be produced by the component, while the values are 
the type of the field that should be used to write the output dataset.


<a id="evalute_ragas#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| llm_module_name | str | Module from which the LLM is imported. Defaults to langchain.llms | langchain.chat_models |
| llm_class_name | str | Name of the selected llm | ChatOpenAI |
| llm_kwargs | dict | Arguments of the selected llm | {'model_name': 'gpt-3.5-turbo'} |

<a id="evalute_ragas#usage"></a>
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
        # "llm_module_name": "langchain.chat_models",
        # "llm_class_name": "ChatOpenAI",
        # "llm_kwargs": {'model_name': 'gpt-3.5-turbo'},
    },
    produces={
         <field_name>: <field_schema>,
         ..., # Add fields
    },
)
```

<a id="evalute_ragas#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
