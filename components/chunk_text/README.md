# Chunk text

<a id="chunk_text#description"></a>
## Description
Component that chunks text into smaller segments 

This component takes a body of text and chunks into small chunks. The id of the returned dataset
consists of the id of the original document followed by the chunk index.

Different chunking strategies can be used. The default is to use the "recursive" strategy which
  recursively splits the text into smaller chunks until the chunk size is reached. 

More information about the different chunking strategies can be here:
- https://python.langchain.com/docs/modules/data_connection/document_transformers/
- https://www.pinecone.io/learn/chunking-strategies/


<a id="chunk_text#inputs_outputs"></a>
## Inputs / outputs 

<a id="chunk_text#consumes"></a>
### Consumes 
**This component consumes:**

- text: string




<a id="chunk_text#produces"></a>  
### Produces 
**This component produces:**

- text: string
- original_document_id: string



<a id="chunk_text#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| chunk_strategy | str | The strategy to use for chunking the text. One of ['RecursiveCharacterTextSplitter', 'HTMLHeaderTextSplitter', 'CharacterTextSplitter', 'Language', 'MarkdownHeaderTextSplitter', 'MarkdownTextSplitter', 'SentenceTransformersTokenTextSplitter', 'LatexTextSplitter', 'SpacyTextSplitter', 'TokenTextSplitter', 'NLTKTextSplitter', 'PythonCodeTextSplitter', 'character', 'NLTK', 'SpaCy'] | RecursiveCharacterTextSplitter |
| chunk_kwargs | dict | The arguments to pass to the chunking strategy | / |
| language_text_splitter | str | The programming language to use for splitting text into sentences if "language" is selected as the splitter. Check  https://python.langchain.com/docs/modules/data_connection/document_transformers/code_splitter for more information on supported languages. | / |

<a id="chunk_text#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "chunk_text",
    arguments={
        # Add arguments
        # "chunk_strategy": "RecursiveCharacterTextSplitter",
        # "chunk_kwargs": {},
        # "language_text_splitter": ,
    },
)
```

<a id="chunk_text#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
