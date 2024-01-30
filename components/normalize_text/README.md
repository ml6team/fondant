# normalize_text

<a id="normalize_text#description"></a>
## Description
This component implements several text normalization techniques to clean and preprocess textual 
data:

- Apply lowercasing: Converts all text to lowercase
- Remove unnecessary whitespaces: Eliminates extra spaces between words, e.g. tabs
- Apply NFC normalization: Converts characters to their canonical representation
- Remove common seen patterns in webpages following the implementation of 
  [Penedo et al.](https://arxiv.org/pdf/2306.01116.pdf)
- Remove punctuation: Strips punctuation marks from the text

These text normalization techniques are valuable for preparing text data before using it for
the training of large language models.


<a id="normalize_text#inputs_outputs"></a>
## Inputs / outputs 

<a id="normalize_text#consumes"></a>
### Consumes 
**This component consumes:**

- text: string




<a id="normalize_text#produces"></a>  
### Produces 
**This component produces:**

- text: string



<a id="normalize_text#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| remove_additional_whitespaces | bool | If true remove all additional whitespace, tabs. | / |
| apply_nfc | bool | If true apply nfc normalization | / |
| normalize_lines | bool | If true analyze documents line-by-line and apply various rules to discard or edit lines. Used to removed common patterns in webpages, e.g. counter | / |
| do_lowercase | bool | If true apply lowercasing | / |
| remove_punctuation | str | If true punctuation will be removed | / |

<a id="normalize_text#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "normalize_text",
    arguments={
        # Add arguments
        # "remove_additional_whitespaces": False,
        # "apply_nfc": False,
        # "normalize_lines": False,
        # "do_lowercase": False,
        # "remove_punctuation": ,
    },
)
```

<a id="normalize_text#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
