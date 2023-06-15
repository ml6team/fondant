# Starcoder pipeline

This pipeline illustrates a tiny portion of the data preparation of StarCoder, an open-source version of Github CoPilot, trained as part of the the [BigCode](https://www.bigcode-project.org/) project.

The pipeline is based on [this repository](https://github.com/bigcode-project/bigcode-dataset).

The pipeline includes the following components:

- loading a code dataset from the Hugging Face hub
- filtering code based on comment to code ratio
- filtering code based on line length
- detecting and replacing PII (personal identifiable information) from code.