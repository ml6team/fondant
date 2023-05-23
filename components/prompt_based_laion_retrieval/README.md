# prompt_based_laion_retrieval

### Description
This component retrieves image URLs from the [LAION-5B dataset](https://laion.ai/blog/laion-5b/) based on text prompts. The retrieval itself is done based on CLIP embeddings similarity between the prompt sentences and the captions in the LAION dataset. This component doesn’t return the actual images, only URLs.

### **Inputs/Outputs**

See the [`component specification`](fondant_component.yaml) for a more detailed description of all the input/output parameters. 

