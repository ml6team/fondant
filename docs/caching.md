## Caching 
When Fondant runs a pipeline, it checks to see whether an execution exists in the base path based on the cache key of each component. 

The cache key is defined as the combination of the following:

1) The **pipeline step's inputs.** These inputs include the input arguments' value (if any).

2) **The component's specification.** This specification includes the image tag and the fields consumed and produced by each component.

3) **The component resources.** Defines the hardware that was used to run the component (GPU, nodepool).

If there is a matching execution in the base path (checked based on the output manifests),
the outputs of that execution are used and the step computation is skipped.
This helps to reduce costs by skipping computations that were completed in a previous pipeline run.


Additionally, only the pipelines with the same pipeline name will share the cache. Caching for components
with the `latest` image tag is disabled by default. This is because using "latest" image tags can lead to unpredictable behavior due to 
image updates. Moreover, if one component in the pipeline is not cached then caching will be disabled for all 
subsequent components. 

You can turn off execution caching at component level by setting the following:

```python
from fondant.pipeline.pipeline import ComponentOp

caption_images_op = ComponentOp(  
    component_dir="...",  
    arguments={
        ... 
    },  
    cache=False,  
)
```