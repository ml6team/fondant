## What is caching?

Fondant supports caching of workflow executions. If a certain component and its arguments
are exactly the same as in some previous execution, then its execution can be skipped and the output
dataset of the previous execution can be used instead.

Caching offers the following benefits:  
1) **Reduced costs.** Skipping the execution of certain components can help avoid unnecessary costly computations.  
2) **Faster workflow runs.** Skipping the execution of certain components results in faster workflow execution.  
3) **Faster dataset development.** Caching allows you develop and test your datasets faster.  
4) **Reproducibility.** Caching allows you to reproduce the results of a run by reusing
   the outputs of a previous run.

!!! note "IMPORTANT"  

      The cached runs are tied to the working directory which stores the caching key of previous component runs. 
      Changing the orking directory will invalidate the cache of previous materialized datasets.

The caching feature is **enabled** by default. 


## Disabling caching
You can turn off execution caching at component level by setting the following:

```python
from fondant.dataset.dataset import ComponentOp

caption_images_op = ComponentOp(
    component_dir="...",
    arguments={
        ...
    },
    cache=False,
)
```

## How caching works
When Fondant materializes a dataset, it checks to see whether an execution exists in the working directory based on
the cache key of each component.

The cache key is defined as the combination of the following:

1) The **operation step's inputs.** These inputs include the input arguments' value (if any).

2) **The component's specification.** This specification includes the image tag and the fields
   consumed and produced by each component.

3) **The component resources.** Defines the hardware that was used to run the component (GPU,
   nodepool).

If there is a matching execution in the base path (checked based on the output manifests),
the outputs of that execution are used and the step computation is skipped.

Additionally, only datasets with the same dataset name will share the cache. Caching for
components
with the `latest` image tag is disabled by default. This is because using `latest` image tags can
lead to unpredictable behavior due to
image updates. Moreover, if one component in the dataset is not cached then caching will be
disabled for all subsequent components.

