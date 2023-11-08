## How Fondant handles partitions

When working with Fondant, each component deals with datasets. Fondant leverages [Dask](https://www.dask.org/) internally 
to handle datasets larger than the available memory. To achieve this, the data is divided 
into smaller chunks called "partitions" that can be processed in parallel. Ensuring a sufficient number of partitions
enables parallel processing, where multiple workers process different partitions simultaneously, 
and smaller partitions ensure they fit into memory.

Check this [link](https://docs.dask.org/en/latest/dataframe-design.html#:~:text=dd.from_delayed.-,Partitions%C2%B6,-Internally%2C%20a%20Dask) for more info on Dask partitions. 
### How Fondant handles partitions

Fondant repartitions the loaded dataframe if the number of partitions is fewer than the available workers on the data processing instance.
By repartitioning, the maximum number of workers can be efficiently utilized, leading to faster
and parallel processing.


### Customizing Partitioning

By default, Fondant automatically handles the partitioning, but you can disable this and create your
own custom partitioning logic if you have specific requirements.

Here's an example of disabling the automatic partitioning:

```python
from fondant.pipeline.pipeline import ComponentOp

caption_images_op = ComponentOp(  
    component_dir="components/captioning_component",  
    arguments={  
        "model_id": "Salesforce/blip-image-captioning-base",  
        "batch_size": 2,  
        "max_new_tokens": 50,  
    },  
    input_partition_rows='disable',  
)
```

The code snippet above disables automatic partitions for both the loaded and written dataframes, 
allowing you to define your own partitioning logic inside the components.

Moreover, you have the flexibility to set your own custom partitioning parameters to override the default settings:

```python
from fondant.pipeline.pipeline import ComponentOp

caption_images_op = ComponentOp(  
    component_dir="components/captioning_component",  
    arguments={  
        "model_id": "Salesforce/blip-image-captioning-base",  
        "batch_size": 2,  
        "max_new_tokens": 50,  
    },  
    input_partition_rows=100, 
)
```

In the example above, each partition of the loaded dataframe will contain approximately one hundred rows,
and the size of the output partitions will be around 10MB. This capability is useful in scenarios
where processing one row significantly increases the number of rows in the dataset
(resulting in dataset explosion) or causes a substantial increase in row size (e.g., fetching images from URLs).  

By setting a lower value for input partition rows, you can mitigate issues where the processed data
grows larger than the available memory before being written to disk.