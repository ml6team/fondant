# Implementing custom components

This guide will teach you how to build custom components and integrate them in your pipeline.

## Overview

In the [previous tutorial](/build_a_simple_pipeline.md), you learned how to create your first
Fondant pipeline. While the example demonstrates how to build a pipeline from reusable components,
this is only the beginning.

Reusable components consume data in a specific format, defined in a data contract.
Therefore, it is often necessary to implement custom components to connect the reusable components to your
specific data. The easiest way to do this is to implement a **Lightweight Component**. 

In this tutorial, we will guide you through the process of implementing your very own custom
component. We will illustrate this by building a transform component that uppercases the `alt_text` of the image dataset.

If you want to build a complex custom component or share the component within your organization or even the community, 
take a look at how to build [reusable components](../components/containerized_components.md).

This pipeline is an extension of the one introduced in
the [previous tutorial](../guides/build_a_simple_pipeline.md).
Make sure you have completed the tutorial before diving into this one.

In the last tutorial, we implemented this pipeline:

```python
from fondant.pipeline import Pipeline
import pyarrow as pa

pipeline = Pipeline(
    name="creative_commons_pipline",
    base_path="./data"
)

dataset = pipeline.read(
    "load_from_hf_hub",
    arguments={
        "dataset_name": "fondant-ai/fondant-cc-25m",
        "n_rows_to_load": 100,
    },
    produces={
        "alt_text": pa.string(),
        "image_url": pa.string(),
        "license_location": pa.string(),
        "license_type": pa.string(),
        "webpage_url": pa.string(),
    }
)

images = dataset.apply(
    "download_images",
)

english_images = images.apply(
    "filter_language",
    arguments={
        "language": "en"
    },
    consumes={
        "text": "alt_text"
    }
)
```

We want to extend the pipeline and apply a simple text transformation to the `alt_text`. Let's
consider that the `alt_text` is so important that the text has to be transformed into uppercase
letters.

## Implement your  Lightweight component

Now, it's time to implement the component logic.

We will subclass the `PandasTransformComponent` offered by Fondant. This is the most basic type
of component. The following method should be implemented:

- **`transform()`**: This method receives a chunk of the input data as a Pandas `DataFrame`.
  Fondant automatically chunks your data you can process larger-than-memory data, and your
  component is executed in parallel across the available cores.

```python
"""A component that transform the alt text of the dataframe into uppercase."""
import pandas as pd
from fondant.component import PandasTransformComponent
from fondant.pipeline import lightweight_component


@lightweight_component
class UpperCaseTextComponent(PandasTransformComponent):

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Transform the alt text into upper case."""
        dataframe["alt_text"] = dataframe["alt_text"].apply(lambda x: x.upper())
        return dataframe
```

!!! note "IMPORTANT"

    Note that we have used a decorator `@lightweight_component`. This decorator is necessary to inform
    Fondant that this class is a lightweight component and can be used as a component in your pipeline.

We apply the uppercase transformation to the `alt_text` column of the dataframe. Afterward, we
return the transformed dataframe from the `transform` method, which Fondant will use to
automatically update the index.

The lightweight components provide an easy way to start with your component implementation. However, the
lightweight component implementation still allows you to define all advanced component configurations,
including installing extra arguments or defining component arguments. These concepts are more
advanced and not needed for quick exploration and experiments. You can find more information on
these topics in
the [documentation of the lightweight components](../components/lightweight_components.md).

### Using the component

Now were we have defined our lightweight component we can start using it in our pipeline.
For instance we can put this component at the end of our pipeline.

```python

uppercase_alt_text = english_images.apply(
    UpperCaseTextComponent
)

```

Instead of providing the name of the component, as we did with the reusable components,
we now provide the component implementation.

Now, you can execute the pipeline once more and examine the results. In the final output,
the `alt_text` is in uppercase.

Of course, it is debatable whether uppercasing the alt_text is genuinely useful. This is just a
constructive and simple example to showcase how to use lightweight components as glue code within your
pipeline, helping you connect reusable components to each other.

## Next steps

We now have a pipeline that downloads a dataset from the HuggingFace hub, filters the urls by
image type, downloads the images, and filters them by alt text language.

If you want to inspect your final dataset without using the data explorer or use the 
dataset for further tasks, we recommend to write the final dataset to a destination. 
We offer [write components](../components//hub.md) to perform this task, for instance the `write_to_file` component, 
which allows you to export the dataset either to a local file or a remote storage bucket.

```python
uppercase_alt_text.write(ref="write_to_file", arguments={"path": "/data/export"})
```

You can open the path and use any tools of your choice to inspect the resulting Parquet dataset.
