# Manifest

A Fondant manifest describes the contents of a dataset and is used as a reference to the dataset 
to be passed between Fondant components.

Creating the manifest is handled by Fondant, as each component will automatically generate the 
manifest that represents its output dataset. While you should not have to create the manifest 
yourself, it is still useful to understand how it works.

## Contents

A manifest consists of the following sections:

```json
{
  "metadata": {
    ...
  },
  "index": {
    ...
  },
  "subsets": {
    ...
  }
}
```

### Metadata

The metadata section tracks metadata about the dataset. Currently, it tracks the location where 
the data is stored and references to the pipeline and component that were used to create it.

```json
{
  "metadata": {
    "base_path": "gs://bucket",
    "run_id": "12345",
    "component_id": "67890"
  }
}
```

This metadata is currently only used to build the location where the data should be stored, but 
can in the future be extended and used for more advanced use cases such as data lineage and 
pipeline caching.

### Index

The index represents which data points are part of the dataset. The index is tracked separately 
from the data, so data points can be filtered out of a dataset without actually having to touch 
the underlying data. The index is stored in a fixed format, which is why the manifest only 
tracks the location where it is saved.

Note that the `location` of the index is relative to the `base_path` defined in the manifest metadata.

```json
{
  "index": {
    "location": "/index"
  }
}
```

### Subsets

Each subset represents different features of the data contained in the dataset. Different subsets 
could for instance be images, captions, and embeddings, all reflecting the same underlying data 
points.

Each subset is stored separately to prevent expensive operations to the whole row when a single 
subset is added or modified.

Each subset contains two properties:
- A location which points to the location where the underlying data is stored. It is relative to 
  the `base_path` defined in the manifest metadata.
- A `"fields"` object which defines the columns available in the subset. Each field is defined 
  by a name and an [Arrow data type](https://arrow.apache.org/docs/python/api/datatypes.html). 

```json
{
  "subsets": {
    "images": {
      "location": "/images",
      "fields": {
        "data": {
          "type": "binary"
        },
        "height": {
          "type": "int32"
        },
        "width": {
          "type": "int32"
        }
      }
    },
    "captions": {
      "location": "/captions",
      "fields": {
        "data": {
          "type": "binary"
        }
      }
    }
  }
}
```

All subsets should contain at least the data points defined by the index.