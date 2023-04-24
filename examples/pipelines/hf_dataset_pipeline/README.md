This folder contains an example KubeFlow pipeline that consists of 2 components:

- a loader component which loads a dataset from the 🤗 hub and uploads it to GCS, leveraging the `HFDatasetsLoaderComponent` class of Fondant.
- a transform component which adds captions to the dataset using BLIP, leveraging the `HFDatasetsTransformComponent` class.



