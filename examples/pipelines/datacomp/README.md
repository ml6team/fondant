# DataComp pipeline

[DataComp](https://www.datacomp.ai/) is a competition organized by the University of Washington and others to come up with the best possible image-text dataset to train a fixed CLIP model. Hence, it's an ideal use case for Fondant, as we can leverage reusable components to filter large, noisy image-text datasets.

Currently, 2 pipelines are implemented:

- a simple pipeline (`simple_pipeline.py`), which loads the DataComp dataset from the hub and applies 2 basic filtering steps (filtering on image resolution and caption complexity). This pipeline serves as a baseline and could serve as a first submission.
- a more complex pipeline (`pipeline.py`), which loads the DataComp dataset from the hub, loads the actual images based on the URLs, and applies text detection and text recognition models to filter the dataset.