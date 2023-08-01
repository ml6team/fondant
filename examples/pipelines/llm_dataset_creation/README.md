# Local language dataset creation

This pipeline demonstrates the step-by-step approach to building a local language dataset. The process begins by
utilizing the common crawl as a foundation, extracting webpage contents, and then reduce the content to a
particular language. Subsequently, additional text preprocessing steps, such as text normalization and deduplication,
are applied to refine the dataset further.

The component implementations are mainly based on the idea of [Penedo et al.](https://arxiv.org/pdf/2306.01116.pdf)

The pipeline consists of the following components:

- Load common crawl index segments
- Download webpages of the common crawl index
- Text normalization
- Text length filter
- Language filter component
- Minhash generation component
- K-Mean clustering component
- Deduplication of text fragments per cluster based on LSH