# Text normalization component

This component implements several text normalization techniques to clean and preprocess textual data:

- Apply lowercasing: Converts all text to lowercase
- Remove unnecessary whitespaces: Eliminates extra spaces between words, e.g. tabs
- Apply NFC normalization: Converts characters to their canonical representation
- Remove common seen patterns in webpages following the implementation of [Penedo et al.](https://arxiv.org/pdf/2306.01116.pdf)
- Remove punctuation: Strips punctuation marks from the text

These text normalization techniques are valuable for preparing text data before using it for 
the training of large language models.