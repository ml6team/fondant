# Common Crawl Download component

The Common Crawl Download Component serves as a LoadComponent for initializing a dataset from the
Common Crawl. The component takes a list of
[columnar common crawl index files](https://commoncrawl.org/2018/03/index-to-warc-files-and-urls-in-columnar-format/)
and applies provided pyarrow filters to narrow down the download to relevant webpages.
The available filters allow you to precisely define the scope of the download. The component offers
extraction capabilities for obtaining either plain HTML of the webpages or extracting the plain
text, providing flexibility for your specific use case.

**Component Arguments**
- common_crawl_indices: List of common crawl index files

- filters:

  PyArrow filter defintions following the format
  `{"field": "...", "operator": "...", "value": "..."}`
  E.g. `{"field": "content_mime_type", "operator": "==", "value": "text/html"}`


- extract_plain_text: If true component will extract will convert the html content to plain text

- n_records_to_download: limit the number of records to download


Common Crawl data is accessible via HTTP requests, with the columnar index's Parquet files residing
in a publicly available S3 bucket. You can access these files through HTTP requests. However,
obtaining a comprehensive list of available index files using unauthenticated HTTP requests is not
feasible.

To retrieve a list of all columnar index files for a specific crawl you can utilise the `aws cli` as
follows:

```bash
CRAWL=CC-MAIN-2023-23 # replace it with the specific crawl identifier
aws s3 ls --recursive --human-readable --summarize "s3://commoncrawl/cc-index/table/cc-main/warc/crawl=${CRAWL}"
```

> Note: A list of all available crawl identifies you can find
> on [here](https://data.commoncrawl.org/cc-index/collections/index.html)

Depending on your specific use case, we recommend restricting the initial list of index files. For
instance, the index for `CC-MAIN-2023-23` comprises 900 files, totaling around 265GiB in compressed
size.