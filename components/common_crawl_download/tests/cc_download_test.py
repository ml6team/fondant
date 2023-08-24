from main import CommonCrawlDownloadComponent


def test_load_warc_file_content():
    """Test components transform method."""
    # TODO implement a real test case
    n_records_to_download = 10
    component = CommonCrawlDownloadComponent(
        filters=
        [{"field": "content_mime_type", "operator": "==", "value": "text/html"},
         {"field": "content_languages", "operator": "==", "value": "deu"},
         {"field": "content_charset", "operator": "==", "value": "UTF-8"}],
        extract_plain_text=True,
        n_records_to_download= n_records_to_download,
        common_crawl_indices=[
            "cc-index/table/cc-main/warc/crawl=CC-MAIN-2023-23/subset=warc/part-00125-ffa3bf93-6ba1-4a27-adea-b0baae3b4389.c000.gz.parquet",
        ])

    ddf = component.load()
    assert len(ddf) == n_records_to_download
