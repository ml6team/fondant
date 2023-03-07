"""
General helpers functions for reading and updating parquet datasets
"""

import os
import itertools
import logging
from typing import Dict, Callable, List, Union, Iterable, Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc
from pyarrow.dataset import Expression, Scanner

# pylint: disable=import-error
from .logger import get_logger
from .io_helpers import *

LOGGER = get_logger(name=__name__, level=logging.INFO)


def get_batches_from_iterable(data_iterable: Iterable, schema: pa.schema,
                              chunk_size: int) -> pa.record_batch:
    """
    Function that returns batches from an iterable to write to a parquet dataset
    Args:
        data_iterable (Callable): an iterator that yields a tuple. Used to return
         the data to write (in chunks)
        schema (List[Tuple[str, any]): the schema of the file to write
        chunk_size (int): the chunk size that is used to restrict the number of rows that are
        being read and written to the parquet file on each loop (avoid memory overflow when
        reading large amounts of data)

    """
    rows_it = iter(data_iterable)
    while True:
        batch = pa.RecordBatch.from_pandas(
            df=pd.DataFrame(itertools.islice(rows_it, chunk_size), columns=schema.names),
            schema=schema, preserve_index=False)
        if not batch:
            break
        yield batch


def write_parquet_file(parquet_path: str, data_iterable: Iterable,
                       schema: pa.schema, chunk_size: int = 1000) -> None:
    """
    Function that writes a parquet file from an iterable functions and stores it in a local storage
    Args:
        parquet_path (str): the path to the parquet file
        data_iterable (Callable): an iterator that yields a tuple. Used to return
         the data to write (in chunks)
        schema (List[Tuple[str, any]): the schema of the file to write
        chunk_size (int): the chunk size that is used to restrict the number of rows that are
        being read and written to the parquet file on each loop (avoid memory overflow when
        reading large amounts of data)
    """

    batches = get_batches_from_iterable(data_iterable=data_iterable, schema=schema,
                                        chunk_size=chunk_size)

    # Write the batches
    with pq.ParquetWriter(parquet_path, schema=schema) as writer:
        for batch in batches:
            writer.write_batch(batch)


def append_to_parquet_file(parquet_path: str, data_iterable: Iterable, tmp_path: str,
                           chunk_size: int = 1000) -> None:
    """
    Function that append the results of an iterable function to an existing parquet file
    Args:
        parquet_path (str): the path to the parquet file to append to
        data_iterable (Callable): an iterator that yields a tuple. Used to return
         the data to write (in chunks)
        tmp_path (str): the temporary path where to write the temporary appended parquet file
        chunk_size (int): the chunk size that is used to restrict the number of rows that are
        being read and written to the parquet file on each loop (avoid memory overflow when
        reading large amounts of data)
    """

    src_dataset = ds.dataset(parquet_path)
    src_batches = src_dataset.to_batches()
    batches_to_append = get_batches_from_iterable(data_iterable=data_iterable,
                                                  schema=src_dataset.schema,
                                                  chunk_size=chunk_size)
    # Results will be written to a temporary parquet first since we cannot overwrite the original
    # while reading batches from it
    file_name = f"{io_helpers.get_file_name(parquet_path)}_tmp.parquet"
    tmp_parquet_path = os.path.join(tmp_path, file_name)
    # Write the batches
    with pq.ParquetWriter(tmp_parquet_path, schema=src_dataset.schema) as writer:
        for src_batch in src_batches:
            writer.write_batch(src_batch)
        # Write the appended batches
        for batch_to_append in batches_to_append:
            writer.write_batch(batch_to_append)
    os.replace(tmp_parquet_path, parquet_path)


def get_column_list_from_parquet(parquet_scanner_or_path: Union[str, Scanner],
                                 column_name: str) -> List[Any]:
    """
    Function that returns a defined parquet column as a list
    Args:
        parquet_scanner_or_path: Union[str, Scanner]: a path to a parquet file or the
        parquet dataset
        column_name (str): the name of the column
    Returns:
        List[str]: a list containing the column rows
    """
    column_list = []
    if isinstance(parquet_scanner_or_path, str):
        parquet_batches_getter = ds.dataset(parquet_scanner_or_path).to_batches()
    else:
        parquet_batches_getter = parquet_scanner_or_path.to_batches()
    for batch in parquet_batches_getter:
        column_list.extend(batch.to_pydict()[column_name])

    return column_list


def remove_common_duplicates(dataset_to_filter_path: str, reference_dataset_path: str,
                             duplicate_columns_name: str, tmp_path: str):
    """
    Function that removes overlapping duplicates from a dataset with respect to a field (columns)
    in a reference dataset
    Args:
        dataset_to_filter_path (str): the source to the dataset from which to filter out the
         duplicates
        reference_dataset_path (str): the reference dataset containing the duplicate ids
        duplicate_columns_name (str): the reference columns where the duplicates exist
        tmp_path (str): the temporary path where to write the temporary appended parquet file
    """
    index_dataset = get_column_list_from_parquet(
        parquet_scanner_or_path=dataset_to_filter_path,
        column_name=duplicate_columns_name)
    index_reference_dataset = get_column_list_from_parquet(
        parquet_scanner_or_path=reference_dataset_path,
        column_name=duplicate_columns_name)
    # Pyarrow does not have a "not_in" filter we have to find the non overlapping column elements
    # between the two datasets
    non_overlapping_indices = list(set(index_dataset) ^ set(index_reference_dataset))
    nb_duplicates = ((len(index_dataset) + len(index_reference_dataset)) - len(
        non_overlapping_indices)) / 2
    # Construct parquet filters and filter based on the criteria
    filters = (pc.field(duplicate_columns_name).isin(non_overlapping_indices))

    file_name = f"{io_helpers.get_file_name(dataset_to_filter_path)}_tmp.parquet"
    tmp_parquet_path = os.path.join(tmp_path, file_name)

    filtered_dataset = filter_parquet_file(file_path=dataset_to_filter_path, filters=filters)

    # Write the new filtered parquet file
    with pq.ParquetWriter(tmp_parquet_path, schema=filtered_dataset.dataset_schema) as writer:
        for batch in filtered_dataset.to_batches():
            writer.write_batch(batch)

    os.replace(tmp_parquet_path, dataset_to_filter_path)

    LOGGER.info("Number of removed duplicates from %s was %s", dataset_to_filter_path,
                nb_duplicates)


def write_index_parquet(index_parquet_path: str, data_iterable_producer: Callable, **kwargs):
    """
    Function that writes the index id parquet information to a parquet file
    Args:

        index_parquet_path (str): the path to the index id parquet file
        data_iterable_producer (Callable): an iterable function that returns a tuple and is
         used to return the data to write (in chunks)
    """

    # Define index id schema
    schema = pa.schema([
        pa.field('index', pa.string())],
        metadata={"description": "Parquet file containing the list of index ids of images"
                                 "in the format <namespace>_<uid>"})

    write_parquet_file(parquet_path=index_parquet_path,
                       data_iterable=data_iterable_producer(**kwargs),
                       schema=schema)

    LOGGER.info("index id parquet file written to %s.", index_parquet_path)


def write_dataset_parquet(dataset_parquet_path: str, data_iterable_producer: Callable, **kwargs):
    """
    Function that dataset parquet information to a parquet file
    Args:
        dataset_parquet_path (str): the path to the dataset parquet file
        data_iterable_producer (Callable): an iterable function that returns a tuple and
        is used to return the data to write (in chunks)
    """

    schema = pa.schema([
        pa.field('file_uri', pa.string()),
        pa.field('file_id', pa.string()),
        pa.field('file_size', pa.int32()),
        pa.field('file_extension', pa.string())
    ],
        metadata={"description": "Parquet file containing info of the image dataset (id,"
                                 "uri, size, extension)"})

    write_parquet_file(parquet_path=dataset_parquet_path,
                       data_iterable=data_iterable_producer(**kwargs),
                       schema=schema)

    LOGGER.info("dataset parquet file written to %s.", dataset_parquet_path)


def write_captions_parquet(caption_parquet_path: str, data_iterable_producer: Callable, **kwargs):
    """
    Function that dataset parquet information to a parquet file
    Args:
        caption_parquet_path (str): the path to the caption parquet file
        data_iterable_producer (Callable): an iterable function that returns a tuple and is used to
        return the data to write (in chunks)
    """

    schema = pa.schema([
        pa.field('file_id', pa.string()),
        pa.field('file_uri', pa.string()),
        pa.field('file_captions', pa.list_(pa.string()))
    ],
        metadata={"description": "Parquet file containing info of the image dataset (id,"
                                 "uri, size, extension)"})

    write_parquet_file(parquet_path=caption_parquet_path,
                       data_iterable=data_iterable_producer(**kwargs),
                       schema=schema)

    LOGGER.info("dataset captions file written to %s.", caption_parquet_path)


def write_clip_retrieval_parquet(clip_retrieval_parquet_path: str,
                                 data_iterable_producer: Callable, **kwargs):
    """
    Function that dataset parquet information to a parquet file
    Args:
        clip_retrieval_parquet_path (str): the path to the clip retrieval parquet file
        data_iterable_producer (Callable): an iterable function that returns a tuple and is used to
        return the data to write (in chunks)
    """

    schema = pa.schema([
        pa.field('id', pa.int64()),
        pa.field('url', pa.string())
    ],
        metadata={"description": "Parquet file containing info of the clip retrieval job"})

    write_parquet_file(parquet_path=clip_retrieval_parquet_path,
                       data_iterable=data_iterable_producer(**kwargs),
                       schema=schema)

    LOGGER.info("clip retrieval parquet path file written to %s.", clip_retrieval_parquet_path)


def insert_metadata(file_name: str, file_path: str, custom_metadata: Dict[any, any]):
    """
    Update metadata of parquet file
    Args:
        file_name (str): the name of the parquet file
        file_path (str): the path to the parquet file
        custom_metadata (Dict[any, any]): the new metadata to update
    """
    file_path = os.path.join(file_path, file_name)
    parquet_table = pa.parquet.read_table(file_path)
    existing_metadata = parquet_table.schema.metadata
    merged_metadata = {**custom_metadata, **existing_metadata}
    parquet_table = parquet_table.replace_schema_metadata(merged_metadata)
    pq.write_table(parquet_table, file_path)


def filter_parquet_file(file_path: str, filters: Expression,
                        batch_size: Optional[int] = 10_000) -> Scanner:
    """
    Function that loads in a parquet file and filters it
    Args:
        file_path (str): the path to the parquet file
        filters (Expression): a collection of filters to apply. See
        "https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html"
        for more details
        batch_size (int): the batch size of the scanner
    Returns:
        Scanner: scanned dataset
    """
    dataset = ds.dataset(file_path).scanner(filter=filters, batch_size=batch_size)

    return dataset


def get_nb_rows_from_parquet(parquet_dataset_or_path: Union[str, pq.ParquetDataset]) -> int:
    """
    Function that returns the number of rows of a parquet file
    Args:
        parquet_dataset_or_path: Union[str, pq.ParquetDataset]: a path to a parquet file or the
        parquet dataset
    Returns:
        int: the total number of rows
    """
    if isinstance(parquet_dataset_or_path, str):
        dataset = pq.ParquetDataset(parquet_dataset_or_path, use_legacy_dataset=False)
    else:
        dataset = parquet_dataset_or_path
    return sum([fragment.metadata.num_rows for fragment in dataset.fragments])
