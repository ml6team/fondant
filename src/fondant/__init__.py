import logging

import dask

logging.basicConfig(
    format="[%(asctime)s | %(name)s | %(levelname)s] %(message)s",
    level=logging.INFO,
)


dask.config.set({"dataframe.convert-string": False})
dask.config.set({"distributed.worker.daemon": False})
