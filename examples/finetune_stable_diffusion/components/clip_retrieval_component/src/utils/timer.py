# pylint: disable-all
"""Timer helper function for benchmarking query time """
import time
import logging

import pandas as pd

from helpers.logger import get_logger

LOGGER = get_logger(name=__name__, level=logging.INFO)


class CatchTime:
    """Helper class for timing and benchmarking pieces of code.
     The benchmarks are saved in a list."""

    def __init__(self):
        self.benchmarks = []
        self.name = None
        self.enabled = False

    def __enter__(self):
        if self.enabled:
            if self.name is None:
                raise Exception("Name has to be set")
            self.begin = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if self.enabled:
            total = time.time() - self.begin
            self.benchmarks.append({'name': self.name, 'time': total})
            LOGGER.info(f"{self.name}:\t{total:.2f}s")

    def __call__(self, name):
        self.name = name

    def reset(self):
        self.name = None
        self.benchmarks = []

    def show_times(self):
        df = pd.DataFrame(self.benchmarks)
        return df

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def is_enabled(self):
        return self.enabled
