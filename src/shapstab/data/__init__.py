"""Seeded, deterministic data generation. No file in this package reads the network
or the clock: a reviewer reproduces every figure from the integer seed alone.
"""

from shapstab.data.generator import DatasetSpec, make_dataset

__all__ = ["DatasetSpec", "make_dataset"]
