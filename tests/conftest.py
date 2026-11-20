"""Shared fixtures. The synthetic helpers here let a ranking test hand the analyser an
importance matrix it designed by hand, so a verdict branch can be provoked on demand
rather than hunted for in the default dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from shapstab.config import Policy
from shapstab.resample import ImportanceSamples


@pytest.fixture
def policy() -> Policy:
    return Policy(
        n_resamples=200,
        base_seed=1234,
        top_k=10,
        confidence=0.95,
        tie_band=0.02,
        reliable_dominance=0.90,
        ridge_alpha=1.0,
    )


def make_samples(columns: dict[str, np.ndarray]) -> ImportanceSamples:
    """Build an ImportanceSamples from named importance columns.

    Each value is a length-B array of that feature's importance across resamples. The
    single fit is taken as the column means, which is a faithful stand-in for one fit on
    the full data when the test only cares about the ranking machinery.
    """
    names = tuple(columns)
    matrix = np.column_stack([columns[n] for n in names])
    single = matrix.mean(axis=0)
    return ImportanceSamples(samples=matrix, single_fit=single, feature_names=names)


@pytest.fixture
def samples_factory():
    return make_samples
