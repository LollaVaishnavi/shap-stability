from __future__ import annotations

import numpy as np
import pytest

from shapstab.data import DatasetSpec, make_dataset
from shapstab.model import ModelError
from shapstab.resample import (
    ImportanceSamples,
    resample_importances,
    single_fit_importance,
)


def _small():
    return make_dataset(DatasetSpec(n_rows=300, seed=11))


def test_the_sample_matrix_has_one_row_per_resample_and_one_column_per_feature():
    x, y = _small()
    imp = resample_importances(x, y, n_resamples=60, base_seed=1, alpha=1.0)
    assert imp.samples.shape == (60, x.shape[1])
    assert imp.single_fit.shape == (x.shape[1],)
    assert imp.n_resamples == 60
    assert imp.n_features == x.shape[1]


def test_the_same_base_seed_reproduces_the_whole_matrix():
    x, y = _small()
    a = resample_importances(x, y, n_resamples=40, base_seed=5, alpha=1.0)
    b = resample_importances(x, y, n_resamples=40, base_seed=5, alpha=1.0)
    assert np.array_equal(a.samples, b.samples)


def test_a_different_base_seed_moves_the_matrix():
    x, y = _small()
    a = resample_importances(x, y, n_resamples=40, base_seed=5, alpha=1.0)
    b = resample_importances(x, y, n_resamples=40, base_seed=6, alpha=1.0)
    assert not np.array_equal(a.samples, b.samples)


def test_importances_are_non_negative_because_they_are_mean_absolute_values():
    x, y = _small()
    imp = resample_importances(x, y, n_resamples=30, base_seed=1, alpha=1.0)
    assert np.all(imp.samples >= 0.0)


def test_single_fit_importance_returns_a_vector_matched_eval_and_standardiser():
    x, y = _small()
    imp, z, std = single_fit_importance(x, y, alpha=1.0)
    assert imp.shape == (x.shape[1],)
    assert z.shape == x.shape
    assert std.mean.shape == (x.shape[1],)


def test_too_few_resamples_is_refused():
    x, y = _small()
    with pytest.raises(ModelError, match="at least 2"):
        resample_importances(x, y, n_resamples=1, base_seed=1, alpha=1.0)


def test_a_feature_name_count_mismatch_is_refused():
    x, y = _small()
    with pytest.raises(ModelError, match="feature_names"):
        resample_importances(
            x, y, n_resamples=10, base_seed=1, alpha=1.0, feature_names=("only-one",)
        )


def test_default_feature_names_are_supplied_when_omitted():
    x, y = _small()
    imp = resample_importances(x, y, n_resamples=10, base_seed=1, alpha=1.0)
    assert imp.feature_names[0] == "x00"


def test_importance_samples_is_frozen():
    imp = ImportanceSamples(np.zeros((2, 2)), np.zeros(2), ("a", "b"))
    with pytest.raises(Exception):  # noqa: B017
        imp.samples = np.ones((2, 2))  # type: ignore[misc]
