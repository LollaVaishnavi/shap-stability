from __future__ import annotations

import numpy as np
import pytest

from shapstab.data import DatasetSpec, make_dataset
from shapstab.data.generator import DataError


def test_the_same_seed_reproduces_the_dataset_byte_for_byte():
    a_x, a_y = make_dataset(DatasetSpec(seed=7))
    b_x, b_y = make_dataset(DatasetSpec(seed=7))
    assert np.array_equal(a_x, b_x)
    assert np.array_equal(a_y, b_y)


def test_a_different_seed_gives_a_different_dataset():
    a_x, _ = make_dataset(DatasetSpec(seed=7))
    b_x, _ = make_dataset(DatasetSpec(seed=8))
    assert not np.array_equal(a_x, b_x)


def test_shapes_match_the_spec():
    spec = DatasetSpec(n_rows=321, n_features=9, seed=1)
    x, y = make_dataset(spec)
    assert x.shape == (321, 9)
    assert y.shape == (321,)


def test_the_pair_is_correlated_near_its_target():
    spec = DatasetSpec(pair_rho=0.995, seed=2)
    x, _ = make_dataset(spec)
    corr = np.corrcoef(x[:, 0], x[:, 1])[0, 1]
    assert corr > 0.98


def test_the_trio_shares_signal_and_the_tail_does_not():
    spec = DatasetSpec(trio_rho=0.9, seed=2)
    x, _ = make_dataset(spec)
    trio_corr = np.corrcoef(x[:, 2], x[:, 3])[0, 1]
    tail_corr = abs(np.corrcoef(x[:, 6], x[:, 7])[0, 1])
    assert trio_corr > 0.8
    assert tail_corr < 0.2


def test_true_coef_places_the_groups_as_documented():
    spec = DatasetSpec(pair_coef=1.6, trio_coef=1.0, tail_coef=0.9)
    coef = spec.true_coef
    assert coef[0] == coef[1] == 1.6
    assert coef[2] == coef[3] == coef[4] == 1.0
    # Tail decays, so each tail coefficient is smaller than the previous.
    tail = coef[5:]
    assert np.all(np.diff(tail) < 0)


def test_default_feature_names_are_zero_padded_and_count_matches():
    spec = DatasetSpec(n_features=12)
    assert spec.feature_names[0] == "x00"
    assert len(spec.feature_names) == 12


def test_custom_feature_names_must_match_feature_count():
    with pytest.raises(DataError, match="feature_names length"):
        DatasetSpec(n_features=6, feature_names=("a", "b"))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_rows": 1}, "n_rows"),
        ({"n_features": 5}, "n_features"),
        ({"trio_rho": 1.0}, "trio_rho"),
        ({"pair_rho": 1.0}, "pair_rho"),
        ({"noise_sd": -1.0}, "noise_sd"),
        ({"pair_coef": 0.0}, "coefficients"),
    ],
)
def test_a_contradictory_spec_is_refused_with_a_named_reason(kwargs, match):
    with pytest.raises(DataError, match=match):
        DatasetSpec(**kwargs)
