from __future__ import annotations

import numpy as np
import pytest

from shapstab.model import (
    ModelError,
    Standardiser,
    fit_ridge,
    fit_standardiser,
)


def test_standardiser_centres_and_scales_each_column():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, size=(400, 4))
    z = fit_standardiser(x).transform(x)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-9)


def test_a_constant_column_gets_scale_one_and_contributes_no_signal():
    x = np.ones((10, 2))
    x[:, 1] = np.arange(10)
    std = fit_standardiser(x)
    assert std.scale[0] == 1.0  # not a division by zero
    z = std.transform(x)
    assert np.allclose(z[:, 0], 0.0)


def test_ridge_recovers_known_coefficients_on_clean_data():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((2000, 3))
    beta_true = np.array([2.0, -1.0, 0.5])
    y = z @ beta_true
    beta = fit_ridge(z, y, alpha=1e-6)
    assert np.allclose(beta, beta_true, atol=1e-2)


def test_ridge_matches_scikit_learn_within_tolerance():
    # Cross-check the hand-written solve against a reference implementation. The two
    # standardise and centre identically, so the coefficients should agree closely.
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(2)
    z = rng.standard_normal((500, 5))
    y = z @ np.array([1.0, 0.0, -2.0, 0.3, 0.0]) + rng.standard_normal(500)
    alpha = 2.0
    mine = fit_ridge(z, y, alpha)
    ref = sklearn_linear.Ridge(alpha=alpha, fit_intercept=True).fit(z, y)
    assert np.allclose(mine, ref.coef_, atol=1e-8)


def test_ridge_is_unchanged_by_a_constant_shift_in_a_feature():
    # The war story, pinned. fit_ridge once centred only y, so a feature with a non-zero
    # mean (which every bootstrap resample has) folded its offset into the coefficients.
    # A proper intercept makes the fit invariant to shifting a whole column by a constant;
    # the earlier version failed this and disagreed with scikit-learn at the third decimal.
    rng = np.random.default_rng(7)
    z = rng.standard_normal((400, 4))
    y = z @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.standard_normal(400)
    shifted = z.copy()
    shifted[:, 0] += 13.0  # a large constant offset on one column
    assert np.allclose(fit_ridge(z, y, 1.0), fit_ridge(shifted, y, 1.0), atol=1e-8)


def test_ridge_rejects_a_row_count_mismatch():
    with pytest.raises(ModelError, match="same number of rows"):
        fit_ridge(np.zeros((5, 2)), np.zeros(4), alpha=1.0)


def test_ridge_rejects_a_negative_penalty():
    with pytest.raises(ModelError, match="non-negative"):
        fit_ridge(np.zeros((5, 2)), np.zeros(5), alpha=-1.0)


def test_a_non_two_dimensional_design_matrix_is_refused():
    with pytest.raises(ModelError, match="2-D"):
        fit_standardiser(np.zeros(10))


def test_an_empty_design_matrix_is_refused():
    with pytest.raises(ModelError, match="at least one row"):
        fit_standardiser(np.zeros((0, 3)))


def test_a_non_finite_design_matrix_is_refused():
    x = np.ones((4, 2))
    x[0, 0] = np.nan
    with pytest.raises(ModelError, match="non-finite"):
        fit_standardiser(x)


def test_standardiser_is_frozen():
    std = Standardiser(mean=np.zeros(2), scale=np.ones(2))
    with pytest.raises(Exception):  # noqa: B017
        std.mean = np.ones(2)  # type: ignore[misc]
