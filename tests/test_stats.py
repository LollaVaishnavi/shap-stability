from __future__ import annotations

import math

import pytest

from shapstab.stats import normal_quantile


def test_the_median_quantile_is_zero():
    assert abs(normal_quantile(0.5)) < 1e-12


def test_the_classic_ninety_five_percent_z_is_recovered():
    assert math.isclose(normal_quantile(0.975), 1.959963984540054, rel_tol=1e-9)


def test_the_quantile_is_symmetric():
    assert math.isclose(normal_quantile(0.1), -normal_quantile(0.9), rel_tol=1e-9)


@pytest.mark.parametrize("p", [1e-6, 0.01, 0.2, 0.8, 0.99, 0.999999])
def test_matches_scipy_across_the_range(p: float):
    scipy_stats = pytest.importorskip("scipy.stats")
    assert math.isclose(normal_quantile(p), float(scipy_stats.norm.ppf(p)), rel_tol=1e-9,
                        abs_tol=1e-9)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.2])
def test_probabilities_outside_the_open_unit_interval_are_refused(p: float):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        normal_quantile(p)
