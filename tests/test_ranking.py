from __future__ import annotations

import numpy as np
import pytest

from shapstab.ranking import (
    INCONCLUSIVE,
    STABLE,
    UNSTABLE,
    Boundary,
    FeatureRank,
    _classify_boundary,
    analyse_ranking,
    rank_descending,
    rank_matrix,
    wilson_interval,
)


def test_rank_descending_puts_the_largest_importance_at_rank_one():
    ranks = rank_descending(np.array([0.1, 0.9, 0.5]))
    assert list(ranks) == [3, 1, 2]


def test_rank_descending_breaks_ties_by_feature_order_for_determinism():
    # Two equal importances: the earlier feature takes the better rank, every time.
    ranks = rank_descending(np.array([0.5, 0.5, 0.1]))
    assert list(ranks) == [1, 2, 3]


def test_rank_matrix_has_one_row_of_ranks_per_resample():
    samples = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert rank_matrix(samples).tolist() == [[2, 1], [1, 2]]


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(45, 100, 0.95)
    assert low < 0.45 < high
    assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_stays_inside_zero_one_when_saturated():
    low, high = wilson_interval(100, 100, 0.95)
    assert high <= 1.0
    assert low < 1.0  # a saturated proportion is not certainty


def test_wilson_interval_rejects_impossible_counts():
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(11, 10, 0.95)
    with pytest.raises(ValueError, match="n must be positive"):
        wilson_interval(0, 0, 0.95)


@pytest.mark.parametrize(
    "dominance, ci, expected",
    [
        (0.99, (0.95, 1.0), "resolved"),
        (0.70, (0.60, 0.80), "weak"),
        (0.10, (0.05, 0.20), "reversed"),
        (0.50, (0.44, 0.56), "tie"),
        (0.58, (0.49, 0.66), "underpowered"),
    ],
)
def test_classify_boundary_covers_every_status(dominance, ci, expected):
    assert _classify_boundary(dominance, ci, tie_band=0.02, reliable=0.90) == expected


def _constant_importances(values, n=400):
    # Each feature gets a near-constant importance with a tiny deterministic wobble, so the
    # ranking is whatever the values dictate and no boundary is a coin flip.
    cols = {}
    for i, v in enumerate(values):
        wobble = 1e-6 * np.sin(np.arange(n) + i)
        cols[f"f{i}"] = np.full(n, v) + wobble
    return cols


def test_a_clearly_separated_ranking_is_called_stable(samples_factory):
    imp = samples_factory(_constant_importances([5.0, 4.0, 3.0, 2.0, 1.0]))
    report = analyse_ranking(imp, top_k=5, confidence=0.95, tie_band=0.02, reliable_dominance=0.9)
    assert report.verdict == STABLE
    assert all(b.status == "resolved" for b in report.boundaries)


def test_two_features_that_swap_evenly_make_the_ranking_unstable(samples_factory):
    rng = np.random.default_rng(0)
    n = 600
    # f0 and f1 are exchangeable: each resample, one is slightly above the other at random.
    base = rng.standard_normal(n)
    cols = {
        "f0": 5.0 + base + rng.standard_normal(n) * 0.5,
        "f1": 5.0 + base + rng.standard_normal(n) * 0.5,
        "f2": np.full(n, 1.0),
    }
    imp = samples_factory(cols)
    report = analyse_ranking(imp, top_k=3, confidence=0.95, tie_band=0.05, reliable_dominance=0.9)
    assert report.verdict == UNSTABLE


def test_a_real_but_tiny_gap_with_too_few_resamples_is_inconclusive(samples_factory):
    rng = np.random.default_rng(1)
    n = 40  # deliberately small, so a real gap cannot be resolved
    cols = {
        "f0": 3.0 + rng.standard_normal(n),
        "f1": 2.9 + rng.standard_normal(n),
        "f2": np.full(n, 0.1),
    }
    imp = samples_factory(cols)
    report = analyse_ranking(imp, top_k=3, confidence=0.95, tie_band=0.001, reliable_dominance=0.9)
    assert report.verdict == INCONCLUSIVE
    assert report.resamples_needed > n


def test_the_resolution_floor_shrinks_as_the_budget_grows(samples_factory):
    imp_small = samples_factory(_constant_importances([3.0, 2.0], n=100))
    imp_large = samples_factory(_constant_importances([3.0, 2.0], n=900))
    kw = dict(top_k=2, confidence=0.95, tie_band=0.02, reliable_dominance=0.9)
    r_small = analyse_ranking(imp_small, **kw)
    r_large = analyse_ranking(imp_large, **kw)
    assert r_large.resolution_floor < r_small.resolution_floor


def test_top_k_is_clamped_to_the_feature_count(samples_factory):
    imp = samples_factory(_constant_importances([3.0, 2.0, 1.0]))
    report = analyse_ranking(imp, top_k=99, confidence=0.95, tie_band=0.02, reliable_dominance=0.9)
    assert report.top_k == 3
    assert len(report.boundaries) == 2


def test_a_stable_feature_keeps_its_rank_almost_always(samples_factory):
    imp = samples_factory(_constant_importances([5.0, 4.0, 3.0, 2.0, 1.0]))
    report = analyse_ranking(imp, top_k=5, confidence=0.95, tie_band=0.02, reliable_dominance=0.9)
    assert report.stable_rank_count == 5
    assert report.exact_topk_prob == 1.0
    assert report.set_topk_prob == 1.0


def test_report_serialises_to_plain_data_for_the_receipts(samples_factory):
    imp = samples_factory(_constant_importances([3.0, 2.0, 1.0]))
    report = analyse_ranking(imp, top_k=3, confidence=0.95, tie_band=0.02, reliable_dominance=0.9)
    data = report.to_dict()
    assert data["verdict"] == STABLE
    assert isinstance(data["features"], list)
    assert isinstance(data["boundaries"][0], dict)


def test_feature_rank_span_is_the_interval_width():
    fr = FeatureRank("a", 1, 2.0, 1, 4, 0.5)
    assert fr.rank_span == 3


def test_boundary_carries_its_comparison_and_status():
    b = Boundary("a", "b", 0.9, 0.85, 0.94, "resolved")
    assert b.upper_name == "a"
    assert b.status == "resolved"
