"""Why this file exists: this is where the missing variance is manufactured. A single
fit gives one importance vector and no error bar. Here the training rows are bootstrap
resampled B times; each resample refits ridge and recomputes exact SHAP importance on
one fixed evaluation set. The output is a B-by-p matrix of importances, which is the
distribution every later claim is read from. The evaluation set and the standardiser
are fixed across resamples on purpose, so that the only thing moving from row to row of
that matrix is the coefficient estimate, which is precisely the seed sensitivity under
test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shapstab.linear_shap import global_importance
from shapstab.model import ModelError, Standardiser, fit_ridge, fit_standardiser


@dataclass(frozen=True)
class ImportanceSamples:
    """A B-by-p matrix of global SHAP importances and the single-fit baseline.

    ``samples[b, i]`` is feature i's importance under bootstrap resample b.
    ``single_fit`` is the importance vector from one fit on the full data, the vector
    a practitioner would actually publish and rank on. ``feature_names`` keeps every
    downstream table in words rather than column indices.
    """

    samples: np.ndarray
    single_fit: np.ndarray
    feature_names: tuple[str, ...]

    @property
    def n_resamples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.samples.shape[1])


def single_fit_importance(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, Standardiser]:
    """Importance from one fit on all rows, plus the fixed eval matrix and standardiser.

    Returned together because every resample must reuse the same standardiser and the
    same evaluation features; computing them here once keeps that invariant in one place.
    """
    standardiser = fit_standardiser(x)
    z = standardiser.transform(x)
    beta = fit_ridge(z, y, alpha)
    importance = global_importance(beta, z)
    return importance, z, standardiser


def resample_importances(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int,
    base_seed: int,
    alpha: float,
    feature_names: tuple[str, ...] | None = None,
) -> ImportanceSamples:
    """Bootstrap the training rows ``n_resamples`` times and collect importances.

    Resample b draws its row indices with a generator seeded ``base_seed + b``, so the
    whole matrix is one deterministic function of ``base_seed``. The evaluation matrix
    ``z`` (standardised full data) is held fixed; only the fitted beta changes, which is
    what isolates coefficient instability from evaluation-set noise.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if n_resamples < 2:
        raise ModelError("n_resamples must be at least 2 to form a distribution")

    single, z, _ = single_fit_importance(x, y, alpha)
    n_rows, n_features = x.shape

    if feature_names is None:
        feature_names = tuple(f"x{i:02d}" for i in range(n_features))
    if len(feature_names) != n_features:
        raise ModelError("feature_names length must match the number of features")

    samples = np.empty((n_resamples, n_features), dtype=np.float64)
    for b in range(n_resamples):
        rng = np.random.default_rng(base_seed + b)
        idx = rng.integers(0, n_rows, size=n_rows)
        beta_b = fit_ridge(z[idx], y[idx], alpha)
        samples[b] = global_importance(beta_b, z)

    return ImportanceSamples(samples=samples, single_fit=single, feature_names=tuple(feature_names))
