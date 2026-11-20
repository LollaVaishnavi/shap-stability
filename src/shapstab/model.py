"""Why this file exists: the instability this tool measures lives in the model's
coefficients, so the fit has to be exact and deterministic rather than a black box
with its own random state. A closed-form ridge solve in numpy gives both: the same
data always yields the same coefficients, byte for byte, and there is no solver seed
to confound with the bootstrap seed. Standardisation is folded in because exact SHAP
for a linear model is only comparable across features once they share a scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ModelError(ValueError):
    """Raised when a design matrix cannot support a fit (empty, non-finite, degenerate)."""


@dataclass(frozen=True)
class Standardiser:
    """Column means and standard deviations, fixed once on the full dataset.

    Fixing the standardiser (rather than refitting it per bootstrap) is deliberate:
    it pins the evaluation features so that the only thing varying across resamples
    is the coefficient estimate, which is the quantity under study.
    """

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


def fit_standardiser(x: np.ndarray) -> Standardiser:
    """Compute per-column mean and standard deviation, guarding a zero-variance column.

    A constant column has zero standard deviation; dividing by it would make every
    downstream importance NaN. Such a column carries no SHAP signal for a linear
    model anyway (its centred value is zero), so its scale is set to 1 and it
    contributes exactly zero importance.
    """
    x = _as_matrix(x)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return Standardiser(mean=mean, scale=scale)


def fit_ridge(x_std: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Return ridge coefficients with an intercept via the normal equations.

    beta = (Xc^T Xc + alpha I)^-1 Xc^T yc, where both X and y are centred first, solved
    with a linear solve rather than an explicit inverse. Centring X as well as y is what
    keeps the intercept out of the penalised coefficients; it matches scikit-learn's
    Ridge(fit_intercept=True) to machine precision. This matters here specifically
    because every bootstrap resample has non-zero column means even when the full data is
    centred, so a version that centred only y would fold each resample's mean offset into
    the coefficients and add a confound to the very variance this tool is measuring.
    Exact and deterministic: no iterative solver, no random state.
    """
    x_std = _as_matrix(x_std)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.shape[0] != x_std.shape[0]:
        raise ModelError("x and y must have the same number of rows")
    if alpha < 0.0:
        raise ModelError("alpha must be non-negative")

    x_centred = x_std - x_std.mean(axis=0)
    y_centred = y - y.mean()
    n_features = x_std.shape[1]
    gram = x_centred.T @ x_centred + alpha * np.eye(n_features)
    rhs = x_centred.T @ y_centred
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - alpha>0 keeps gram SPD
        raise ModelError("ridge normal equations are singular; raise ridge_alpha") from exc
    return beta


def _as_matrix(x: np.ndarray) -> np.ndarray:
    """Validate and coerce a design matrix to a finite 2-D float array."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ModelError("design matrix must be 2-D (rows, features)")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ModelError("design matrix must have at least one row and one column")
    if not np.all(np.isfinite(x)):
        raise ModelError("design matrix contains non-finite values")
    return x
