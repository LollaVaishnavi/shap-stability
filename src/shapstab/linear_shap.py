"""Why this file exists: for a linear model the exact Shapley attribution has a
closed form, so there is no reason to pull in a kernel approximator or the ``shap``
library and inherit its sampling noise. Under the standard feature-independence
assumption the attribution of feature i on row x is beta_i * (x_i - E[x_i]); on
standardised, centred features that is simply beta_i * z_i. Global importance is the
mean absolute attribution over a fixed evaluation set. Owning this one formula is
what lets the rest of the package treat importance as deterministic and attribute
every wobble to the coefficient, not to an explainer.

Reference for the closed form: Lundberg and Lee, "A Unified Approach to Interpreting
Model Predictions" (2017), and the LinearExplainer "interventional" case where
features are treated as independent.
"""

from __future__ import annotations

import numpy as np

from shapstab.model import ModelError


def local_shap_values(beta: np.ndarray, z_eval: np.ndarray) -> np.ndarray:
    """Exact per-row SHAP values for a linear model on standardised features.

    phi[r, i] = beta_i * z_eval[r, i]. Because z_eval is already centred, the
    attribution is just the scaled feature value, which is the exact Shapley value
    under feature independence rather than an approximation of it.
    """
    beta = np.asarray(beta, dtype=np.float64).reshape(-1)
    z_eval = np.asarray(z_eval, dtype=np.float64)
    if z_eval.ndim != 2:
        raise ModelError("z_eval must be 2-D (rows, features)")
    if beta.shape[0] != z_eval.shape[1]:
        raise ModelError("beta length must match the number of evaluation features")
    return z_eval * beta  # broadcasting over rows


def global_importance(beta: np.ndarray, z_eval: np.ndarray) -> np.ndarray:
    """Mean absolute SHAP value per feature over the evaluation set.

    This is the quantity a practitioner reads off a SHAP summary plot and then ranks
    on. For the linear case it factorises as |beta_i| * mean_r |z_eval[r, i]|, so the
    evaluation term is a fixed per-feature constant and all run-to-run movement comes
    from beta. That factorisation is the mechanism the whole tool rests on.
    """
    phi = local_shap_values(beta, z_eval)
    return np.abs(phi).mean(axis=0)


def importance_scale(z_eval: np.ndarray) -> np.ndarray:
    """The fixed per-feature evaluation constant mean_r |z_eval[r, i]|.

    Exposed so an experiment can show that importance is |beta_i| times a constant,
    and so a test can assert the factorisation holds exactly.
    """
    z_eval = np.asarray(z_eval, dtype=np.float64)
    if z_eval.ndim != 2:
        raise ModelError("z_eval must be 2-D (rows, features)")
    return np.abs(z_eval).mean(axis=0)
