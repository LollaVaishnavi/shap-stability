from __future__ import annotations

import numpy as np
import pytest

from shapstab.linear_shap import global_importance, importance_scale, local_shap_values
from shapstab.model import ModelError


def test_local_shap_value_is_coefficient_times_feature():
    beta = np.array([2.0, -3.0])
    z = np.array([[1.0, 1.0], [2.0, 0.5]])
    phi = local_shap_values(beta, z)
    assert np.allclose(phi, np.array([[2.0, -3.0], [4.0, -1.5]]))


def test_global_importance_factorises_into_coefficient_and_a_fixed_scale():
    # The property the whole tool rests on: importance_i = |beta_i| * mean|z_i|. If this
    # identity ever breaks, instability can no longer be attributed to the coefficient.
    rng = np.random.default_rng(3)
    z = rng.standard_normal((300, 4))
    beta = np.array([1.5, -2.0, 0.0, 4.0])
    direct = global_importance(beta, z)
    factorised = np.abs(beta) * importance_scale(z)
    assert np.allclose(direct, factorised)


def test_a_zero_coefficient_feature_has_zero_importance():
    z = np.random.default_rng(4).standard_normal((50, 3))
    beta = np.array([0.0, 1.0, 2.0])
    imp = global_importance(beta, z)
    assert imp[0] == 0.0


def test_local_shap_rejects_a_beta_eval_width_mismatch():
    with pytest.raises(ModelError, match="beta length"):
        local_shap_values(np.array([1.0, 2.0]), np.zeros((5, 3)))


def test_local_shap_rejects_a_one_dimensional_eval_matrix():
    with pytest.raises(ModelError, match="2-D"):
        local_shap_values(np.array([1.0]), np.zeros(5))


def test_importance_scale_rejects_a_one_dimensional_eval_matrix():
    with pytest.raises(ModelError, match="2-D"):
        importance_scale(np.zeros(5))
