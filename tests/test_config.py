from __future__ import annotations

import math
from pathlib import Path

import pytest

from shapstab.config import ConfigError, Policy, load_policy

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_shipped_policy_file_loads_and_validates():
    policy = load_policy(REPO_ROOT / "policy.toml")
    assert policy.n_resamples >= 2
    assert 0.5 < policy.confidence < 1.0
    assert 0.5 < policy.reliable_dominance <= 1.0


def test_resolution_floor_is_one_over_sqrt_of_the_budget():
    policy = Policy(500, 1, 10, 0.95, 0.02, 0.9, 1.0)
    assert math.isclose(policy.resolution_floor, 1.0 / math.sqrt(500))


def test_a_missing_policy_file_is_a_config_error_not_a_crash():
    with pytest.raises(ConfigError, match="not found"):
        load_policy(REPO_ROOT / "does-not-exist.toml")


def test_a_missing_section_names_the_section(tmp_path: Path):
    path = tmp_path / "p.toml"
    path.write_text("[resample]\nn_resamples = 10\nbase_seed = 1\n")
    with pytest.raises(ConfigError, match=r"\[ranking\]"):
        load_policy(path)


def test_a_missing_key_names_the_key(tmp_path: Path):
    path = tmp_path / "p.toml"
    path.write_text(
        "[resample]\nn_resamples = 10\nbase_seed = 1\n"
        "[ranking]\ntop_k = 5\nconfidence = 0.95\ntie_band = 0.02\n"
        "[model]\nridge_alpha = 1.0\n"
    )
    with pytest.raises(ConfigError, match="reliable_dominance"):
        load_policy(path)


def test_too_few_resamples_cannot_form_a_distribution():
    with pytest.raises(ConfigError, match="n_resamples"):
        Policy(1, 1, 10, 0.95, 0.02, 0.9, 1.0)


def test_top_k_must_be_at_least_one():
    with pytest.raises(ConfigError, match="top_k"):
        Policy(10, 1, 0, 0.95, 0.02, 0.9, 1.0)


@pytest.mark.parametrize("confidence", [0.5, 1.0, 1.2, 0.4])
def test_confidence_must_be_strictly_between_half_and_one(confidence: float):
    with pytest.raises(ConfigError, match="confidence"):
        Policy(10, 1, 10, confidence, 0.02, 0.9, 1.0)


@pytest.mark.parametrize("band", [-0.1, 0.5, 0.7])
def test_tie_band_must_lie_in_zero_to_half(band: float):
    with pytest.raises(ConfigError, match="tie_band"):
        Policy(10, 1, 10, 0.95, band, 0.9, 1.0)


@pytest.mark.parametrize("reliable", [0.5, 0.4, 1.1])
def test_reliable_dominance_must_exceed_a_half_and_not_pass_one(reliable: float):
    with pytest.raises(ConfigError, match="reliable_dominance"):
        Policy(10, 1, 10, 0.95, 0.02, reliable, 1.0)


def test_negative_ridge_alpha_is_rejected():
    with pytest.raises(ConfigError, match="ridge_alpha"):
        Policy(10, 1, 10, 0.95, 0.02, 0.9, -1.0)


def test_a_policy_is_frozen_so_a_run_cannot_mutate_a_threshold():
    policy = Policy(10, 1, 10, 0.95, 0.02, 0.9, 1.0)
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raises FrozenInstanceError
        policy.top_k = 3  # type: ignore[misc]
