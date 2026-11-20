"""Shared helpers for the experiments. Each experiment answers one question the README
asks, writes its result to docs/experiments/ as JSON, and prints a table. Keeping the
write and the layout here means every experiment reports the same way and the receipts
collector can read any of them without special cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from shapstab.config import load_policy
from shapstab.data import DatasetSpec, make_dataset
from shapstab.ranking import RankingReport, analyse_ranking
from shapstab.resample import resample_importances

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "experiments"
POLICY = load_policy(REPO_ROOT / "policy.toml")


def write_result(name: str, payload: dict[str, Any]) -> Path:
    """Write an experiment payload to docs/experiments/<name>.json, sorted and stable."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def default_report(n_resamples: int | None = None, trio_rho: float | None = None) -> RankingReport:
    """Analyse the built-in dataset under the shipped policy, with optional overrides."""
    spec_kwargs: dict[str, Any] = {"seed": POLICY.base_seed}
    if trio_rho is not None:
        spec_kwargs["trio_rho"] = trio_rho
    spec = DatasetSpec(**spec_kwargs)
    x, y = make_dataset(spec)
    imp = resample_importances(
        x,
        y,
        n_resamples=n_resamples or POLICY.n_resamples,
        base_seed=POLICY.base_seed,
        alpha=POLICY.ridge_alpha,
        feature_names=spec.feature_names,
    )
    return analyse_ranking(
        imp,
        top_k=POLICY.top_k,
        confidence=POLICY.confidence,
        tie_band=POLICY.tie_band,
        reliable_dominance=POLICY.reliable_dominance,
    )


def exchangeable_pair_dataset(seed: int, n_rows: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """A dataset whose first two features have genuinely equal importance.

    Both are symmetric noisy readings of one latent factor with the same coefficient, so
    the population cannot prefer either. Which one a single fit ranks higher is therefore
    pure sampling luck, which is exactly what the null-calibration experiment measures.
    """
    rng = np.random.default_rng(seed)
    n = n_rows
    latent = rng.standard_normal(n)
    w = float(np.sqrt(0.9))
    f0 = w * latent + np.sqrt(0.1) * rng.standard_normal(n)
    f1 = w * latent + np.sqrt(0.1) * rng.standard_normal(n)
    noise_feats = rng.standard_normal((n, 3))
    x = np.column_stack([f0, f1, noise_feats])
    y = f0 + f1 + 0.4 * noise_feats[:, 0] + rng.standard_normal(n)
    return x, y


def print_table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    """Print a plain, column-aligned table. No colour, no unicode box drawing."""
    widths = [len(h) for h in headers]
    str_rows = [[str(c) for c in row] for row in rows]
    for row in str_rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row, strict=True)]
    print(title)
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)))
    print("  ".join("-" * w for w in widths))
    for row in str_rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))
