"""Why this file exists: the verdict depends on a handful of numbers (how many
resamples, how deep the ranking is judged, what counts as resolved) and those
numbers are decisions, not constants. Keeping them in a commented ``policy.toml``
loaded here means a reviewer changes behaviour by editing documented policy rather
than by grepping the source for a magic number.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when a policy file is missing a key or carries an impossible value.

    It is a distinct type so the CLI can map a bad policy to its own exit code
    rather than conflating it with a bad dataset or an unstable ranking.
    """


@dataclass(frozen=True)
class Policy:
    """The full set of knobs that decide a stability verdict.

    Frozen because a run must be one deterministic function of its policy: nothing
    downstream is allowed to mutate a threshold halfway through.
    """

    n_resamples: int
    base_seed: int
    top_k: int
    confidence: float
    tie_band: float
    reliable_dominance: float
    ridge_alpha: float

    def __post_init__(self) -> None:
        if self.n_resamples < 2:
            raise ConfigError("n_resamples must be at least 2 to form a distribution")
        if self.top_k < 1:
            raise ConfigError("top_k must be at least 1")
        if not 0.5 < self.confidence < 1.0:
            raise ConfigError("confidence must lie strictly between 0.5 and 1.0")
        if not 0.0 <= self.tie_band < 0.5:
            raise ConfigError("tie_band must lie in [0.0, 0.5)")
        if not 0.5 < self.reliable_dominance <= 1.0:
            raise ConfigError("reliable_dominance must lie in (0.5, 1.0]")
        if self.ridge_alpha < 0.0:
            raise ConfigError("ridge_alpha must be non-negative")

    @property
    def resolution_floor(self) -> float:
        """Smallest dominance deviation from 0.5 this resample budget can resolve.

        The binomial standard error at p = 0.5 is 0.5 / sqrt(B), so a gap smaller
        than roughly 1 / sqrt(B) cannot be told from a coin flip no matter how the
        point estimate falls. Reported next to every verdict so the budget's reach
        is never implicit.
        """
        return 1.0 / (self.n_resamples**0.5)


_REQUIRED = {
    "resample": ("n_resamples", "base_seed"),
    "ranking": ("top_k", "confidence", "tie_band", "reliable_dominance"),
    "model": ("ridge_alpha",),
}


def load_policy(path: str | Path) -> Policy:
    """Load and validate a policy from a TOML file.

    Every required key is checked explicitly so a truncated or hand-edited file
    fails with the name of the missing key rather than with a KeyError deep in a
    later computation.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"policy file not found: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    for section, keys in _REQUIRED.items():
        if section not in raw:
            raise ConfigError(f"policy file is missing the [{section}] section")
        for key in keys:
            if key not in raw[section]:
                raise ConfigError(f"policy file is missing {section}.{key}")

    return Policy(
        n_resamples=int(raw["resample"]["n_resamples"]),
        base_seed=int(raw["resample"]["base_seed"]),
        top_k=int(raw["ranking"]["top_k"]),
        confidence=float(raw["ranking"]["confidence"]),
        tie_band=float(raw["ranking"]["tie_band"]),
        reliable_dominance=float(raw["ranking"]["reliable_dominance"]),
        ridge_alpha=float(raw["model"]["ridge_alpha"]),
    )
