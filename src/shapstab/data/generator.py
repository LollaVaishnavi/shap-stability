"""Why this file exists: the point the repository makes needs data where a single-fit
ranking looks confident and is not, and the honest way to get that is to build the
instability in on purpose and label it. The generator plants a block of mutually
correlated features that share the signal. Any linear fit must split one true effect
across the block, and which member gets the credit is decided by the bootstrap draw,
so the block's internal ranking is a seed artifact by construction. Independent
features outside the block keep a stable rank and act as the control.

Everything is a pure function of the seed: same seed, byte-identical matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class DataError(ValueError):
    """Raised when a dataset specification is self-contradictory."""


@dataclass(frozen=True)
class DatasetSpec:
    """A reproducible regression problem with three known groups of features.

    The layout is built so a single fit produces a confident ranking that resampling
    then contradicts in three distinct, labelled ways:

    - the **pair** (features 0 and 1): two near-duplicate sensors, each a noisy reading of
      one shared quantity at correlation ``pair_rho``. They are exchangeable, carry the
      most signal and split it, so which one ranks higher is a coin flip. A genuine tie.
    - the **trio** (features 2, 3 and 4): a moderately correlated cluster that shares its
      signal, so the members swap order under resampling without being outright ties.
      This produces weak boundaries: an order that holds more often than not, but not
      reliably.
    - the **tail** (features 5 onward): independent, with coefficients that decay, so
      each has a clear and reproducible rank. This is the control that proves the tool
      calls a stable ranking stable.

    ``feature_names`` is derived so reports read in words rather than column indices.
    """

    n_rows: int = 1200
    n_features: int = 12
    pair_coef: float = 1.6
    pair_rho: float = 0.995
    trio_rho: float = 0.9
    trio_coef: float = 1.0
    tail_coef: float = 0.9
    noise_sd: float = 1.0
    seed: int = 20260601
    feature_names: tuple[str, ...] = field(default=())

    # Fixed group layout. The structure, not the counts, is what the story needs, so the
    # group sizes are constants rather than knobs: features 0-1 pair, 2-4 trio, 5+ tail.
    PAIR = (0, 1)
    TRIO = (2, 3, 4)

    def __post_init__(self) -> None:
        if self.n_rows < 2:
            raise DataError("n_rows must be at least 2")
        if self.n_features < 6:
            raise DataError("n_features must be at least 6 (pair + trio + one tail feature)")
        if not -1.0 < self.trio_rho < 1.0:
            raise DataError("trio_rho must lie strictly in (-1, 1)")
        if not -1.0 < self.pair_rho < 1.0:
            raise DataError("pair_rho must lie strictly in (-1, 1)")
        if self.noise_sd < 0.0:
            raise DataError("noise_sd must be non-negative")
        if self.pair_coef <= 0.0 or self.trio_coef <= 0.0 or self.tail_coef <= 0.0:
            raise DataError("group coefficients must be positive")
        if not self.feature_names:
            names = tuple(f"x{i:02d}" for i in range(self.n_features))
            object.__setattr__(self, "feature_names", names)
        if len(self.feature_names) != self.n_features:
            raise DataError("feature_names length must equal n_features")

    @property
    def true_coef(self) -> np.ndarray:
        """The data-generating coefficients. The tool never sees these.

        The pair and trio members each carry their group coefficient; the tail decays
        from ``tail_coef`` so it has a genuine, resolvable internal order.
        """
        coef = np.zeros(self.n_features, dtype=np.float64)
        coef[list(self.PAIR)] = self.pair_coef
        coef[list(self.TRIO)] = self.trio_coef
        tail_start = self.TRIO[-1] + 1
        for offset, j in enumerate(range(tail_start, self.n_features)):
            coef[j] = self.tail_coef / (offset + 1)
        return coef


def make_dataset(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for a spec. Pure function of ``spec.seed``.

    Correlated groups are drawn from shared latent factors rather than by inverting a
    covariance matrix, which keeps the construction transparent and exactly reproducible.
    Both members of a correlated group get their own idiosyncratic noise around the shared
    latent, so the members are exchangeable and no member is secretly the better copy.
    """
    rng = np.random.default_rng(spec.seed)
    n, p = spec.n_rows, spec.n_features
    x = np.empty((n, p), dtype=np.float64)

    # The pair: two symmetric noisy readings of one shared quantity.
    pair_latent = rng.standard_normal(n)
    wp = float(np.sqrt(spec.pair_rho))
    for j in spec.PAIR:
        x[:, j] = wp * pair_latent + np.sqrt(1.0 - spec.pair_rho) * rng.standard_normal(n)

    # The trio: a shared latent factor mixed to the target correlation.
    shared = rng.standard_normal(n)
    w = float(np.sqrt(spec.trio_rho))
    for j in spec.TRIO:
        x[:, j] = w * shared + np.sqrt(1.0 - spec.trio_rho) * rng.standard_normal(n)

    # The tail: independent features.
    for j in range(spec.TRIO[-1] + 1, p):
        x[:, j] = rng.standard_normal(n)

    y = x @ spec.true_coef + spec.noise_sd * rng.standard_normal(n)
    return x, y
