"""Why this file exists: a matrix of importances is not yet an answer. The question a
reader actually has is "can I trust this ranking?", and answering it means turning the
distribution into three things: how wide each feature's rank wanders, whether one
feature reliably outranks the next, and a single verdict that does not overclaim. The
inference here is deliberately boundary-wise. A ranking is only as trustworthy as its
weakest adjacent comparison, so the verdict is decided by the least resolved boundary
inside the judged depth, and it separates a genuine tie (a coin flip no budget fixes)
from a comparison that merely needs more resamples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import norm

from shapstab.resample import ImportanceSamples

# Verdict strings, also used as the CLI's spoken outcome.
STABLE = "STABLE"
UNSTABLE = "UNSTABLE"
INCONCLUSIVE = "INCONCLUSIVE"


def rank_descending(importance: np.ndarray) -> np.ndarray:
    """Rank a single importance vector, 1 = most important.

    A stable sort breaks the measure-zero case of exactly equal importances by feature
    order, so the ranking is a deterministic function of the vector and never depends on
    sort implementation details.
    """
    importance = np.asarray(importance, dtype=np.float64).reshape(-1)
    order = np.argsort(-importance, kind="stable")
    ranks = np.empty(importance.shape[0], dtype=np.int64)
    ranks[order] = np.arange(1, importance.shape[0] + 1)
    return ranks


def rank_matrix(samples: np.ndarray) -> np.ndarray:
    """Rank every resample row, returning a B-by-p integer matrix of ranks."""
    return np.vstack([rank_descending(row) for row in samples])


def wilson_interval(successes: int, n: int, confidence: float) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because the proportions here run right up to
    0 and 1 (features that dominate in every resample), where the normal interval would
    spill past [0, 1] or collapse to zero width and call a 34-of-34 result certain.
    """
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    if n <= 0:
        raise ValueError("n must be positive")
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass(frozen=True)
class FeatureRank:
    """Rank behaviour of one feature across the resamples."""

    name: str
    single_fit_rank: int
    median_rank: float
    rank_low: int
    rank_high: int
    keep_rank_prob: float  # fraction of resamples at exactly single_fit_rank

    @property
    def rank_span(self) -> int:
        return self.rank_high - self.rank_low


@dataclass(frozen=True)
class Boundary:
    """A single-fit adjacent comparison and whether the resamples uphold it."""

    upper_name: str  # feature the single fit ranked higher
    lower_name: str
    dominance: float  # P(upper more important than lower)
    ci_low: float
    ci_high: float
    status: str  # "resolved", "weak", "reversed", "tie", "underpowered"


@dataclass(frozen=True)
class RankingReport:
    """Everything a verdict rests on, serialisable for the report and the receipts."""

    verdict: str
    top_k: int
    confidence: float
    reliable_dominance: float
    n_resamples: int
    resolution_floor: float
    stable_rank_count: int
    exact_topk_prob: float
    set_topk_prob: float
    min_unresolved_gap: float
    resamples_needed: int
    features: tuple[FeatureRank, ...]
    boundaries: tuple[Boundary, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["features"] = [asdict(f) for f in self.features]
        data["boundaries"] = [asdict(b) for b in self.boundaries]
        return data


def _feature_ranks(
    ranks: np.ndarray, single_ranks: np.ndarray, names: tuple[str, ...], confidence: float
) -> list[FeatureRank]:
    lo_pct = 100.0 * (1.0 - confidence) / 2.0
    hi_pct = 100.0 - lo_pct
    out: list[FeatureRank] = []
    for i, name in enumerate(names):
        col = ranks[:, i]
        sfr = int(single_ranks[i])
        out.append(
            FeatureRank(
                name=name,
                single_fit_rank=sfr,
                median_rank=float(np.median(col)),
                rank_low=int(np.percentile(col, lo_pct, method="nearest")),
                rank_high=int(np.percentile(col, hi_pct, method="nearest")),
                keep_rank_prob=float(np.mean(col == sfr)),
            )
        )
    return out


def _classify_boundary(
    dominance: float, ci: tuple[float, float], tie_band: float, reliable: float
) -> str:
    """Grade one adjacent comparison.

    The order of the checks encodes the policy: a confident reversal and a genuine tie
    are the two definitive failures, an interval that still straddles 0.5 is only
    underpowered, an interval clear of 0.5 but short of the reliability threshold is
    weak (significant but not trustworthy), and only an interval above the threshold is
    resolved.
    """
    low, high = ci
    if high < 0.5:
        return "reversed"
    if abs(dominance - 0.5) <= tie_band:
        return "tie"
    if low <= 0.5:
        return "underpowered"
    if low >= reliable:
        return "resolved"
    return "weak"


def analyse_ranking(
    imp: ImportanceSamples,
    *,
    top_k: int,
    confidence: float,
    tie_band: float,
    reliable_dominance: float,
) -> RankingReport:
    """Turn an importance distribution into rank intervals, boundaries and a verdict.

    The verdict is the status of the weakest boundary inside the single fit's top_k: a
    confident reversal, a genuine tie, or a boundary that holds better than chance but
    below the reliability threshold makes the whole ranking UNSTABLE; a boundary that is
    merely underpowered makes it INCONCLUSIVE; and only an all-resolved top_k is STABLE.
    This mirrors how a ranking is used: one unreliable swap near the top is enough to
    mislead a reader who trusted the order.
    """
    k = min(top_k, imp.n_features)
    ranks = rank_matrix(imp.samples)
    single_ranks = rank_descending(imp.single_fit)
    names = imp.feature_names
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))

    features = _feature_ranks(ranks, single_ranks, names, confidence)

    # Single-fit order, top k features by index.
    order = np.argsort(single_ranks, kind="stable")  # index of rank 1, rank 2, ...
    top_idx = order[:k]

    boundaries: list[Boundary] = []
    for r in range(k - 1):
        up, lo = int(top_idx[r]), int(top_idx[r + 1])
        wins = int(np.sum(imp.samples[:, up] > imp.samples[:, lo]))
        dominance = wins / imp.n_resamples
        ci = wilson_interval(wins, imp.n_resamples, confidence)
        status = _classify_boundary(dominance, ci, tie_band, reliable_dominance)
        boundaries.append(
            Boundary(
                upper_name=names[up],
                lower_name=names[lo],
                dominance=dominance,
                ci_low=ci[0],
                ci_high=ci[1],
                status=status,
            )
        )

    statuses = {b.status for b in boundaries}
    if {"reversed", "tie", "weak"} & statuses:
        verdict = UNSTABLE
    elif "underpowered" in statuses:
        verdict = INCONCLUSIVE
    else:
        verdict = STABLE

    # How many top-k features keep their exact single-fit rank often enough to trust.
    topk_names = {names[i] for i in top_idx}
    stable_rank_count = sum(
        1 for f in features if f.name in topk_names and f.keep_rank_prob >= confidence
    )

    # Reproducibility of the whole top-k, as an ordered list and as a set.
    single_topk_order = tuple(top_idx.tolist())
    single_topk_set = frozenset(single_topk_order)
    exact = 0
    setwise = 0
    for b in range(imp.n_resamples):
        rb = ranks[b]
        ob = np.argsort(rb, kind="stable")[:k]
        if tuple(ob.tolist()) == single_topk_order:
            exact += 1
        if frozenset(ob.tolist()) == single_topk_set:
            setwise += 1

    # The smallest dominance gap on a boundary that only more resamples would settle,
    # and the budget that would settle it. Weak and tie boundaries are excluded: more
    # resamples confirm them, they do not rescue them.
    unresolved = [b for b in boundaries if b.status == "underpowered"]
    if unresolved:
        min_gap = min(abs(b.dominance - 0.5) for b in unresolved)
        gap_for_budget = max(min_gap, 1.0 / imp.n_resamples)
        resamples_needed = int(np.ceil((z * 0.5 / gap_for_budget) ** 2))
    else:
        min_gap = 0.0
        resamples_needed = 0

    return RankingReport(
        verdict=verdict,
        top_k=k,
        confidence=confidence,
        reliable_dominance=reliable_dominance,
        n_resamples=imp.n_resamples,
        resolution_floor=1.0 / (imp.n_resamples**0.5),
        stable_rank_count=stable_rank_count,
        exact_topk_prob=exact / imp.n_resamples,
        set_topk_prob=setwise / imp.n_resamples,
        min_unresolved_gap=float(min_gap),
        resamples_needed=resamples_needed,
        features=tuple(features),
        boundaries=tuple(boundaries),
    )
