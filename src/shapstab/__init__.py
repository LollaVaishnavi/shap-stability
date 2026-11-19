"""shapstab: is a SHAP feature ranking real, or an artifact of one random seed?

A single model fit produces one SHAP attribution vector and therefore one feature
ranking. That ranking has no variance attached, so on its own it cannot say whether
feature A genuinely outranks feature B or merely happened to this time. This package
refits a linear model across bootstrap resamples, recomputes exact closed-form SHAP
importances on a fixed evaluation set for each refit, and reports the distribution
the single fit structurally lacks: per-feature rank intervals, pairwise dominance
probabilities, and a verdict that separates a stable ranking from a seed artifact.
"""

from shapstab.config import Policy, load_policy
from shapstab.ranking import RankingReport, analyse_ranking
from shapstab.resample import ImportanceSamples, resample_importances

__all__ = [
    "ImportanceSamples",
    "Policy",
    "RankingReport",
    "analyse_ranking",
    "load_policy",
    "resample_importances",
]
