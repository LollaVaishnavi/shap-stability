"""Experiment 1: how much of the single-fit top-k survives resampling?

Question the README asks: if you read a SHAP ranking off one fit and trust the order,
how often is that order the one you would have got? Answer: on the default data, almost
never for the exact order, and most top features do not hold their exact rank.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import default_report, print_table, write_result  # noqa: E402


def main() -> None:
    report = default_report()

    widest = max(
        (f for f in report.features if f.single_fit_rank <= report.top_k),
        key=lambda f: f.rank_span,
    )

    payload = {
        "verdict": report.verdict,
        "top_k": report.top_k,
        "n_resamples": report.n_resamples,
        "resolution_floor": report.resolution_floor,
        "resamples_needed": report.resamples_needed,
        "stable_rank_count": report.stable_rank_count,
        "exact_topk_prob": report.exact_topk_prob,
        "set_topk_prob": report.set_topk_prob,
        "widest_feature": widest.name,
        "widest_single_fit_rank": widest.single_fit_rank,
        "widest_rank_low": widest.rank_low,
        "widest_rank_high": widest.rank_high,
    }
    write_result("exp01_single_fit_overconfidence", payload)

    print_table(
        f"Single-fit top-{report.top_k} reproducibility over {report.n_resamples} resamples",
        ["metric", "value"],
        [
            ["verdict", report.verdict],
            ["top-k features keeping exact rank", f"{report.stable_rank_count}/{report.top_k}"],
            ["exact top-k order recurs", f"{report.exact_topk_prob:.1%}"],
            ["same top-k set recurs", f"{report.set_topk_prob:.1%}"],
            [
                f"{widest.name}: single-fit rank -> interval",
                f"{widest.single_fit_rank} -> [{widest.rank_low}, {widest.rank_high}]",
            ],
        ],
    )


if __name__ == "__main__":
    main()
