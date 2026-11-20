"""Experiment 3: what does a bigger resample budget actually buy?

Question the README asks: if a boundary is undecided, is it the data or the budget?
Answer: the smallest dominance gap resolvable scales as one over the square root of the
budget, so halving the floor costs four times the resamples. This is the price list the
verdict quotes next to every inconclusive boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import default_report, print_table, write_result  # noqa: E402


def main() -> None:
    budgets = [50, 100, 200, 500, 1000]
    rows = []
    series = []
    for b in budgets:
        report = default_report(n_resamples=b)
        underpowered = sum(1 for bd in report.boundaries if bd.status == "underpowered")
        series.append(
            {
                "n_resamples": b,
                "resolution_floor": report.resolution_floor,
                "underpowered_boundaries": underpowered,
            }
        )
        rows.append([str(b), f"{report.resolution_floor:.3f}", str(underpowered)])

    payload = {
        "series": series,
        "floor_at_50": series[0]["resolution_floor"],
        "floor_at_1000": series[-1]["resolution_floor"],
    }
    write_result("exp03_resolution_floor", payload)

    print_table(
        "Resolution floor (smallest resolvable dominance gap) versus budget",
        ["resamples", "floor", "underpowered boundaries"],
        rows,
    )


if __name__ == "__main__":
    main()
