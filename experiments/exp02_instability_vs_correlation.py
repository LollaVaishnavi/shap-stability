"""Experiment 2: does the instability come from the correlation, as claimed?

Question the README asks: is the wandering rank really caused by collinearity, or is it
just noise? Answer: sweep the trio correlation and watch the trio's mean rank-interval
width rise with it, while the independent tail stays put. This ties the symptom to the
mechanism.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import POLICY, default_report, print_table, write_result  # noqa: E402

TRIO = (2, 3, 4)
TAIL = (6, 7, 8, 9, 10, 11)


def _mean_span(report, indices_names):
    spans = [f.rank_span for f in report.features if f.name in indices_names]
    return sum(spans) / len(spans)


def main() -> None:
    names = tuple(f"x{i:02d}" for i in range(12))
    trio_names = {names[i] for i in TRIO}
    tail_names = {names[i] for i in TAIL}

    rhos = [0.0, 0.3, 0.6, 0.9, 0.97]
    rows = []
    series = []
    for rho in rhos:
        report = default_report(trio_rho=rho)
        trio_span = _mean_span(report, trio_names)
        tail_span = _mean_span(report, tail_names)
        series.append({"trio_rho": rho, "trio_mean_span": trio_span, "tail_mean_span": tail_span})
        rows.append([f"{rho:.2f}", f"{trio_span:.2f}", f"{tail_span:.2f}"])

    payload = {
        "n_resamples": POLICY.n_resamples,
        "series": series,
        "trio_span_low_rho": series[0]["trio_mean_span"],
        "trio_span_high_rho": series[-1]["trio_mean_span"],
    }
    write_result("exp02_instability_vs_correlation", payload)

    print_table(
        "Mean rank-interval width (95%) as trio correlation rises",
        ["trio_rho", "trio width", "tail width"],
        rows,
    )


if __name__ == "__main__":
    main()
