"""Re-measure every published figure and write docs/metrics.json. Nothing is typed by
hand: the test count and coverage come from the machine-readable pytest reports, and the
headline numbers come from the experiments' own JSON output. This is one half of the
receipts; tools/check_numbers.py is the other, and the Makefile runs them in order.

Usage:
    python tools/collect_metrics.py            # run the suite and the experiments
    python tools/collect_metrics.py --skip-tests   # read existing reports (used in CI)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EXP_DIR = ROOT / "docs" / "experiments"
METRICS_PATH = ROOT / "docs" / "metrics.json"

EXPERIMENTS = [
    "exp01_single_fit_overconfidence",
    "exp02_instability_vs_correlation",
    "exp03_resolution_floor",
    "exp04_null_calibration",
]


def _run_tests() -> None:
    REPORTS.mkdir(exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"--junitxml={REPORTS / 'junit.xml'}",
            "--cov=shapstab",
            f"--cov-report=json:{REPORTS / 'coverage.json'}",
        ],
        cwd=ROOT,
        check=True,
    )


def _test_count() -> int:
    root = ET.parse(REPORTS / "junit.xml").getroot()
    suites = root.findall("testsuite") or [root]
    total = 0
    for suite in suites:
        tests = int(suite.get("tests", 0))
        skipped = int(suite.get("skipped", 0))
        total += tests - skipped
    return total


def _coverage_pct() -> float:
    data = json.loads((REPORTS / "coverage.json").read_text())
    return float(data["totals"]["percent_covered"])


def _run_experiments() -> None:
    for name in EXPERIMENTS:
        subprocess.run([sys.executable, "-I", str(ROOT / "experiments" / f"{name}.py")],
                       cwd=ROOT, check=True)


def _exp(name: str) -> dict:
    return json.loads((EXP_DIR / f"{name}.json").read_text())


def _pct(x: float, places: int = 1) -> float:
    return round(100.0 * x, places)


def build_metrics() -> dict:
    e1 = _exp("exp01_single_fit_overconfidence")
    e2 = _exp("exp02_instability_vs_correlation")
    e3 = _exp("exp03_resolution_floor")
    e4 = _exp("exp04_null_calibration")

    metrics = {
        "tests_total": _test_count(),
        "coverage_line_pct": round(_coverage_pct()),
        "verdict": e1["verdict"],
        "top_k": e1["top_k"],
        "n_resamples": e1["n_resamples"],
        "resolution_floor": round(e1["resolution_floor"], 3),
        "resamples_needed": e1["resamples_needed"],
        "stable_rank_count": e1["stable_rank_count"],
        "exact_topk_prob_pct": _pct(e1["exact_topk_prob"]),
        "set_topk_prob_pct": _pct(e1["set_topk_prob"]),
        "widest_feature": e1["widest_feature"],
        "widest_single_fit_rank": e1["widest_single_fit_rank"],
        "widest_rank_low": e1["widest_rank_low"],
        "widest_rank_high": e1["widest_rank_high"],
        "trio_width_low_rho": round(e2["trio_span_low_rho"], 2),
        "trio_width_high_rho": round(e2["trio_span_high_rho"], 2),
        "floor_at_50": round(e3["floor_at_50"], 3),
        "floor_at_1000": round(e3["floor_at_1000"], 3),
        "null_single_fit_winner_pct": _pct(e4["single_fit_f0_wins_rate"]),
        "null_within_confident_pct": int(round(100.0 * e4["within_dataset_confident_rate"])),
    }

    # Anchor phrases: the exact wording a document must use with the value substituted in.
    # Alternatives, any one matching is enough. Kept short so they survive line wrapping.
    anchors = {
        "tests_total": ["{} tests"],
        "coverage_line_pct": ["{}% line coverage", "{} percent line coverage"],
        "n_resamples": ["{} resamples", "{} bootstrap"],
        "resolution_floor": ["floor of {}", "floor is {}", "floor {}"],
        "resamples_needed": ["about {} resamples", "{} resamples to"],
        "stable_rank_count": ["{} of the top", "only {} keep", "{} keep their exact"],
        "top_k": ["top {}", "{} features by"],
        "exact_topk_prob_pct": ["{}% of resamples", "recurs in {}%", "in {}% of"],
        "set_topk_prob_pct": ["set in {}%", "same set in {}%", "{}% of resamples"],
        "widest_rank_low": ["ranks {} to", "from {}st", "{} to "],
        "widest_rank_high": ["to {}th", "to {} across", "to {} under"],
        "trio_width_low_rho": ["{} at no correlation", "from {} ", "{} ranks wide"],
        "trio_width_high_rho": ["to {} at", "{} at the", "to {} ranks"],
        "floor_at_50": ["{} at 50", "{} with 50"],
        "floor_at_1000": ["{} at 1000", "{} with 1000"],
        "null_single_fit_winner_pct": ["{}% of", "winner in {}%", "{}% of the time"],
        "null_within_confident_pct": ["{}% of", "in {}% of", "{}% of those"],
    }

    checked = ["README.md", "docs/adr/ADR-003-dominance-and-resolution-floor.md"]
    return {
        "metrics": metrics,
        "anchors": anchors,
        "checked_documents": checked,
        "note": "every value here is produced by running the suite and the four experiments",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-measure published figures.")
    parser.add_argument("--skip-tests", action="store_true",
                        help="read existing pytest reports instead of running them (CI)")
    parser.add_argument("--skip-experiments", action="store_true",
                        help="read existing experiment JSON instead of re-running")
    args = parser.parse_args()

    if not args.skip_tests:
        _run_tests()
    if not args.skip_experiments:
        _run_experiments()

    payload = build_metrics()
    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {METRICS_PATH.relative_to(ROOT)} with {len(payload['metrics'])} metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
