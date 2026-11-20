"""Why this file exists: a stability check is only useful in a pipeline if its answer
is in the exit code, so a CI step or a pre-register hook can branch on it without
parsing text. The codes draw the distinction that matters here: a ranking judged
unstable (the tool did its job and the answer is "do not trust it") is not the same
event as the tool failing to run, and neither is the same as "not enough resamples to
decide yet". Each gets its own code, documented in --help and in the README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from shapstab.config import ConfigError, Policy, load_policy
from shapstab.data import DatasetSpec, make_dataset
from shapstab.model import ModelError
from shapstab.ranking import INCONCLUSIVE, STABLE, UNSTABLE, analyse_ranking
from shapstab.report import html_report, text_summary
from shapstab.resample import resample_importances

EXIT_STABLE = 0
EXIT_USAGE = 1  # the tool could not answer: bad policy, bad data, bad arguments
EXIT_UNSTABLE = 2  # the tool answered: the ranking is a seed artifact
EXIT_INCONCLUSIVE = 3  # the tool answered: undecided at this resample budget

_VERDICT_EXIT = {STABLE: EXIT_STABLE, UNSTABLE: EXIT_UNSTABLE, INCONCLUSIVE: EXIT_INCONCLUSIVE}


def _load_csv(path: Path, target: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Read a numeric CSV with the standard library, so there is no pandas at runtime.

    The header names the columns; one is the target and the rest are features. Every cell is
    parsed as a float, and a non-numeric cell is reported with its column rather than left to
    fail deep in the fit.
    """
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ModelError(f"{path} is empty") from None
        if target not in header:
            raise ModelError(f"target column {target!r} not found in {path}")
        target_idx = header.index(target)
        feature_cols = [c for i, c in enumerate(header) if i != target_idx]
        if not feature_cols:
            raise ModelError("the CSV has no feature columns besides the target")

        xs: list[list[float]] = []
        ys: list[float] = []
        for lineno, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                values = [float(cell) for cell in row]
            except ValueError as exc:
                raise ModelError(f"non-numeric value in {path} at line {lineno}") from exc
            ys.append(values[target_idx])
            xs.append([v for i, v in enumerate(values) if i != target_idx])

    if not xs:
        raise ModelError(f"{path} has a header but no data rows")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), tuple(feature_cols)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shapstab",
        description=(
            "Report whether a SHAP feature ranking is reproducible across bootstrap "
            "refits, or an artifact of one random seed."
        ),
        epilog=(
            "exit codes: 0 stable, 2 unstable (seed artifact), "
            "3 inconclusive (needs more resamples), 1 usage or data error."
        ),
    )
    parser.add_argument(
        "--policy", default="policy.toml", help="path to the policy TOML (default: policy.toml)"
    )
    parser.add_argument("--csv", help="analyse a CSV instead of the built-in synthetic dataset")
    parser.add_argument("--target", help="name of the target column when --csv is given")
    parser.add_argument(
        "--rho",
        type=float,
        help="override the synthetic trio correlation (ignored with --csv)",
    )
    parser.add_argument(
        "--resamples", type=int, help="override policy n_resamples (useful for demos)"
    )
    parser.add_argument("--top-k", type=int, help="override policy top_k")
    parser.add_argument("--html", help="also write the HTML report to this path")
    parser.add_argument("--json", help="also write the report as JSON to this path")
    parser.add_argument(
        "--quiet", action="store_true", help="print only the verdict line, not the full report"
    )
    return parser


def _effective_policy(policy: Policy, args: argparse.Namespace) -> Policy:
    from dataclasses import replace

    changes: dict = {}
    if args.resamples is not None:
        changes["n_resamples"] = args.resamples
    if args.top_k is not None:
        changes["top_k"] = args.top_k
    return replace(policy, **changes) if changes else policy


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        policy = load_policy(args.policy)
        policy = _effective_policy(policy, args)
    except ConfigError as exc:
        print(f"policy error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        if args.csv:
            if not args.target:
                raise ModelError("--target is required when --csv is given")
            x, y, names = _load_csv(Path(args.csv), args.target)
        else:
            spec_kwargs: dict = {"seed": policy.base_seed}
            if args.rho is not None:
                spec_kwargs["trio_rho"] = args.rho
            spec = DatasetSpec(**spec_kwargs)
            x, y = make_dataset(spec)
            names = spec.feature_names

        imp = resample_importances(
            x,
            y,
            n_resamples=policy.n_resamples,
            base_seed=policy.base_seed,
            alpha=policy.ridge_alpha,
            feature_names=names,
        )
    except (ModelError, ValueError) as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    report = analyse_ranking(
        imp,
        top_k=policy.top_k,
        confidence=policy.confidence,
        tie_band=policy.tie_band,
        reliable_dominance=policy.reliable_dominance,
    )

    if args.html:
        Path(args.html).write_text(html_report(report), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    if args.quiet:
        print(f"verdict: {report.verdict}")
    else:
        print(text_summary(report))

    return _VERDICT_EXIT[report.verdict]


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
