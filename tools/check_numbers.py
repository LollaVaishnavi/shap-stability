"""Fail the build when a document quotes a figure the code no longer produces. This is
the half of the receipts that a reader never sees run but relies on: it loads
docs/metrics.json, loads every checked document, and verifies each metric against the
prose by anchor phrase rather than by bare digits, so a stale sentence around a correct
number is still caught.

Forward check (fails the build): every anchored metric must appear, with its value
substituted, in at least one checked document. This is what breaks when a README figure
is edited to the wrong value.

Reverse check (warns only): any number in the prose that matches no metric is reported,
because it is a number nothing re-measures. It only warns, since policy constants and
structural numbers legitimately appear and failing on them would train a reader to ignore
the section.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "docs" / "metrics.json"

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE = re.compile(r"`[^`]*`")
_TAG = re.compile(r"<[^>]+>")
_LINK_TARGET = re.compile(r"\]\([^)]*\)")  # the (url) part of a markdown link or image
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def strip_code(text: str) -> str:
    """Remove fenced blocks, inline code spans and HTML tags, leaving prose only.

    An example invocation is allowed to contain a made-up row count, so the checker must
    not read numbers out of code. HTML tags are dropped so an attribute value such as a
    width is not mistaken for a published figure.
    """
    text = _FENCE.sub(" ", text)
    text = _INLINE.sub(" ", text)
    text = _LINK_TARGET.sub("] ", text)  # drop URLs so a badge value is not read as prose
    text = _TAG.sub(" ", text)
    # Collapse whitespace so an anchor phrase that wraps across a line still matches. A
    # multi-word anchor straddling a newline in wrapped markdown was a real false failure.
    text = re.sub(r"\s+", " ", text)
    return text


def value_candidates(value: object) -> list[str]:
    """Every plausible string rendering of a metric value, so prose spelling can vary."""
    if isinstance(value, bool):
        return [str(value)]
    if isinstance(value, int):
        return [str(value), f"{value:.1f}"]
    if isinstance(value, float):
        out = {repr(value), str(value)}
        for places in (0, 1, 2, 3):
            out.add(f"{value:.{places}f}")
        # An integral float should also match the bare integer spelling.
        if value == int(value):
            out.add(str(int(value)))
        return sorted(out)
    return [str(value)]


def anchor_matches(prose: str, templates: list[str], value: object) -> bool:
    for template in templates:
        for candidate in value_candidates(value):
            if template.format(candidate) in prose:
                return True
    return False


def forward_check(prose_by_doc: dict[str, str], payload: dict) -> list[str]:
    metrics = payload["metrics"]
    anchors = payload["anchors"]
    failures = []
    for name, templates in anchors.items():
        if name not in metrics:
            failures.append(f"anchor for unknown metric: {name}")
            continue
        value = metrics[name]
        if not any(anchor_matches(prose, templates, value) for prose in prose_by_doc.values()):
            rendered = value_candidates(value)[0]
            failures.append(
                f"{name} = {value}: no document uses an anchor phrase "
                f"(for example {templates[0].format(rendered)!r})"
            )
    return failures


def reverse_check(prose_by_doc: dict[str, str], payload: dict) -> list[str]:
    metric_strings: set[str] = set()
    for value in payload["metrics"].values():
        metric_strings.update(value_candidates(value))
    warnings = []
    for doc, prose in prose_by_doc.items():
        for match in _NUMBER.finditer(prose):
            token = match.group(0)
            if token in metric_strings:
                continue
            warnings.append(f"{doc}: number {token!r} matches no metric")
    return warnings


def main() -> int:
    if not METRICS_PATH.is_file():
        print("docs/metrics.json not found; run tools/collect_metrics.py first", file=sys.stderr)
        return 2
    payload = json.loads(METRICS_PATH.read_text())

    prose_by_doc: dict[str, str] = {}
    for rel in payload["checked_documents"]:
        path = ROOT / rel
        if not path.is_file():
            print(f"checked document missing: {rel}", file=sys.stderr)
            return 2
        prose_by_doc[rel] = strip_code(path.read_text(encoding="utf-8"))

    failures = forward_check(prose_by_doc, payload)
    warnings = reverse_check(prose_by_doc, payload)

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if failures:
        print(f"\nFAIL: {len(failures)} figure(s) are not anchored in the prose:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"ok: {len(payload['anchors'])} anchored figures verified across "
          f"{len(prose_by_doc)} document(s); {len(warnings)} reverse-check warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
