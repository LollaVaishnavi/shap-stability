from __future__ import annotations

import re

import numpy as np

from shapstab.ranking import analyse_ranking
from shapstab.report import format_dominance, html_report, text_summary


def _report(samples_factory, values, n=400, **kw):
    cols = {}
    for i, v in enumerate(values):
        cols[f"f{i}"] = np.full(n, v) + 1e-6 * np.sin(np.arange(n) + i)
    imp = samples_factory(cols)
    return analyse_ranking(
        imp,
        top_k=kw.get("top_k", len(values)),
        confidence=0.95,
        tie_band=0.02,
        reliable_dominance=0.9,
    )


def test_format_dominance_reports_a_saturated_high_value_as_a_bound():
    assert ">=" in format_dominance(1.0, 500)
    assert "500" in format_dominance(1.0, 500)


def test_format_dominance_reports_a_saturated_low_value_as_a_bound():
    assert "<=" in format_dominance(0.0, 500)


def test_format_dominance_shows_an_ordinary_value_plainly():
    assert format_dominance(0.734, 500) == "0.734"


def test_text_summary_leads_with_the_verdict(samples_factory):
    report = _report(samples_factory, [3.0, 2.0, 1.0])
    text = text_summary(report)
    assert text.splitlines()[0].startswith("verdict:")
    assert "resolution floor" in text


def test_text_summary_mentions_resamples_needed_when_a_boundary_is_underpowered(samples_factory):
    rng = np.random.default_rng(2)
    n = 36
    cols = {
        "f0": 3.0 + rng.standard_normal(n),
        "f1": 2.9 + rng.standard_normal(n),
        "f2": np.full(n, 0.1),
    }
    imp = samples_factory(cols)
    report = analyse_ranking(imp, top_k=3, confidence=0.95, tie_band=0.001, reliable_dominance=0.9)
    assert "resamples" in text_summary(report)


def test_html_report_carries_a_verdict_badge_with_a_class(samples_factory):
    report = _report(samples_factory, [3.0, 2.0, 1.0])
    page = html_report(report)
    assert "verdict-stable" in page
    assert report.verdict in page


def test_every_css_class_used_in_the_markup_has_a_rule_in_the_stylesheet(samples_factory):
    # The evidence screenshot is only trustworthy if the verdict badge is styled. A class
    # used in markup with no stylesheet rule is exactly how a badge renders invisible, so
    # this test catches that collision at the source rather than in a screenshot.
    rng = np.random.default_rng(3)
    n = 300
    base = rng.standard_normal(n)
    cols = {
        "f0": 5.0 + base + rng.standard_normal(n) * 0.5,
        "f1": 5.0 + base + rng.standard_normal(n) * 0.5,
        "f2": np.full(n, 1.0),
    }
    imp = samples_factory(cols)
    report = analyse_ranking(imp, top_k=3, confidence=0.95, tie_band=0.05, reliable_dominance=0.9)
    page = html_report(report)

    used = set(re.findall(r"class=['\"]([a-z0-9 \-]+)['\"]", page))
    class_tokens = {tok for group in used for tok in group.split()}
    style_block = page.split("<style>")[1].split("</style>")[0]
    defined = set(re.findall(r"\.([a-z0-9\-]+)\s*[,{]", style_block))
    # badge is a shared base class; every status/verdict token must be defined.
    missing = {c for c in class_tokens if c not in defined}
    assert not missing, f"classes used but never styled: {missing}"
