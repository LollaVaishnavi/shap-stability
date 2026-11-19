"""Why this file exists: the same analysis has to be read two ways, by a person at a
terminal and by a browser that a screenshot tool photographs for the README. Both
renderings are built from the one RankingReport so they can never disagree. The HTML
carries a verdict badge whose colour is driven by a CSS class that the evidence tooling
reads back and refuses to ship if it is illegible, which is why the class names here are
stable and every class used has a rule in the stylesheet.
"""

from __future__ import annotations

import html

from shapstab.ranking import INCONCLUSIVE, STABLE, UNSTABLE, RankingReport

_VERDICT_CLASS = {
    STABLE: "verdict-stable",
    UNSTABLE: "verdict-unstable",
    INCONCLUSIVE: "verdict-inconclusive",
}

_VERDICT_LINE = {
    STABLE: "Every top boundary is reliably ordered across resamples at the set confidence.",
    UNSTABLE: "A top boundary is a coin flip, a reversal, or holds too weakly to trust.",
    INCONCLUSIVE: "A top boundary is unresolved; more resamples would settle it.",
}


def format_dominance(dominance: float, n: int) -> str:
    """Render a dominance probability, reporting a saturated value as a bound.

    A measured 1.000 over n resamples is not certainty, it is "at least 1 - 1/n", so it
    is shown that way. Treating a saturated proportion as exact is the single most common
    way a resampling result overclaims.
    """
    if dominance >= 1.0:
        return f">= {1.0 - 1.0 / n:.3f} (saturated at {n})"
    if dominance <= 0.0:
        return f"<= {1.0 / n:.3f} (saturated at {n})"
    return f"{dominance:.3f}"


def text_summary(report: RankingReport) -> str:
    """A compact terminal report: verdict, the awkward counts, then every boundary."""
    lines: list[str] = []
    lines.append(f"verdict: {report.verdict}")
    lines.append(_VERDICT_LINE[report.verdict])
    lines.append("")
    lines.append(
        f"judged depth: top {report.top_k}   resamples: {report.n_resamples}   "
        f"confidence: {report.confidence:.2f}"
    )
    lines.append(
        f"resolution floor: {report.resolution_floor:.3f} "
        f"(smallest dominance gap this budget can resolve)"
    )
    lines.append(
        f"of the single-fit top {report.top_k}, {report.stable_rank_count} keep their exact "
        f"rank in >= {report.confidence:.0%} of resamples"
    )
    lines.append(
        f"the exact single-fit top {report.top_k} order recurs in "
        f"{report.exact_topk_prob:.1%} of resamples; the same set in {report.set_topk_prob:.1%}"
    )
    if report.resamples_needed:
        lines.append(
            f"smallest unresolved gap: {report.min_unresolved_gap:.3f}; "
            f"resolving it needs about {report.resamples_needed} resamples"
        )
    lines.append("")
    lines.append("adjacent boundaries inside the judged depth:")
    header = f"  {'upper':>6} > {'lower':<6}  {'dominance':>12}  {'95% interval':>18}  status"
    lines.append(header)
    for b in report.boundaries:
        dom = format_dominance(b.dominance, report.n_resamples)
        ci = f"[{b.ci_low:.3f}, {b.ci_high:.3f}]"
        lines.append(f"  {b.upper_name:>6} > {b.lower_name:<6}  {dom:>12}  {ci:>18}  {b.status}")
    lines.append("")
    lines.append("per-feature rank interval (single-fit rank -> where the rank actually lands):")
    for f in sorted(report.features, key=lambda f: f.single_fit_rank):
        lines.append(
            f"  {f.name:>6}  single {f.single_fit_rank:>2}  median {f.median_rank:>4.1f}  "
            f"interval [{f.rank_low:>2}, {f.rank_high:>2}]  keeps rank {f.keep_rank_prob:.1%}"
        )
    return "\n".join(lines)


_STYLE = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  color: #1a1a1a; background: #ffffff; margin: 0; padding: 32px; }
h1 { font-size: 22px; margin: 0 0 4px 0; }
h2 { font-size: 16px; margin: 28px 0 8px 0; border-bottom: 1px solid #e3e3e3; padding-bottom: 4px; }
p.sub { color: #555; margin: 0 0 16px 0; }
.badge { display: inline-block; padding: 6px 14px; border-radius: 6px; font-weight: 700;
  font-size: 15px; letter-spacing: 0.3px; }
.verdict-stable { background: #1b7f3b; color: #ffffff; }
.verdict-unstable { background: #b11919; color: #ffffff; }
.verdict-inconclusive { background: #9a6a00; color: #ffffff; }
table { border-collapse: collapse; margin: 8px 0 4px 0; font-size: 13px; }
th, td { border: 1px solid #d7d7d7; padding: 5px 10px; text-align: right; }
th { background: #f2f2f2; }
td.name, th.name { text-align: left; font-variant-numeric: tabular-nums; }
.stat { font-size: 13px; color: #333; margin: 2px 0; }
.stat b { color: #111; }
.tie, .reversed, .weak { color: #b11919; font-weight: 600; }
.resolved { color: #1b7f3b; }
.underpowered { color: #9a6a00; }
"""


def html_report(report: RankingReport) -> str:
    """A self-contained HTML page. No external assets, so it renders offline."""
    cls = _VERDICT_CLASS[report.verdict]
    rows = []
    for b in report.boundaries:
        dom = html.escape(format_dominance(b.dominance, report.n_resamples))
        up = html.escape(b.upper_name)
        lo = html.escape(b.lower_name)
        rows.append(
            f"<tr><td class='name'>{up} &gt; {lo}</td>"
            f"<td>{dom}</td><td>[{b.ci_low:.3f}, {b.ci_high:.3f}]</td>"
            f"<td class='{b.status}'>{b.status}</td></tr>"
        )
    boundary_rows = "\n".join(rows)

    frows = []
    for f in sorted(report.features, key=lambda f: f.single_fit_rank):
        frows.append(
            f"<tr><td class='name'>{html.escape(f.name)}</td><td>{f.single_fit_rank}</td>"
            f"<td>{f.median_rank:.1f}</td><td>[{f.rank_low}, {f.rank_high}]</td>"
            f"<td>{f.keep_rank_prob:.1%}</td></tr>"
        )
    feature_rows = "\n".join(frows)

    conf = f"{report.confidence:.0%}"
    head_budget = (
        f"Judged depth <b>top {report.top_k}</b> over <b>{report.n_resamples}</b> "
        f"resamples at <b>{conf}</b> confidence. "
        f"Resolution floor <b>{report.resolution_floor:.3f}</b>."
    )
    head_stable = (
        f"Of the single-fit top {report.top_k}, <b>{report.stable_rank_count}</b> "
        f"keep their exact rank in at least {conf} of resamples."
    )
    head_order = (
        f"The exact single-fit top {report.top_k} order recurs in "
        f"<b>{report.exact_topk_prob:.1%}</b> of resamples; "
        f"the same set in {report.set_topk_prob:.1%}."
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>shap-stability report</title>
<style>{_STYLE}</style></head><body>
<h1>SHAP ranking stability</h1>
<p class="sub">Is the single-fit feature ranking real, or an artifact of one seed?</p>
<p><span class="badge {cls}">{html.escape(report.verdict)}</span></p>
<p class="stat">{html.escape(_VERDICT_LINE[report.verdict])}</p>
<h2>Headline</h2>
<p class="stat">{head_budget}</p>
<p class="stat">{head_stable}</p>
<p class="stat">{head_order}</p>
<h2>Adjacent boundaries</h2>
<table><thead><tr><th class="name">comparison</th><th>dominance</th><th>95% interval</th>
<th>status</th></tr></thead><tbody>
{boundary_rows}
</tbody></table>
<h2>Per-feature rank interval</h2>
<table><thead><tr><th class="name">feature</th><th>single-fit rank</th><th>median</th>
<th>95% interval</th><th>keeps rank</th></tr></thead><tbody>
{feature_rows}
</tbody></table>
</body></html>
"""
