"""Screenshot the real HTML report with Chromium, and refuse to ship an illegible one.

The image in the README has to be the tool's actual output, not a mock-up, so it is
generated here from the same report the CLI writes. The one check that only a screenshot
tool can do is pixel legibility: it reads back the computed style of the verdict badge and
fails if the badge is invisible or has no size, which is how a CSS collision once shipped a
headline that rendered as green on green. The report is built fresh if missing, so the shot
is never of a stale file.
"""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from shapstab.config import load_policy
from shapstab.data import DatasetSpec, make_dataset
from shapstab.ranking import analyse_ranking
from shapstab.report import html_report
from shapstab.resample import resample_importances

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots" / "report.png"
MANIFEST = ROOT / "docs" / "screenshots" / "manifest.json"


def build_report_html() -> str:
    policy = load_policy(ROOT / "policy.toml")
    spec = DatasetSpec(seed=policy.base_seed)
    x, y = make_dataset(spec)
    imp = resample_importances(
        x, y, n_resamples=policy.n_resamples, base_seed=policy.base_seed,
        alpha=policy.ridge_alpha, feature_names=spec.feature_names,
    )
    report = analyse_ranking(
        imp, top_k=policy.top_k, confidence=policy.confidence,
        tie_band=policy.tie_band, reliable_dominance=policy.reliable_dominance,
    )
    return html_report(report)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = build_report_html()
    page_file = OUT.parent / "report.html"
    page_file.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 900, "height": 700})
        page.goto(page_file.as_uri())

        reading = page.evaluate(
            """() => {
              const badge = document.querySelector('.badge');
              if (!badge) return null;
              const s = getComputedStyle(badge);
              return {colour: s.color, background: s.backgroundColor,
                      visible: badge.offsetWidth > 0 && badge.offsetHeight > 0};
            }"""
        )
        if reading is None or not reading["visible"]:
            browser.close()
            raise SystemExit("verdict badge is missing or has zero size; refusing to ship the shot")
        if reading["colour"] == reading["background"]:
            browser.close()
            raise SystemExit("verdict badge text and background are the same colour")

        page.screenshot(path=str(OUT), full_page=True)
        browser.close()

    MANIFEST.write_text(
        json.dumps(
            {"output": str(OUT.relative_to(ROOT)), "source_report": "built from policy.toml",
             "badge": reading},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(ROOT)}; badge reading {reading}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
