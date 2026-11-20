"""Render the README the way GitHub does and assert every image actually loads.

A broken image or an unrendered diagram is the first thing a reviewer notices, and
python-markdown disagrees with GitHub about exactly the constructs a designed README uses.
So the page is rendered with cmark-gfm, the library GitHub itself uses, loaded in Chromium,
and every <img> is checked for a non-zero naturalWidth. This is a dev-time check, kept out
of the pure-Python verify path.
"""

from __future__ import annotations

from pathlib import Path

import cmarkgfm
from cmarkgfm.cmark import Options
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def main() -> int:
    html = cmarkgfm.markdown_to_html_with_extensions(
        README.read_text(encoding="utf-8"),
        options=Options.CMARK_OPT_UNSAFE,
        extensions=["table", "autolink", "strikethrough", "tagfilter"],
    )
    # A <base> pointing at the repo root so the README's relative image paths resolve the
    # way they do on GitHub, wherever the preview file itself is written.
    base = ROOT.as_uri() + "/"
    page_file = ROOT / "docs" / "readme-preview.html"
    page_file.write_text(
        f"<!doctype html><meta charset='utf-8'><base href='{base}'><body>{html}",
        encoding="utf-8",
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        page.goto(page_file.as_uri())
        page.wait_for_timeout(500)
        images = page.evaluate(
            """() => Array.from(document.images).map(i => ({src: i.getAttribute('src'),
                 w: i.naturalWidth}))"""
        )
        browser.close()

    page_file.unlink(missing_ok=True)
    # Only committed local images are the repository's responsibility; a remote badge may be
    # blocked by an offline runner and is not a defect in the README.
    local = [i for i in images if not str(i["src"]).startswith("http")]
    broken = [i for i in local if not i["w"]]
    for img in images:
        scope = "remote" if str(img["src"]).startswith("http") else "local"
        print(f"  naturalWidth {img['w']:>5}  ({scope})  {img['src']}")
    if broken:
        print(f"\nFAIL: {len(broken)} local image(s) did not resolve")
        return 1
    print(f"ok: {len(local)} local image(s) resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
