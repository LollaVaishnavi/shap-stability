"""Render the architecture diagram from the README's mermaid source to a committed SVG.

GitHub cannot be relied on to render a mermaid fence, so the picture is produced here and
embedded with <img src>. Three traps are handled: htmlLabels are turned off (so the file is
valid XML and loads as an image), the root element is given explicit width and height from
the viewBox (so a browser has an aspect ratio), and a light background is painted (so edge
labels are legible in a dark theme). The mermaid source is read from the README itself, so
the picture can never drift from the source printed next to it.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OUT = ROOT / "docs" / "diagrams" / "architecture.svg"
MANIFEST = ROOT / "docs" / "diagrams" / "manifest.json"
CHROME = "/usr/bin/google-chrome"

_SOURCE = re.compile(r"```mermaid-source\n(.*?)```", re.DOTALL)


def extract_source() -> str:
    match = _SOURCE.search(README.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("no ```mermaid-source fence found in README.md")
    return match.group(1).strip() + "\n"


def preserve_label_whitespace(svg: str) -> str:
    """Keep the spaces between words in multi-word labels.

    With htmlLabels off, mermaid splits a label into one tspan per word and carries the
    space as a leading space on each tspan. SVG collapses leading whitespace unless told
    not to, so without this a renderer shows "datasetorCSV". Marking every text element
    xml:space="preserve" keeps the spaces in every renderer, browser and rasteriser alike.
    """
    return re.sub(r"<text(?![\w-])", '<text xml:space="preserve"', svg)


def set_intrinsic_dimensions(svg: str) -> str:
    """Replace width="100%" with explicit pixel width and height taken from the viewBox."""
    viewbox = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    if not viewbox:
        raise SystemExit("rendered SVG has no viewBox to size from")
    width, height = round(float(viewbox.group(1))), round(float(viewbox.group(2)))
    svg = re.sub(r'\swidth="100%"', f' width="{width}" height="{height}"', svg, count=1)
    return svg


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    source = extract_source()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mmd = tmp_path / "diagram.mmd"
        mmd.write_text(source, encoding="utf-8")
        raw_svg = tmp_path / "diagram.svg"
        (tmp_path / "puppeteer.json").write_text(
            json.dumps({"args": ["--no-sandbox", "--disable-gpu"], "executablePath": CHROME})
        )
        (tmp_path / "mermaid.json").write_text(
            json.dumps({"htmlLabels": False, "flowchart": {"htmlLabels": False}})
        )
        subprocess.run(
            [
                "mmdc",
                "-i", str(mmd),
                "-o", str(raw_svg),
                "-c", str(tmp_path / "mermaid.json"),
                "-p", str(tmp_path / "puppeteer.json"),
                "-b", "white",
            ],
            check=True,
            env={"PUPPETEER_EXECUTABLE_PATH": CHROME, "PATH": __import__("os").environ["PATH"]},
        )
        svg = raw_svg.read_text(encoding="utf-8")

    if "foreignObject" in svg:
        raise SystemExit("SVG contains foreignObject: htmlLabels were not turned off")
    svg = preserve_label_whitespace(svg)
    svg = set_intrinsic_dimensions(svg)
    OUT.write_text(svg, encoding="utf-8")

    MANIFEST.write_text(
        json.dumps(
            {"source": "README.md ```mermaid-source", "output": str(OUT.relative_to(ROOT)),
             "html_labels": False, "background": "white"},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
