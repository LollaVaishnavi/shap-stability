"""Record a replay of the tool's real output as a GIF for the README.

This is a replay, not a screen recording, and the README says so. Each command is run once,
its real stdout, exit code and wall time captured, then an HTML terminal types the command
and reveals the captured output paced by that measured duration. Playwright records the page
to webm and ffmpeg transcodes it to a GIF and an MP4. Every line of terminal text is real
captured output; nothing is hand written. The manifest records every command with its exit
code and duration.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "docs" / "video"
GIF = VIDEO_DIR / "demo.gif"
MP4 = VIDEO_DIR / "demo.mp4"
MANIFEST = VIDEO_DIR / "manifest.json"

# Ordered as the argument, not the feature list: the verdict first, then the full report
# every demo shows, then the calibration no other tool reports, then the receipts.
COMMANDS = [
    ["shapstab", "--quiet"],
    ["shapstab", "--top-k", "6"],
    ["python", "experiments/exp04_null_calibration.py"],
    ["python", "tools/check_numbers.py"],
]


def run(cmd: list[str]) -> dict:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    duration = time.perf_counter() - start
    out = (proc.stdout + proc.stderr).rstrip("\n")
    return {"cmd": " ".join(cmd), "output": out, "exit": proc.returncode,
            "duration_ms": round(duration * 1000)}


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
body { margin:0; background:#0f1115; }
#term { font-family: Menlo, Consolas, monospace; font-size:15px; line-height:1.45;
  color:#e6e6e6; padding:18px 20px; white-space:pre-wrap; height:100vh; box-sizing:border-box;
  overflow:hidden; }
.prompt { color:#6ad36a; } .exit0 { color:#6ad36a; } .exitN { color:#ff7a7a; }
</style></head><body><div id="term"></div>
<script>
const segs = __SEGMENTS__;
const term = document.getElementById('term');
const sleep = ms => new Promise(r => setTimeout(r, ms));
function line(html) {
  const d=document.createElement('div'); d.innerHTML=html; term.appendChild(d);
}
async function type(cmd) {
  const d=document.createElement('div');
  d.innerHTML='<span class="prompt">$ </span>'; term.appendChild(d);
  for (const ch of cmd) { d.innerHTML += ch.replace('<','&lt;'); await sleep(22); }
}
async function run() {
  await sleep(500);
  for (const s of segs) {
    await type(s.cmd);
    await sleep(260);
    const lines = s.output.length ? s.output.split('\\n') : [];
    const per = Math.min(90, Math.max(12, Math.round((s.duration_ms*0.5)/(lines.length+1))));
    for (const ln of lines) {
      line(ln.replace(/&/g,'&amp;').replace(/</g,'&lt;'));
      term.scrollTop = term.scrollHeight;
      await sleep(per);
    }
    const cls = s.exit===0 ? 'exit0' : 'exitN';
    line('<span class="'+cls+'">[exit '+s.exit+']</span>');
    await sleep(700);
  }
  window.__done = true;
}
run();
</script></body></html>"""


def main() -> int:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    env_path = os.environ["PATH"]

    # Warm the import so the first timed segment does not pay for it.
    subprocess.run(["shapstab", "--quiet"], cwd=ROOT, capture_output=True, env={"PATH": env_path})

    segments = [run(cmd) for cmd in COMMANDS]
    page_html = _PAGE.replace("__SEGMENTS__", json.dumps(segments))
    page_file = VIDEO_DIR / "demo.html"
    page_file.write_text(page_html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        context = browser.new_context(
            viewport={"width": 860, "height": 560},
            record_video_dir=str(VIDEO_DIR),
            record_video_size={"width": 860, "height": 560},
        )
        page = context.new_page()
        page.goto(page_file.as_uri())
        page.wait_for_function("window.__done === true", timeout=60000)
        time.sleep(0.5)
        video_path = page.video.path()
        context.close()
        browser.close()

    webm = Path(video_path)
    palette = VIDEO_DIR / "palette.png"
    # The GIF is kept small (lower fps, narrower, a bounded palette) so it autoplays inline
    # on GitHub; the MP4 keeps the full width and frame rate for anyone who wants quality.
    gif_filter = "fps=10,scale=720:-2:flags=lanczos"
    mp4_scale = "scale=860:-2:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-i", str(webm), "-vf",
                    f"{gif_filter},palettegen=max_colors=128", str(palette)],
                   check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(webm), "-i", str(palette), "-lavfi",
                    f"{gif_filter} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3",
                    str(GIF)], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(webm), "-vf", mp4_scale, "-movflags",
                    "+faststart", "-pix_fmt", "yuv420p", str(MP4)],
                   check=True, capture_output=True)
    palette.unlink(missing_ok=True)
    webm.unlink(missing_ok=True)

    MANIFEST.write_text(json.dumps({"segments": segments}, indent=2) + "\n", encoding="utf-8")
    size_mb = GIF.stat().st_size / 1e6
    print(f"wrote {GIF.relative_to(ROOT)} ({size_mb:.1f} MB) and {MP4.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
