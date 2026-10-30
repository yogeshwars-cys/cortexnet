"""Render the README art (hero module, diagrams) from these HTML sources to docs/readme/*.png at 2x.

Needs Chrome or Edge and a network connection (Geist comes from Google Fonts). Usage: python -I render.py [name ...]
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
PAGES = {"hero": (960, 424), "architecture": (1040, 600), "workflow": (1040, 480)}   # CSS px; each page fixes its body to this
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", "google-chrome", "chromium"]


def browser():
    for b in BROWSERS:
        if os.path.exists(b) or shutil.which(b):
            return b
    sys.exit("no Chrome / Edge found")


def main(names):
    exe = browser()
    tmp_root = os.environ.get("CORTEX_RENDER_TMP") or tempfile.gettempdir()
    for name in names or PAGES:
        w, h = PAGES[name]
        src, out = os.path.join(HERE, name + ".html"), os.path.join(OUT, name + ".png")
        with tempfile.TemporaryDirectory(dir=tmp_root) as profile:
            subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                            f"--window-size={w},{h}", "--default-background-color=00000000",
                            "--virtual-time-budget=8000", f"--user-data-dir={profile}", f"--screenshot={out}",
                            "file:///" + src.replace("\\", "/")], check=True, capture_output=True, timeout=90)
        print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
