"""Build the Cortex CLI banner art from docs/cortex_logo.png (dev-time only; needs Pillow).

The brain becomes half-block pixel art: each character cell holds two pixels, the top one as the foreground of `▀`
and the bottom one as its background, both 24-bit colors sampled from the logo, so the bars and folds stay smooth.
Pixels darker than the logo's background are left out, so the terminal's own background shows through.
Writes cortexnet/assets/brain.ans (UTF-8, ANSI 24-bit color). The CORTEX wordmark is drawn at run time
(cortexnet/cli_ui.py), with a gradient that follows the logo's letters.
Usage: python scripts/make_banner.py [--cols 34]
"""
import argparse
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LOGO = os.path.join(REPO, "docs", "cortex_logo.png")
OUT = os.path.join(REPO, "cortexnet", "assets", "brain.ans")
BRAIN = (317, 35, 707, 367)          # the brain's bounding box in the logo (x0, y0, x1, y1), measured
DARK = 48                            # luminance below this is background (the logo's background is ~19)


def cell(top, bot):
    """One character for two vertically stacked pixels (None = transparent)."""
    fg = lambda c: f"\x1b[38;2;{c[0]};{c[1]};{c[2]}m"  # noqa: E731
    bg = lambda c: f"\x1b[48;2;{c[0]};{c[1]};{c[2]}m"  # noqa: E731
    if top is None and bot is None:
        return "\x1b[0m "
    if bot is None:
        return "\x1b[0m" + fg(top) + "▀"
    if top is None:
        return "\x1b[0m" + fg(bot) + "▄"
    return fg(top) + bg(bot) + "▀"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=34, help="banner width in characters (= pixels)")
    a = ap.parse_args()
    img = Image.open(LOGO).convert("RGB").crop(BRAIN)
    w, h = img.size
    rows = int(round(a.cols * h / w / 2))                 # a cell is ~2x as tall as wide: 2 pixels per row
    small = np.asarray(img.resize((a.cols, rows * 2), Image.LANCZOS)).astype(int).clip(0, 255)
    lum = small @ np.array([0.299, 0.587, 0.114])
    px = [[None if lum[y, x] < DARK else tuple(int(v) for v in small[y, x]) for x in range(a.cols)]
          for y in range(rows * 2)]
    lines = []
    for r in range(rows):
        line = "".join(cell(px[2 * r][x], px[2 * r + 1][x]) for x in range(a.cols))
        lines.append(line.rstrip() + "\x1b[0m")
    while lines and set(lines[-1].replace("\x1b[0m", "")) <= {" "}:
        lines.pop()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {a.cols} x {len(lines)} cells")


if __name__ == "__main__":
    main()
