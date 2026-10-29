"""Working set of a coding session: how many distinct files an agent needs within a window of file operations, and
how soon it comes back to a file it used before. Sets the fade's parameters from data instead of by hand.

Source: the same local Claude Code logs as bench/replay_sessions.py (reads, edits and writes, in order; only aggregates
are written). Per session, the sequence of file operations; for windows of W operations, the share of the session's
files active in the window and the count of distinct files; for every return to a file, the gap (operations since
it was last used). A file the agent returns to after g operations would have been faded under a keep window < g.

Usage: python -I bench/working_set.py [projects_dir]   ->  bench/results/working_set.md
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_sessions import load_sessions  # noqa: E402

WINDOWS = (4, 8, 16, 32, 64)


def main():
    sessions = [evs for _, _, evs in load_sessions()]
    seqs = [[e["path"] for e in evs if e["op"] in ("read", "edit", "write")] for evs in sessions]
    seqs = [s for s in seqs if len(s) >= 8]
    gaps, distinct, share, files = [], {w: [] for w in WINDOWS}, {w: [] for w in WINDOWS}, []
    for s in seqs:
        n_files = len(set(s))
        files.append(n_files)
        last = {}
        for i, p in enumerate(s):
            if p in last:
                gaps.append(i - last[p])
            last[p] = i
        for w in WINDOWS:
            if len(s) < w:
                continue
            d = [len(set(s[a:a + w])) for a in range(0, len(s) - w + 1)]
            distinct[w].append(np.mean(d))
            share[w].append(np.mean(d) / n_files)
    gaps = np.array(gaps)
    ops = sum(len(s) for s in seqs)
    L = ["# Working set of coding sessions", "",
         f"{len(seqs)} sessions, {ops:,} file operations (reads, edits, writes), median {int(np.median(files))} distinct "
         f"files per session (max {max(files)}). Source: local agent logs (aggregates only).", "",
         "| window (file ops) | distinct files in the window (mean) | share of the session's files |", "|---|---|---|"]
    for w in WINDOWS:
        if distinct[w]:
            L.append(f"| {w} | {np.mean(distinct[w]):.1f} | {np.mean(share[w]):.0%} |")
    L += ["", f"Returns to a file used before: {len(gaps):,} ({len(gaps) / max(ops, 1):.0%} of operations).", "",
          "| gap since the file was last used (ops) | share of returns |", "|---|---|"]
    for lo, hi in ((1, 1), (2, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 10 ** 9)):
        L.append(f"| {lo}-{hi if hi < 10 ** 9 else 'more'} | {np.mean((gaps >= lo) & (gaps <= hi)):.0%} |")
    for q in (0.5, 0.8, 0.9, 0.95):
        L.append(f"\n{q:.0%} of returns come within {int(np.quantile(gaps, q))} operations." if q == 0.5 else
                 f"{q:.0%} of returns come within {int(np.quantile(gaps, q))} operations.")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "working_set.md")
    open(out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
