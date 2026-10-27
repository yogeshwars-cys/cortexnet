"""Does the agent's own reasoning before a tool call predict which file it opens next?

From the local Claude Code logs (aggregates only). A transition is an access to file B (Read, Edit, or a path in a
shell command) whose previous file access was a different file. Candidates: every source file currently on disk in
the directories the session has touched so far (cap 300), plus files touched before. Definitions come from the file
on disk, as a live index would see them. The label is the file that was opened; it is never derived from features.
The snapshot is the model's thinking + text written since the previous tool call.

Rankers: random, recency (LRU; never-touched files rank last), snapshot names a symbol defined in the file,
snapshot names the file, and logistic mixes fitted on the first half of sessions (by time), tested on the second.
Reported separately for re-opens (file touched earlier) and first-time opens.
Usage: python -I bench/snapshot_signal.py [projects_dir]
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortexnet import symbols  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), ".claude", "projects")
EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".java", ".kt",
       ".dart", ".md", ".html", ".css", ".json", ".sh", ".toml", ".yaml", ".yml", ".txt"}
PATH = re.compile(r"""["']?((?:[A-Za-z]:[\\/]|/[a-zA-Z]/|~/|\.{0,2}/)?[\w.\\/@+-]*\.(?:%s))(?![\w.])["']?"""
                  % "|".join(e[1:] for e in EXT), re.I)
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
COMMON = {"self", "main", "init", "test", "data", "file", "path", "text", "name", "list", "dict", "none", "true",
          "false", "this", "that", "with", "from", "return", "value", "type", "result", "index", "read", "write",
          "readme", "config", "utils", "setup", "args", "line", "lines", "string", "print", "output", "input"}
DEFS = {}


def norm(p, cwd):
    p = p.strip("\"'").replace("\\", "/")
    m = re.match(r"^/([a-zA-Z])/(.*)", p)
    if m:
        p = f"{m.group(1)}:/{m.group(2)}"
    if p.startswith("~"):
        p = os.path.expanduser("~").replace("\\", "/") + p[1:]
    if not re.match(r"^[a-zA-Z]:/", p):
        if not cwd:
            return None
        p = cwd.replace("\\", "/").rstrip("/") + "/" + p
    return os.path.normpath(p).replace("\\", "/").lower()


def defs(p):
    if p not in DEFS:
        out = set()
        try:
            if os.path.getsize(p) < 400_000:
                with open(p, encoding="utf8", errors="replace") as f:
                    out = {s.name.lower() for s in symbols(f.read(), p)
                           if s.kind in ("def", "class", "const") and len(s.name) >= 4} - COMMON
        except OSError:
            out = None
        DEFS[p] = out
    return DEFS[p]


def listdir(d, cache={}):
    if d not in cache:
        try:
            cache[d] = [f"{d}/{n}".lower() for n in sorted(os.listdir(d))
                        if os.path.splitext(n)[1].lower() in EXT and os.path.isfile(os.path.join(d, n))][:300]
        except OSError:
            cache[d] = []
    return cache[d]


def sessions():
    out = []
    for f in glob.glob(os.path.join(SRC, "*", "*.jsonl")) + glob.glob(os.path.join(SRC, "*", "*", "subagents", "*.jsonl")):
        last, touch, dirs, prev, snap, rows, t0, step = {}, {}, [], None, [], [], None, 0
        for line in open(f, encoding="utf8", errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t0 = t0 or r.get("timestamp")
            c = (r.get("message") or {}).get("content")
            if r.get("type") != "assistant" or not isinstance(c, list):
                continue
            cwd = r.get("cwd") or ""
            for b in c:
                if not isinstance(b, dict):
                    continue
                if b.get("type") in ("text", "thinking"):
                    snap.append(b.get("text") or b.get("thinking") or "")
                    continue
                if b.get("type") != "tool_use":
                    continue
                inp = b.get("input") or {}
                if b.get("name") in ("Read", "Edit", "Write") and inp.get("file_path"):
                    paths = [norm(inp["file_path"], cwd)]
                elif b.get("name") == "Bash" and inp.get("command"):
                    paths = [norm(m.group(1), cwd) for m in PATH.finditer(inp["command"])]
                else:
                    paths = []
                paths = [p for p in dict.fromkeys(paths) if p and os.path.exists(p)]
                text = " ".join(snap)
                low = text.lower()
                words = {w.lower() for w in WORD.findall(text)} - COMMON
                for p in paths:
                    step += 1
                    if prev and p != prev and b.get("name") != "Write" and dirs:
                        cands = list(dict.fromkeys([q for d in dirs for q in listdir(d)] + list(last)))
                        cands = [q for q in cands if q != prev]
                        if p in cands and len(cands) >= 2:
                            F = []
                            for q in cands:
                                stem = os.path.splitext(os.path.basename(q))[0]
                                dq = defs(q) or set()
                                F.append([math.log1p(step - last[q]) if q in last else 8.0, math.log1p(touch.get(q, 0)),
                                          math.log1p(len(words & dq)),
                                          float(len(stem) > 2 and (os.path.basename(q) in low or stem in words)),
                                          float(os.path.dirname(q) == os.path.dirname(prev))])
                            rows.append(dict(F=np.array(F, float), y=np.array([q == p for q in cands]),
                                             first=p not in last, snap=bool(text.strip()), text=text[-2000:],
                                             cands=cands))
                    last[p] = step
                    touch[p] = touch.get(p, 0) + 1
                    d = os.path.dirname(p)
                    if d not in dirs:
                        dirs.append(d)
                    prev = p
                if b.get("type") == "tool_use":
                    snap = []
        if rows:
            out.append((t0 or "", rows))
    return sorted(out, key=lambda x: x[0])


def metrics(rows, score):
    t1, t5, mrr = [], [], []
    for x in rows:
        s = score(x["F"]) + 1e-9 * np.random.default_rng(len(x["F"])).random(len(x["F"]))
        pos = int(np.where(x["y"][np.argsort(-s)])[0][0]) + 1
        t1.append(pos == 1)
        t5.append(pos <= 5)
        mrr.append(1 / pos)
    return np.mean(t1), np.mean(t5), np.mean(mrr)


def fit(rows, cols):
    F = np.vstack([x["F"] for x in rows])[:, cols]
    y = np.concatenate([x["y"] for x in rows])
    mu, sd = F.mean(0), F.std(0) + 1e-6
    lr = LogisticRegression(C=1.0, max_iter=3000).fit((F - mu) / sd, y)
    return lambda G: ((G[:, cols] - mu) / sd) @ lr.coef_[0], lr.coef_[0]


def main():
    S = sessions()
    half = len(S) // 2
    tr = [x for _, rs in S[:half] for x in rs]
    te = [x for _, rs in S[half:] for x in rs]
    f_rec, _ = fit(tr, [0, 1, 4])
    f_all, w_all = fit(tr, [0, 1, 2, 3, 4])
    rankers = [("random", lambda F: np.zeros(len(F))), ("recency (LRU)", lambda F: -F[:, 0]),
               ("snapshot names a symbol defined in the file", lambda F: F[:, 2] - 1e-3 * F[:, 0]),
               ("snapshot names the file", lambda F: F[:, 3] - 1e-3 * F[:, 0]),
               ("logistic: recency + frequency + same dir", f_rec),
               ("logistic: + snapshot (symbol + file mention)", f_all)]
    allr = tr + te
    L = ["# Snapshot signal: does the model's reasoning before a call predict the next file it opens?", "",
         f"{len(allr)} file-to-file transitions in {len(S)} sessions. Candidates per transition: median "
         f"{int(np.median([len(x['y']) for x in allr]))} files. Snapshot (thinking + text since the previous call) non-empty "
         f"in {np.mean([x['snap'] for x in allr]):.0%}. First-time opens: {np.mean([x['first'] for x in allr]):.0%}. "
         f"Logistic fits on the first half of sessions by time ({len(tr)}), tested on the second ({len(te)}).", ""]
    for nm, sub in [("all test transitions", te), ("re-opens", [x for x in te if not x["first"]]),
                    ("first-time opens", [x for x in te if x["first"]])]:
        L += [f"## {nm} ({len(sub)})", "", "| ranker | top-1 | top-5 | MRR |", "|---|---|---|---|"]
        for name, fn in rankers:
            a, b, c = metrics(sub, fn) if sub else (float("nan"),) * 3
            L.append(f"| {name} | {a:.2f} | {b:.2f} | {c:.2f} |")
        L.append("")
    ment = [x for x in te if x["F"][x["y"]][0, 2] > 0 or x["F"][x["y"]][0, 3] > 0]
    L += [f"The snapshot names the opened file or one of its symbols in {len(ment) / max(len(te), 1):.0%} of test transitions "
          f"({np.mean([x['first'] for x in ment]) if ment else 0:.0%} of those are first-time opens).",
          "Logistic weights (standardized; recency, frequency, symbol mention, file mention, same dir): "
          + ", ".join(f"{w:+.2f}" for w in w_all)]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "snapshot_signal.md")
    open(out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
