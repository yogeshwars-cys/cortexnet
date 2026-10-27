"""Does the model's reasoning at read time predict which symbols of the file it will need?

From the local Claude Code logs (aggregates only). A case is a full Read of a file large enough for outline mode.
Labels: leaf symbols whose lines the agent then used within the next 30 tool calls: lines it edited (located by the
edit's anchor text in the read text, so later line shifts do not matter) or lines it re-read with offset/limit (only
while the file had not been edited since). The snapshot is the model's thinking + text written since its previous
tool call, i.e. what it was about to look for. Nothing in the labels is derived from the snapshot.

Rankers fill the Reader's include budget (max(400, 25% of the file's tokens)); metrics: recall of needed symbols
within the budget, share of cases where at least one needed symbol is included, MRR of the first needed symbol.
The logistic mix is fitted on the first half of sessions by time and tested on the second.
Usage: python -I bench/symbol_snapshot.py [projects_dir]
"""
import glob
import json
import os
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortexnet import symbols, tokens  # noqa: E402
from cortexnet.learn import idents  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), ".claude", "projects")
LINE = re.compile(r"^\s*(\d+)[\t→](.*)$")
HORIZON = 30
FULL_BELOW = 800


def text_of(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""


def session_events(f):
    evs, pend, snap = [], {}, []
    for line in open(f, encoding="utf8", errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        c = (r.get("message") or {}).get("content")
        if r.get("type") == "user" and isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") in pend:
                    e = pend.pop(b["tool_use_id"])
                    if not b.get("is_error"):
                        body = b["content"] if isinstance(b.get("content"), str) else text_of(b.get("content"))
                        e["rows"] = [(int(m.group(1)), m.group(2)) for m in (LINE.match(l) for l in body.split("\n")) if m]
            continue
        if r.get("type") != "assistant" or not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            if b.get("type") in ("text", "thinking"):
                snap.append(b.get("text") or b.get("thinking") or "")
                continue
            if b.get("type") != "tool_use":
                continue
            inp = b.get("input") or {}
            fp = (inp.get("file_path") or "").replace("\\", "/").lower()
            e = dict(name=b.get("name"), path=fp, inp=inp, snap=" ".join(snap)[-3000:])
            evs.append(e)
            if b.get("name") == "Read" and fp:
                pend[b.get("id")] = e
            snap = []
    return evs


def cases():
    out = []
    files = glob.glob(os.path.join(SRC, "*", "*.jsonl")) + glob.glob(os.path.join(SRC, "*", "*", "subagents", "*.jsonl"))
    for f in files:
        evs = session_events(f)
        t0 = os.path.getmtime(f)
        for i, e in enumerate(evs):
            rows = e.get("rows")
            if e["name"] != "Read" or not rows or e["inp"].get("offset") or e["inp"].get("limit") or rows[0][0] != 1:
                continue
            text = "\n".join(t for _, t in rows)
            if tokens(text) <= FULL_BELOW:
                continue
            lines = text.split("\n")
            need, edited = set(), False
            for g in evs[i + 1:i + 1 + HORIZON]:
                if g["path"] != e["path"]:
                    continue
                if g["name"] == "Read" and not (g["inp"].get("offset") or g["inp"].get("limit")):
                    break
                if g["name"] == "Edit" and g["inp"].get("old_string"):
                    old = g["inp"]["old_string"]
                    k = text.find(old)
                    if k >= 0 and text.find(old, k + 1) < 0:
                        lo = text[:k].count("\n") + 1
                        need.add((lo, lo + old.count("\n")))
                    edited = True
                elif g["name"] == "Read" and not edited and g.get("rows"):
                    need.add((g["rows"][0][0], g["rows"][-1][0]))
            if not need:
                continue
            syms = [s for s in symbols(text, e["path"]) if not s.children]
            y = np.array([any(s.start <= b and s.end >= a for a, b in need) for s in syms])
            if not y.any() or len(syms) < 3:
                continue
            out.append(dict(t=t0, snap=e["snap"], syms=syms, y=y, lines=lines, file_tok=tokens(text),
                            cost=np.array([tokens("\n".join(lines[s.start - 1:s.end])) for s in syms])))
    return sorted(out, key=lambda c: c["t"])


def evaluate(cs, score):
    rec, anyhit, mrr = [], [], []
    for c in cs:
        s = score(c) + 1e-9 * np.random.default_rng(len(c["y"])).random(len(c["y"]))
        order = np.argsort(-s)
        budget = max(400, 0.25 * c["file_tok"])
        used, shown = 0, set()
        for j in order:
            if used + c["cost"][j] <= budget:
                shown.add(j)
                used += c["cost"][j]
        need = set(np.where(c["y"])[0])
        rec.append(len(need & shown) / len(need))
        anyhit.append(bool(need & shown))
        mrr.append(1 / (1 + int(np.where(c["y"][order])[0][0])))
    return np.mean(rec), np.mean(anyhit), np.mean(mrr)


def main():
    cs = cases()
    with_snap = [c for c in cs if c["snap"].strip()]
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    for c in cs:
        docs = [s.head + "\n" + "\n".join(c["lines"][s.start - 1:s.end])[:1500] for s in c["syms"]]
        V = m.encode(docs, batch_size=32, normalize_embeddings=True)
        z = m.encode([c["snap"] or " "], normalize_embeddings=True)[0]
        c["dense"] = V @ z
        q = idents(c["snap"])
        c["lex"] = np.array([len(idents("\n".join(c["lines"][s.start - 1:s.end])) & q) /
                             max(len(idents("\n".join(c["lines"][s.start - 1:s.end]))), 1) for s in c["syms"]])
        c["name"] = np.array([float(s.name.lower() in q) for s in c["syms"]])
        c["F"] = np.c_[np.log1p(c["cost"]), c["lex"], c["name"], c["dense"]]
    half = len(with_snap) // 2
    tr, te = with_snap[:half], with_snap[half:]
    F = np.vstack([c["F"] for c in tr])
    y = np.concatenate([c["y"] for c in tr])
    mu, sd = F.mean(0), F.std(0) + 1e-6
    lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit((F - mu) / sd, y)
    lr_nd = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(((F - mu) / sd)[:, :3], y)
    rankers = [("random", lambda c: np.zeros(len(c["y"]))),
               ("smallest first (size only)", lambda c: -c["cost"]),
               ("lexical overlap with snapshot", lambda c: c["lex"]),
               ("snapshot names the symbol", lambda c: c["name"] - 1e-3 * c["cost"]),
               ("dense similarity to snapshot", lambda c: c["dense"]),
               ("dense similarity per sqrt(token)", lambda c: (c["dense"] + 1) / np.sqrt(c["cost"])),
               ("logistic: size + lexical + name", lambda c: ((c["F"] - mu) / sd)[:, :3] @ lr_nd.coef_[0]),
               ("logistic: + dense", lambda c: ((c["F"] - mu) / sd) @ lr.coef_[0])]
    L = ["# Symbol-level snapshot signal: which symbols of a file will the model need?", "",
         f"{len(cs)} full reads of outline-size files with a later use of specific lines (edit or ranged re-read) within "
         f"{HORIZON} tool calls; {len(with_snap)} have a non-empty snapshot. Median {int(np.median([len(c['y']) for c in cs]))} "
         f"leaf symbols per file, {np.mean([c['y'].mean() for c in cs]):.0%} of them needed on average. "
         f"Test = later half of the cases with a snapshot ({len(te)}); budget = max(400, 25% of file tokens).", "",
         "| ranker | recall of needed symbols in budget | cases with >= 1 needed symbol shown | MRR |", "|---|---|---|---|"]
    for name, fn in rankers:
        r, a, mm = evaluate(te, fn)
        L.append(f"| {name} | {r:.0%} | {a:.0%} | {mm:.2f} |")
    L += ["", "Logistic weights (standardized; log size, lexical, name, dense): "
          + ", ".join(f"{w:+.2f}" for w in lr.coef_[0])]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "symbol_snapshot.md")
    open(out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
