"""Offline replay: what would outline-first reads have cost on real logged edits?

Source: Claude Code session logs (private; this script prints aggregates only). A case is an Edit that follows a
full Read of the same file in the same session. The Read output gives the file text as the agent saw it; the Edit
gives the lines that turned out to be needed.

For each case the real `Reader` serves the file with the query the agent could have supplied at read time, then:
  hit   the edited lines are in what the read showed (outline-first read is enough)
  miss  the model must ask for the enclosing symbol with one range call (cost added)
Costs, in tokens: whole-file read (what happened) vs outline-first read + needed range.
Assumes the model asks for the right symbol from the outline when it misses: an upper bound on savings, not a
measurement of agent behaviour. That needs a live run (bench/live, later).

Usage: python -I bench/replay_edits.py [projects_dir]
"""
import glob
import json
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortexnet import Reader, symbols, tokens  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), ".claude", "projects")
LINE = re.compile(r"^\s*(\d+)[\t→](.*)$")


def text_of(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""


def cases():
    out = []
    files = glob.glob(os.path.join(SRC, "*", "*.jsonl")) + glob.glob(os.path.join(SRC, "*", "*", "subagents", "*.jsonl"))
    for f in files:
        pend, last, prompt, said = {}, {}, "", []
        for line in open(f, encoding="utf8", errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            c = (r.get("message") or {}).get("content")
            if r.get("type") == "user":
                bl = c if isinstance(c, list) else []
                res = [b for b in bl if isinstance(b, dict) and b.get("type") == "tool_result"]
                if not res and not r.get("isMeta"):
                    t = re.sub(r"<system-reminder>.*?</system-reminder>", "", text_of(c), flags=re.S).strip()
                    if t and not t.startswith("<"):
                        prompt, said = t, []
                for b in res:
                    p = pend.pop(b.get("tool_use_id"), None)
                    if p and not b.get("is_error"):
                        body = b["content"] if isinstance(b.get("content"), str) else text_of(b.get("content"))
                        rows = [LINE.match(l) for l in body.split("\n")]
                        rows = [(int(m.group(1)), m.group(2)) for m in rows if m]
                        if rows and rows[0][0] == 1 and rows[-1][0] == len(rows):
                            last[p] = dict(text="\n".join(t for _, t in rows), tok=tokens(body),
                                           prompt=prompt, said=" ".join(said[-2:]))
                continue
            if r.get("type") != "assistant" or not isinstance(c, list):
                continue
            for b in c:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    said.append(b.get("text", ""))
                if b.get("type") != "tool_use":
                    continue
                inp = b.get("input") or {}
                fp = (inp.get("file_path") or "").replace("\\", "/").lower()
                if b.get("name") == "Read" and not (inp.get("offset") or inp.get("limit")):
                    pend[b.get("id")] = fp
                elif b.get("name") == "Edit" and fp in last and inp.get("old_string"):
                    rd = last[fp]
                    lines = rd["text"].split("\n")
                    old = [l for l in inp["old_string"].split("\n") if l.strip()]
                    if not old:
                        continue
                    hit = [i for i, l in enumerate(lines, 1) if old[0].strip() in l]
                    if len(hit) == 1:
                        out.append(dict(path=fp, text=rd["text"], tok=rd["tok"], query=rd["prompt"] + "\n" + rd["said"],
                                        lo=hit[0], hi=hit[0] + len(inp["old_string"].split("\n")) - 1))
    return out


def enclosing(syms, lo, hi, n):
    best = None
    for s in syms:
        if s.kind != "section" and s.start <= lo and s.end >= hi:
            if best is None or s.end - s.start < best.end - best.start:
                best = s
    return (best.start, best.end) if best else (max(1, lo - 20), min(n, hi + 20))


def main():
    cs = cases()
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMP"))
    rows = []
    for k, c in enumerate(cs):
        ext = os.path.splitext(c["path"])[1] or ".txt"
        fp = os.path.join(tmp, f"case{k}{ext}")
        with open(fp, "w", encoding="utf8") as f:
            f.write(c["text"])
        r = Reader(os.path.join(tmp, f"state{k}"))
        for qname, q in [("no query", ""), ("prompt + agent text", c["query"])]:
            out = r.read(fp, q)
            shown = r.files[os.path.abspath(fp)]["shown"]
            ok = all(i in shown for i in range(c["lo"], c["hi"] + 1))
            n = len(c["text"].split("\n"))
            a, b = enclosing(symbols(c["text"], fp), c["lo"], c["hi"], n)
            extra = 0 if ok else tokens("\n".join(f"{i:6d}\t{l}" for i, l in
                                                   enumerate(c["text"].split("\n")[a - 1:b], a)))
            rows.append(dict(q=qname, ext=ext, whole=tokens(out) if "Outline" not in out else None, full=c["tok"],
                             cost=tokens(out) + extra, hit=ok, outline_mode="Outline" in out, n=n))
            r = Reader(os.path.join(tmp, f"state{k}b"))      # fresh state so the two queries do not interact
    L = ["# Replay of logged edits through the outline-first reader", "",
         f"Cases: {len(cs)} edits that followed a full Read of the same file (anchor found once). "
         "Costs in tokens (chars/4). Upper bound: assumes the model asks for the right symbol after a miss.", ""]
    for qn in ["no query", "prompt + agent text"]:
        R = [x for x in rows if x["q"] == qn]
        big = [x for x in R if x["outline_mode"]]
        tf, tc = sum(x["full"] for x in R), sum(x["cost"] for x in R)
        bf, bc = sum(x["full"] for x in big), sum(x["cost"] for x in big)
        L += [f"## Query: {qn}", "",
              f"- all cases: whole-file reads {tf:,} tokens -> outline-first {tc:,} ({tc / tf:.0%})",
              f"- files large enough for outline mode: {len(big)} of {len(R)}; {bf:,} -> {bc:,} tokens ({bc / max(bf, 1):.0%}); "
              f"median per-case ratio {np.median([x['cost'] / x['full'] for x in big]) if big else float('nan'):.2f}",
              f"- edited lines already shown by the read (no range call needed): "
              f"{np.mean([x['hit'] for x in big]) if big else float('nan'):.0%} of outline-mode cases", ""]
    exts = {}
    for x in rows:
        if x["q"] == "prompt + agent text":
            exts[x["ext"]] = exts.get(x["ext"], 0) + 1
    L += ["File types: " + ", ".join(f"{k} {v}" for k, v in sorted(exts.items(), key=lambda t: -t[1]))]
    print("\n".join(L))
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "replay_edits.md"), "w",
         encoding="utf8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
