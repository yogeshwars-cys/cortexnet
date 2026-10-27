"""Session-level replay: real agent sessions through one Reader each, state carried across the session.

Source: local Claude Code logs (private; only aggregates are written). Sessions are replayed in time order and the
include model's weights carry over from session to session (online learning from the agent's own misses).
A per-session workspace mirrors the files the session read or wrote, so neighbour files exist for the menu.

Per logged event:
  full Read        -> cortex read(path, query = user prompt + agent's last words)
  partial Read     -> cortex range(offset, offset+limit-1)           (the agent already asked for a range)
  Edit             -> the needed lines are the edited ones. If they were shown: hit. Else, if an open menu item covers
                      them: approve it. Else: one range call for the enclosing symbol. Then the edit is applied to the
                      workspace file, so the change watcher and the trail see it.
  Write            -> the file is written to the workspace.
Baseline = the tokens the real Read calls returned. Upper bound caveats: the simulated agent always asks for the
right symbol after a miss and approves exactly the menu items it needs.

Usage: python -I bench/replay_sessions.py [projects_dir]
"""
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortexnet import Reader, symbols, tokens  # noqa: E402
from cortexnet.learn import FEATURES  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), ".claude", "projects")
LINE = re.compile(r"^\s*(\d+)[\t→](.*)$")
V1 = [f for f in FEATURES if f not in ("trail", "conn")]
ARMS = {"outline only (no up-front includes)": dict(features=V1, menu=False, tau=1.01),
        "outline only, big symbols split at 600 tok": dict(features=V1, menu=False, tau=1.01, split_tokens=600),
        "outline only, big symbols split at 300 tok": dict(features=V1, menu=False, tau=1.01, split_tokens=300),
        "v1 (query, changed, asked, structure)": dict(features=V1, menu=False),
        "+ trail": dict(features=V1 + ["trail"], menu=False),
        "+ trail + graph": dict(features=FEATURES, menu=False),
        "+ trail + graph + menu": dict(features=FEATURES, menu=True)}


def text_of(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""


def load_sessions():
    out = []
    for f in glob.glob(os.path.join(SRC, "*", "*.jsonl")) + glob.glob(os.path.join(SRC, "*", "*", "subagents", "*.jsonl")):
        evs, pend, prompt, said, t0 = [], {}, "", [], None
        for line in open(f, encoding="utf8", errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t0 = t0 or r.get("timestamp")
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
                    if not p or b.get("is_error"):
                        continue
                    body = b["content"] if isinstance(b.get("content"), str) else text_of(b.get("content"))
                    rows = [LINE.match(l) for l in body.split("\n")]
                    rows = [(int(m.group(1)), m.group(2)) for m in rows if m]
                    if not rows:
                        continue
                    p.update(tok=tokens(body), rows=rows)
                    evs.append(p)
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
                if not fp:
                    continue
                if b.get("name") == "Read":
                    pend[b.get("id")] = dict(op="read", path=fp, off=inp.get("offset"), lim=inp.get("limit"),
                                             query=prompt + "\n" + " ".join(said[-2:]))
                elif b.get("name") == "Edit" and inp.get("old_string") is not None:
                    evs.append(dict(op="edit", path=fp, old=inp["old_string"], new=inp.get("new_string", ""),
                                    all=bool(inp.get("replace_all"))))
                elif b.get("name") == "Write" and isinstance(inp.get("content"), str):
                    evs.append(dict(op="write", path=fp, text=inp["content"]))
        if any(e["op"] == "read" for e in evs) and any(e["op"] == "edit" for e in evs):
            out.append((t0 or "", f, evs))
    return sorted(out)


def ws_path(root, fp):
    return os.path.join(root, fp.replace(":", "").lstrip("/"))


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8", newline="\n") as f:
        f.write(text)


def enclosing(text, path, lo, hi, split_tokens=0):
    best = None
    for s in symbols(text, path, split_tokens):
        if s.kind != "section" and s.start <= lo and s.end >= hi and (best is None or s.end - s.start < best.end - best.start):
            best = s
    n = text.count("\n") + 1
    return (best.start, best.end) if best else (max(1, lo - 20), min(n, hi + 20))


def run_arm(sessions, arm, root):
    state = os.path.join(root, "state")
    st = dict(base=0, cost=0, edits=0, hit=0, approved=0, ranged=0, menu_items=0, menu_tok=0, reads=0, outline_reads=0,
              hit_first=[], hit_second=[])
    half = len(sessions) // 2
    for k, (_, _, evs) in enumerate(sessions):
        ws = os.path.join(root, f"s{k}")
        r = Reader(state, **arm)
        for e in evs:
            wp = ws_path(ws, e["path"])
            if e["op"] == "write":
                write(wp, e["text"])
            elif e["op"] == "read":
                rows = e["rows"]
                full = not (e["off"] or e["lim"]) and rows[0][0] == 1
                if full:
                    write(wp, "\n".join(t for _, t in rows))
                    out = r.read(wp, e["query"])
                    st["reads"] += 1
                    st["outline_reads"] += "Outline" in out
                    st["base"] += e["tok"]
                    st["cost"] += tokens(out)
                elif os.path.exists(wp):
                    a, b = rows[0][0], rows[-1][0]
                    st["base"] += e["tok"]
                    st["cost"] += tokens(r.range(wp, a, b))
            elif e["op"] == "edit" and os.path.exists(wp):
                with open(wp, encoding="utf8") as f:
                    text = f.read()
                i = text.find(e["old"]) if e["old"] else -1
                if i < 0:
                    continue
                lo = text[:i].count("\n") + 1
                hi = lo + e["old"].count("\n")
                p = os.path.abspath(wp)
                shown = r.files.get(p, {}).get("shown", set())
                st["edits"] += 1
                ok = all(j in shown for j in range(lo, hi + 1))
                (st["hit_first"] if k < half else st["hit_second"]).append(ok)
                if ok:
                    st["hit"] += 1
                else:
                    mid = next((m for m, v in r.menu.items() if v["path"] == p and v["a"] <= lo and v["b"] >= hi), None)
                    if mid:
                        st["cost"] += tokens(r.approve([mid]))
                        st["approved"] += 1
                    else:
                        a, b = enclosing(text, p, lo, hi, arm.get("split_tokens", 0))
                        st["cost"] += tokens(r.range(p, a, b))
                        st["ranged"] += 1
                text = text.replace(e["old"], e["new"]) if e["all"] else text.replace(e["old"], e["new"], 1)
                write(wp, text)
                time.sleep(0.002)          # make sure the next mtime differs
        if os.path.exists(r.log_path):
            with open(r.log_path, encoding="utf8") as f:
                for line in f:
                    ev = json.loads(line)
                    if ev["op"] == "read":
                        st["menu_items"] += ev.get("menu", 0)
                        st["menu_tok"] += ev.get("menu_tokens", 0)
            os.remove(r.log_path)
        shutil.rmtree(ws, ignore_errors=True)
    st["model"] = Reader(state).model.w.round(2).tolist()
    return st


def main():
    sessions = load_sessions()
    root = tempfile.mkdtemp(dir=os.environ.get("TMP"))
    print(f"{len(sessions)} sessions with reads and edits", flush=True)
    res = {}
    for name, arm in ARMS.items():
        t = time.time()
        res[name] = run_arm(sessions, arm, os.path.join(root, re.sub(r"\W+", "_", name)))
        print(f"{name}: {time.time() - t:.0f}s", flush=True)
    shutil.rmtree(root, ignore_errors=True)
    L = ["# Session replay: trail memory, symbol graph and connected-symbol menu", "",
         f"{len(sessions)} sessions (with both reads and edits), replayed in time order; include model carried across "
         "sessions. Tokens = chars/4. Baseline = tokens the real Read calls returned. Upper bound: the simulated agent "
         "fetches the right symbol after a miss and approves exactly the menu items it needs.", "",
         "| arm | read tokens vs baseline | edits with lines already shown | first half / second half of sessions | "
         "misses served by menu | misses by range | menu items offered | menu tokens |", "|---|---|---|---|---|---|---|---|"]
    for name, s in res.items():
        L.append(f"| {name} | {s['cost']:,} / {s['base']:,} ({s['cost'] / max(s['base'], 1):.0%}) | "
                 f"{s['hit'] / max(s['edits'], 1):.0%} of {s['edits']} | {np.mean(s['hit_first'] or [0]):.0%} / "
                 f"{np.mean(s['hit_second'] or [0]):.0%} | {s['approved']} | {s['ranged']} | {s['menu_items']} | {s['menu_tok']:,} |")
    s = next(iter(res.values()))
    L += ["", f"Reads replayed: {s['reads']} full reads, {s['outline_reads']} served outline-first (the rest are small files "
          "returned whole).", "", "Learned weights after the last session (" + ", ".join(FEATURES) + "):", ""]
    L += [f"- {n}: {v['model']}" for n, v in res.items()]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "replay_sessions.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
