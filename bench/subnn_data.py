"""Symbol-level training cases for Cortex-SubNN: (context views, file, needed symbols).

Sources (all local; output goes outside the repo because it contains private session text):
  sessions  Claude Code logs. A case is a use of specific lines of a file: an Edit (old_string located in the file text
            the agent had seen), a ranged Read, `sed -n 'a,bp' file` in the shell, or a cortex_range call. The file text
            comes from earlier full Reads / Writes in the session with later Edits applied to it; if the session never
            showed the file, the file on disk is used and an Edit is located by its new_string (source "disk").
            Views: the user prompt in force, the reasoning snapshot (thinking + text since the previous tool call),
            and the symbols used earlier in the session. C_turn is the real context size of that assistant turn.
  git       Local repos: commit message -> leaf symbols the commit changed (no snapshot, no session trail). Messages
            are written after the change, so these are auxiliary training data, not the test set.
Labels: leaf symbols (MCP default split at 600 tokens) overlapping the used span. Cases whose span covers > 40% of the
file, files with < 3 leaf symbols, and cases where every leaf is a label are dropped.

Usage: python -I bench/subnn_data.py [out_dir]     (default D:/dev-cache/cortexnet/subnn)
"""
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortexnet import symbols  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "D:/dev-cache/cortexnet/subnn"
LOGS = os.path.join(os.path.expanduser("~"), ".claude", "projects")
REPOS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = (".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".java", ".kt",
       ".dart", ".md", ".html", ".css", ".sh", ".cs", ".swift", ".rb", ".php", ".lua", ".vue", ".svelte")
LINE = re.compile(r"^\s*(\d+)[\t→](.*)$")
SED = re.compile(r"""sed\s+-n\s+["']?(\d+),(\d+)p["']?\s+(["']?)([^\s"'|;&>]+)\3""")
SPLIT = 600
MAX_TEXT = 400_000


def text_of(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""


def norm(p, cwd):
    p = (p or "").strip("\"'").replace("\\", "/")
    m = re.match(r"^/([a-zA-Z])/(.*)", p)
    if m:
        p = f"{m.group(1)}:/{m.group(2)}"
    if p.startswith("~"):
        p = os.path.expanduser("~").replace("\\", "/") + p[1:]
    if not re.match(r"^[a-zA-Z]:/", p):
        if not cwd:
            return None
        p = cwd.replace("\\", "/").rstrip("/") + "/" + p
    return os.path.normpath(p).replace("\\", "/")


def disk(p):
    try:
        if os.path.getsize(p) <= MAX_TEXT:
            with open(p, encoding="utf8", errors="replace") as f:
                return f.read().replace("\r\n", "\n")
    except OSError:
        pass
    return None


class Store:
    """File texts are stored once by hash; cases point at them."""

    def __init__(self):
        self.texts = {}

    def put(self, text):
        h = hashlib.sha1(text.encode("utf8", "replace")).hexdigest()[:16]
        self.texts.setdefault(h, text)
        return h


def label(text, path, lo, hi):
    """Leaf symbols overlapping lines lo..hi, or None if the case is unusable."""
    n = text.count("\n") + 1
    if hi < lo or hi - lo + 1 > 0.4 * n:
        return None
    syms = [s for s in symbols(text, path, split_tokens=SPLIT) if not s.children]
    if len(syms) < 3:
        return None
    y = [i for i, s in enumerate(syms) if s.start <= hi and s.end >= lo]
    if not y or len(y) == len(syms):
        return None
    return y


def session_cases(store):
    out = []
    files = glob.glob(os.path.join(LOGS, "*", "*.jsonl")) + glob.glob(os.path.join(LOGS, "*", "*", "subagents", "*.jsonl"))
    for f in files:
        sid = os.path.splitext(os.path.basename(f))[0]
        known, pend, snap, prompt, trail, step, ctx, cwd, seen = {}, {}, [], "", [], 0, 0, "", {}
        for line in open(f, encoding="utf8", errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            c = (r.get("message") or {}).get("content")
            cwd = r.get("cwd") or cwd
            if r.get("type") == "user":
                bl = c if isinstance(c, list) else []
                res = [b for b in bl if isinstance(b, dict) and b.get("type") == "tool_result"]
                if not res and not r.get("isMeta"):
                    t = re.sub(r"<system-reminder>.*?</system-reminder>", "", text_of(c), flags=re.S).strip()
                    if t and not t.startswith("<"):
                        prompt, trail = t[:3000], []
                for b in res:
                    p = pend.pop(b.get("tool_use_id"), None)
                    if not p or b.get("is_error"):
                        continue
                    body = b["content"] if isinstance(b.get("content"), str) else text_of(b.get("content"))
                    kind, fp = p["kind"], p["fp"]
                    lo = hi = None
                    if kind in ("read", "range"):
                        seen[fp] = p              # the views at the last time the agent read this file
                    if kind == "read":
                        rows = [LINE.match(l) for l in body.split("\n")]
                        rows = [(int(m.group(1)), m.group(2)) for m in rows if m]
                        if not rows:
                            continue
                        if not p["ranged"] and rows[0][0] == 1 and rows[-1][0] == len(rows):
                            known[fp] = "\n".join(t for _, t in rows)
                            continue
                        if p["ranged"]:
                            lo, hi = rows[0][0], rows[-1][0]
                    elif kind == "write":
                        known[fp] = p["content"]
                        continue
                    elif kind == "edit":
                        text, src = known.get(fp), "session"
                        old, new = p["old"], p["new"]
                        if text is not None and text.count(old) == 1:
                            lo = text[:text.index(old)].count("\n") + 1
                            hi = lo + old.count("\n")
                            label_text = text
                            known[fp] = text.replace(old, new, 1)
                        else:
                            d = disk(p["path"])
                            if d is None or d.count(new) != 1 or not new.strip():
                                continue
                            label_text, src = d, "disk"
                            lo = d[:d.index(new)].count("\n") + 1
                            hi = lo + new.count("\n")
                        p["text"], p["src"] = label_text, src
                    elif kind == "range":
                        lo, hi = p["lo"], p["hi"]
                    if lo is None:
                        continue
                    text = p.get("text")
                    src = p.get("src", "session")
                    if text is None:
                        text = known.get(fp)
                        if text is None:
                            text, src = disk(p["path"]), "disk"
                    if not text or not fp.lower().endswith(EXT):
                        continue
                    y = label(text, fp, lo, hi)
                    if y is None:
                        continue
                    h = store.put(text)
                    syms = [s for s in symbols(text, fp, split_tokens=SPLIT) if not s.children]
                    # decision time = when the file was read: an edit takes the views of the last read of its file
                    v, at = (seen[fp], "read") if kind == "edit" and fp in seen else (p, kind)
                    out.append(dict(source="sessions", sid=sid, ts=p["ts"], step=p["step"], kind=kind, src=src,
                                    path=fp, text=h, y=y, lo=lo, hi=hi, q=v["prompt"], h=v["snap"],
                                    prev=list(v["trail"]), c_turn=v["ctx"], repo=p["repo"], views_at=at))
                    trail.append(dict(path=fp, names=[syms[i].name for i in y],
                                      body="\n".join(text.split("\n")[lo - 1:hi])[:1200], step=p["step"]))
                    del trail[:-6]
                continue
            if r.get("type") != "assistant" or not isinstance(c, list):
                continue
            u = (r.get("message") or {}).get("usage") or {}
            ctx = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + \
                (u.get("cache_creation_input_tokens") or 0) or ctx
            for b in c:
                if not isinstance(b, dict):
                    continue
                if b.get("type") in ("text", "thinking"):
                    snap.append(b.get("text") or b.get("thinking") or "")
                    continue
                if b.get("type") != "tool_use":
                    continue
                step += 1
                inp, name = b.get("input") or {}, b.get("name") or ""
                base = dict(ts=r.get("timestamp") or "", step=step, prompt=prompt, snap=" ".join(snap).strip()[-4000:],
                            trail=list(trail), ctx=ctx, repo=cwd.replace("\\", "/").lower())
                snap = []
                fp, path = None, None
                if name in ("Read", "Edit", "Write", "MultiEdit") and inp.get("file_path"):
                    path = inp["file_path"]
                    fp = norm(path, cwd)
                if name == "Read" and fp:
                    pend[b.get("id")] = dict(base, kind="read", fp=fp, path=path,
                                             ranged=bool(inp.get("offset") or inp.get("limit")))
                elif name == "Write" and fp and isinstance(inp.get("content"), str):
                    pend[b.get("id")] = dict(base, kind="write", fp=fp, path=path, content=inp["content"])
                elif name == "Edit" and fp and inp.get("old_string") and not inp.get("replace_all"):
                    pend[b.get("id")] = dict(base, kind="edit", fp=fp, path=path, old=inp["old_string"],
                                             new=inp.get("new_string") or "")
                elif name == "Bash" and inp.get("command"):
                    m = SED.search(inp["command"])
                    if m:
                        fp = norm(m.group(4), cwd)
                        if fp:
                            pend[b.get("id")] = dict(base, kind="range", fp=fp, path=fp, lo=int(m.group(1)),
                                                     hi=int(m.group(2)))
                elif name.endswith("cortex_range") and inp.get("path"):
                    fp = norm(inp["path"], cwd)
                    if fp and inp.get("start") and inp.get("end"):
                        pend[b.get("id")] = dict(base, kind="range", fp=fp, path=fp, lo=int(inp["start"]),
                                                 hi=int(inp["end"]))
    return out


def git_cases(store):
    sys.argv = sys.argv[:1]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from commit_symbols import repo_cases  # noqa: E402
    out = []
    for d in sorted(os.listdir(REPOS)):
        repo = os.path.join(REPOS, d)
        if not os.path.isdir(os.path.join(repo, ".git")):
            continue
        for c in repo_cases(repo):
            text = "\n".join(c["lines"])
            syms = [s for s in symbols(text, c["path"], split_tokens=SPLIT) if not s.children]
            old = c["syms"]
            spans = [(s.start, s.end) for s, yy in zip(old, c["y"]) if yy]
            y = [i for i, s in enumerate(syms) if any(s.start <= b and s.end >= a for a, b in spans)]
            if len(syms) < 3 or not y or len(y) == len(syms):
                continue
            out.append(dict(source="git", sid=c["repo"], ts="", step=0, kind="commit", src="git", path=c["path"],
                            text=store.put(text), y=y, lo=0, hi=0, q=c["msg"], h="", prev=[], c_turn=0,
                            repo=c["repo"]))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    store = Store()
    ses = session_cases(store)
    print(f"sessions: {len(ses)} cases", flush=True)
    gc = git_cases(store)
    print(f"git: {len(gc)} cases", flush=True)
    with open(os.path.join(OUT, "cases.jsonl"), "w", encoding="utf8") as f:
        for c in ses + gc:
            f.write(json.dumps(c) + "\n")
    with open(os.path.join(OUT, "texts.json"), "w", encoding="utf8") as f:
        json.dump(store.texts, f)
    kinds = {}
    for c in ses:
        k = (c["kind"], c["src"])
        kinds[k] = kinds.get(k, 0) + 1
    print("session cases by kind/source:", kinds)
    print("with snapshot:", sum(bool(c["h"]) for c in ses), "| distinct files:", len(store.texts),
          "| sessions:", len({c["sid"] for c in ses}))


if __name__ == "__main__":
    main()
