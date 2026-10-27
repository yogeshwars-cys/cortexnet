"""Muse Glimmer as a coding agent on SWE-bench Lite, writing live cases for the Cortex-SubNN coprocessor.

The loop drives Glimmer's raw ATEM format (chat_template.jinja): reasoning goes to `assistant to=self`, a tool call is
`assistant to=<tool>` carrying `<atem:function_calls>`, results come back as `tool <name>` with `<tool_output>`.
Tools: read_file (the CortexNET Reader, outline-first), read_lines, grep, list_dir, submit. No code is executed.

Per read_file call the episode keeps the token position just before the call block: the decision state, whose hidden
vectors the Modal job reads with layer hooks after the episode. Labels for a read of file F: leaf symbols of F that
the gold patch changes, that Glimmer later asked for with read_lines, or that it submitted.

Cases are written in bench/subnn_data.py's format (cases.jsonl + texts.json + hrem.json), plus `qv`, the row of the
decision state in hidden_L{layer}.npy, so bench/subnn_train.py trains and evaluates on them unchanged.
"""
import hashlib
import json
import os
import re
import time

from cortexnet import Reader
from cortexnet.coproc import leaves
from cortexnet.outline import outline

MAX_OUT = 12000          # characters of one tool result shown to the model
MAX_READ = 80000         # characters of one read_file result, both arms (whole-file control: about 2000 lines)
WHOLE_LINES = 2000       # the whole-file arm returns at most this many numbered lines, like Claude Code's Read

TOOLS = [
    dict(type="function", function=dict(
        name="read_file", description="Read a source file. Large files come back as an outline (symbols with line "
        "ranges) plus the bodies most likely needed for your query; use read_lines for anything else.",
        parameters=dict(type="object", properties=dict(
            path=dict(type="string", description="path relative to the repository root"),
            query=dict(type="string", description="what you are looking for in this file")), required=["path"]))),
    dict(type="function", function=dict(
        name="read_lines", description="Return lines start..end (1-based, inclusive) of a file, numbered.",
        parameters=dict(type="object", properties=dict(path=dict(type="string"), start=dict(type="integer"),
                                                       end=dict(type="integer")), required=["path", "start", "end"]))),
    dict(type="function", function=dict(
        name="grep", description="Search a regular expression in the repository (or under a sub-path). Returns "
        "file:line: text for up to 40 matches.",
        parameters=dict(type="object", properties=dict(pattern=dict(type="string"), path=dict(type="string")),
                        required=["pattern"]))),
    dict(type="function", function=dict(
        name="list_dir", description="List a directory of the repository.",
        parameters=dict(type="object", properties=dict(path=dict(type="string")), required=[]))),
    dict(type="function", function=dict(
        name="submit", description="Finish: the file and function/class that must change, and the patch you propose.",
        parameters=dict(type="object", properties=dict(file=dict(type="string"), symbol=dict(type="string"),
                                                       patch=dict(type="string")), required=["file", "symbol"]))),
]

SYSTEM = ("You are an expert software engineer working in the repository at the root '.'. You will be given an "
          "issue. Find the code that must change to fix it, by exploring the repository with the tools, then call "
          "submit with the file, the function or class to change, and a proposed patch. You cannot run code. "
          "Prefer read_file with a precise query; use read_lines to see bodies the outline does not show.\n\n"
          "Reasoning strength: low")

# control arm: read_file returns the whole file (numbered), no outline
TOOLS_WHOLE = [dict(t, function=dict(t["function"], description="Read a source file: its full text, numbered lines."))
               if t["function"]["name"] == "read_file" else t for t in TOOLS]
SYSTEM_WHOLE = SYSTEM.replace("Prefer read_file with a precise query; use read_lines to see bodies the outline does "
                              "not show.", "read_file returns the whole file with line numbers.")

INVOKE = re.compile(r'<atem:invoke name="([^"]+)">(.*?)</atem:invoke>', re.S)
PARAM = re.compile(r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>', re.S)
HEADER = re.compile(r'^\s*(?:<\|start\|>assistant)?\s*(?:to=(\S+?))?\s*<\|message\|>', re.S)


def parse_calls(body):
    out = []
    for name, inner in INVOKE.findall(body):
        args = {k: v.strip() for k, v in PARAM.findall(inner)}
        out.append((name.split(".")[-1], args))
    return out


def gold_lines(patch):
    """{file: original line numbers the patch removes or inserts next to}."""
    out, f, ln, prev = {}, None, 0, " "
    for line in patch.split("\n"):
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            f = line[4:].strip()
            f = f[2:] if f.startswith("b/") else f
            continue
        m = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
        if m:
            ln = int(m.group(1))
            continue
        if f is None or not line:
            continue
        if line[0] == "-":
            out.setdefault(f, set()).add(ln)
            ln += 1
        elif line[0] == "+":
            if prev not in "-+":                    # a pure insertion: the lines on both sides
                out.setdefault(f, set()).update({max(1, ln - 1), ln})
        else:
            ln += 1
        prev = line[0]
    return out


class Workspace:
    def __init__(self, root, state, arm="cortex"):
        self.root = os.path.realpath(root)
        self.arm = arm
        self.reader = Reader(state, split_tokens=600)
        self.ranges, self.submitted = [], None          # (call index, rel path, a, b)

    def path(self, p):
        q = os.path.realpath(os.path.join(self.root, (p or ".").lstrip("/")))
        if not (q == self.root or q.startswith(self.root + os.sep)):
            raise ValueError(f"path outside the repository: {p}")
        return q

    def rel(self, q):
        return os.path.relpath(q, self.root).replace(os.sep, "/")

    def call(self, i, name, a):
        try:
            if name == "read_file":
                if self.arm == "whole":
                    with open(self.path(a["path"]), encoding="utf8", errors="replace") as f:
                        lines = f.read().split("\n")
                    more = f"\n[{len(lines) - WHOLE_LINES} more lines]" if len(lines) > WHOLE_LINES else ""
                    return "\n".join(f"{i:>6}\t{l}" for i, l in enumerate(lines[:WHOLE_LINES], 1)) + more
                return self.reader.read(self.path(a["path"]), a.get("query", ""))
            if name == "read_lines":
                q, s, e = self.path(a["path"]), int(a["start"]), int(a["end"])
                self.ranges.append((i, self.rel(q), s, e))
                return self.reader.range(q, s, e)
            if name == "grep":
                return self.grep(a["pattern"], a.get("path", "."))
            if name == "list_dir":
                q = self.path(a.get("path", "."))
                return "\n".join(n + ("/" if os.path.isdir(os.path.join(q, n)) else "") for n in sorted(os.listdir(q)))
            if name == "submit":
                self.submitted = (a.get("file", ""), a.get("symbol", ""), a.get("patch", ""))
                return "submitted"
            return f"unknown tool {name}"
        except (OSError, KeyError, ValueError, re.error) as e:
            return f"{type(e).__name__}: {e}"

    def grep(self, pattern, sub):
        rx, out = re.compile(pattern), []
        for d, dirs, files in os.walk(self.path(sub)):
            dirs[:] = sorted(x for x in dirs if not x.startswith("."))
            for n in sorted(files):
                if not n.endswith((".py", ".pyx", ".rst", ".txt", ".cfg", ".toml", ".c", ".h")):
                    continue
                q = os.path.join(d, n)
                try:
                    with open(q, encoding="utf8", errors="replace") as f:
                        for k, line in enumerate(f, 1):
                            if rx.search(line):
                                out.append(f"{self.rel(q)}:{k}: {line.rstrip()[:200]}")
                                if len(out) >= 40:
                                    return "\n".join(out)
                except OSError:
                    continue
        return "\n".join(out) or "no matches"


def tool_block(name, out):
    return f'<|start|>tool {name}<|message|><tool_output name="{name}">\n{out}\n</tool_output><|eot|>'


def faded_block(ws, path):
    """What stays on the desk once a read fades: the file's outline (the important details), bodies pulled back."""
    with open(ws.path(path), encoding="utf8", errors="replace") as f:
        text = f.read()
    return tool_block("read_file", f"[faded: {path} was moved off the desk; outline below, read_lines for bodies]\n"
                                   + outline(text, path))


def run_episode(gen, encode, decode, start_ids, ws, max_calls=20, max_tokens=60000, fade_age=0):
    """gen(ids) -> generated ids up to and including an end-of-message token. Returns the episode record.

    The context is kept as segments (start, generations, tool results). fade_age > 0: before each turn, a read_file
    result whose file was not touched (read, ranged or submitted) in the last fade_age turns is replaced by the file's
    outline. Positions in the record (pos, reads[].at) are only valid without fading."""
    segs, events, reasoning, calls, turns = [list(start_ids)], [], [], 0, []
    ids = list(start_ids)
    tool_tok, reads, touch, faded = {}, [], {}, 0       # touch: rel path -> last turn that used it
    oom = False
    while calls < max_calls and len(ids) < max_tokens:
        if fade_age:
            T = len(turns)
            for r in reads:
                if not r["faded"] and T - touch.get(r["path"], T) >= fade_age:
                    try:
                        blk = encode(faded_block(ws, r["path"]))
                    except (OSError, ValueError):
                        continue
                    if len(blk) < len(segs[r["seg"]]):
                        segs[r["seg"]], r["faded"] = blk, True
                        faded += 1
            ids = [t for s in segs for t in s]
        before = len(ids)
        t = time.time()
        new = gen(ids)
        dt = time.time() - t
        if new is None:                                 # gen ran out of memory: the context does not fit
            oom = True
            turns.append(dict(to=None, n=0, ctx=before, secs=round(dt, 2)))
            break
        text = decode(new)
        ids += new
        segs.append(new)
        m = HEADER.match(text)
        to = m.group(1) if m else None
        body = text[m.end():] if m else text
        body = re.sub(r"<\|(eom|eot|end)\|>\s*$", "", body)
        turns.append(dict(to=to, n=len(new), ctx=before, secs=round(dt, 2)))  # ctx = prefill tokens of this turn
        if to == "self":
            reasoning.append(body.strip())
        elif to and "<atem:" in body:
            off = text.index("<atem:")
            k = next((i for i in range(1, len(new) + 1) if len(decode(new[:i])) > off), 1)
            pos = before + k - 2                       # last token before the call block: the decision state
            for name, args in parse_calls(body):
                calls += 1
                if name == "read_file":
                    events.append(dict(call=calls, path=args.get("path", ""), query=args.get("query", ""), pos=pos,
                                       snap=" ".join(reasoning).strip()[-4000:], ctx=before))
                out = ws.call(calls, name, args)
                cap = MAX_READ if name == "read_file" else MAX_OUT
                out = out if len(out) <= cap else out[:cap] + "\n[truncated]"
                if max_calls - calls == 3 and name != "submit":
                    out += "\n[3 tool calls left: call submit soon]"
                blk = encode(tool_block(name, out))
                tool_tok[name] = tool_tok.get(name, 0) + len(blk)
                p = args.get("path") or args.get("file")
                try:
                    rel = ws.rel(ws.path(p)) if p else None
                except ValueError:
                    rel = None
                if rel:
                    touch[rel] = len(turns)
                if name == "read_file" and rel:
                    reads.append(dict(call=calls, path=rel, at=len(ids), n=len(blk), out=out, turn=len(turns),
                                      seg=len(segs), faded=False))
                ids += blk
                segs.append(blk)
            if ws.submitted is not None:
                break
        else:
            break                                       # a final message to the user
        nxt = encode("<|start|>assistant")
        ids += nxt
        segs.append(nxt)
    return dict(ids=ids, events=events, calls=calls, turns=turns, submitted=ws.submitted, ranges=ws.ranges,
                tool_tok=tool_tok, hit_limit=len(ids) >= max_tokens, reads=reads, faded=faded, fade_age=fade_age,
                oom=oom)


def score_submission(inst, ws):
    """Did the submission match the gold patch? file_ok: the file is in the patch; symbol_ok: a leaf symbol of that
    name in that file overlaps the changed lines."""
    gold = gold_lines(inst["patch"])
    sub = ws.submitted
    if not sub or not sub[0]:
        return dict(file_ok=False, symbol_ok=False)
    rel = sub[0].strip().lstrip("./")
    if rel not in gold:
        return dict(file_ok=False, symbol_ok=False)
    try:
        with open(ws.path(rel), encoding="utf8", errors="replace") as f:
            syms = leaves(f.read(), rel)
    except (OSError, ValueError):
        return dict(file_ok=True, symbol_ok=False)
    want = (sub[1] or "").split(".")[-1].split("(")[0].strip()
    ok = any(s.name == want and any(s.start <= ln <= s.end for ln in gold[rel]) for s in syms)
    return dict(file_ok=True, symbol_ok=ok)


def cases_from_episode(inst, ep, ws, order, qv0):
    """Live cases (subnn_data.py format) for every read_file of a file with a usable label. Returns cases, texts."""
    gold = gold_lines(inst["patch"])
    cases, texts, trail = [], {}, []
    for k, ev in enumerate(ep["events"]):
        try:
            q = ws.path(ev["path"])
            with open(q, encoding="utf8", errors="replace") as f:
                text = f.read()
        except (OSError, ValueError):
            continue
        rel = ws.rel(q)
        lines = text.split("\n")
        syms = leaves(text, rel)
        need = set(gold.get(rel, set()))
        for i, p, a, b in ep["ranges"]:
            if i > ev["call"] and p == rel:
                need.update(range(a, b + 1))
        sub = ep["submitted"]
        y = [j for j, s in enumerate(syms) if any(s.start <= ln <= s.end for ln in need)
             or (sub and sub[0].lstrip("./") == rel and sub[1] and sub[1].split(".")[-1] == s.name)]
        if len(syms) >= 3 and y and len(y) < len(syms):
            h = hashlib.sha1(text.encode("utf8", "replace")).hexdigest()[:16]
            texts[h] = text
            cases.append(dict(source="sessions", sid=inst["instance_id"], ts=f"{order:05d}-{k:03d}", step=ev["call"],
                              kind="live", src="swe", path=f"{inst['repo']}/{rel}", text=h, y=y, lo=0, hi=0,
                              q=inst["problem_statement"][:4000], h=ev["snap"], prev=list(trail[-6:]),
                              c_turn=ev["ctx"], repo=inst["repo"], views_at="read", qv=qv0 + k))
        nxt = ep["events"][k + 1]["call"] if k + 1 < len(ep["events"]) else 10 ** 9
        for i, p, a, b in ep["ranges"]:            # the trail: what was ranged (any file) before the next read
            if ev["call"] < i < nxt:
                try:
                    with open(ws.path(p), encoding="utf8", errors="replace") as f:
                        t = f.read()
                except (OSError, ValueError):
                    continue
                names = [s.name for s in leaves(t, p) if s.start <= b and s.end >= a]
                trail.append(dict(names=names, body="\n".join(t.split("\n")[a - 1:b])[:1200]))
    return cases, texts
