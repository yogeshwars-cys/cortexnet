"""Egress proxy: CortexNET between any coding agent and its model API.

Point the agent's base URL at the proxy. Every request goes upstream unchanged except one thing: when the newest
turn carries the result of a whole-file read of a large source file, that result is swapped for CortexNET's
outline-first view of the same file (the outline plus the bodies the include model expects to be needed).

Cache-safe by construction. A tool result is rewritten only when it first appears in the newest turn; the rewrite is
memoized by tool-call id and replayed byte-for-byte on every later request, so the provider's prefix cache keeps
matching. Results the proxy never saw fresh (it started mid-session) stay untouched. Assistant messages, the system
prompt and responses are never modified; the optional tool filter is static, so it changes the prefix once.

Escape hatches: reading the same file whole a second time gets the whole file; ranged reads (Read offset/limit,
view_range, sed -n 'a,bp') pass through and label the lines the outline missed, which trains the include model.

Wire formats: Anthropic Messages (Claude Code), OpenAI Chat Completions (most OpenAI-compatible agents, the Meta
Model API) and OpenAI Responses (Codex). Tool calls are matched by name/arguments: read-like tools with a path
argument, and shell tools running `cat FILE` / `sed -n 'a,bp' FILE`.

    cortexnet-proxy run --upstream https://api.anthropic.com -- claude
    cortexnet-proxy serve --upstream https://api.openai.com --port 8787   # OPENAI_BASE_URL=http://127.0.0.1:8787/v1
    cortexnet-proxy stats

Every request appends one line to <state>/proxy/usage.jsonl: usage numbers and rewrite sizes, never bodies.
CORTEXNET_PROXY_REWRITE=0 (or --passthrough) forwards everything unchanged and still logs usage, for A/B baselines.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .outline import tokens
from .reader import SRC_EXT, Reader
from .shellout import is_content_command, outline_output

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding"}
PATH_KEYS = ("file_path", "path", "filename", "file", "target_file", "relative_path", "AbsolutePath",
             "absolute_path", "TargetFile")
RANGE_KEYS = ("offset", "limit", "view_range", "start_line", "end_line", "line_start", "line_end", "StartLine",
              "EndLine")
READ_NAME = re.compile(r"read|view|open|cat", re.I)
NOT_READ = re.compile(r"edit|write|replace|create|insert|patch|delete|cortex|outline|overview|symbol|search|find|"
                      r"grep|list", re.I)
MCP_WRAPPERS = {"call_mcp_tool"}                    # Antigravity: {ServerName, ToolName, Arguments}
FILE_ARG = r"""(['"]?)([^\s'"|;&<>]+)"""
CAT = re.compile(r"^\s*cat\s+" + FILE_ARG + r"\1\s*$")
SED = re.compile(r"""^\s*sed\s+-n\s+['"]?(\d+),(\d+)p['"]?\s+""" + FILE_ARG + r"\3\s*$")
# PowerShell (Codex on Windows): Get-Content / gc / type, whole, the first N lines, or a Skip/First window
PS_WRAP = re.compile(r"""^\s*(?:pwsh|powershell)(?:\.exe)?(?:\s+-\w+)*?\s+-c(?:ommand)?\s+(['"])(.*)\1\s*$""",
                     re.I | re.S)
PS_GC = re.compile(r"""^\s*(?:Get-Content|gc|type)\s+(?:-(?:Literal)?Path\s+)?""" + FILE_ARG + r"\1" +
                   r"""((?:\s+-(?:Raw|Encoding\s+\S+|TotalCount\s+\d+|Head\s+\d+|First\s+\d+))*)"""
                   r"""(?:\s*\|\s*Select-Object\s+((?:-(?:Skip|First)\s+\d+\s*)+))?\s*;?\s*$""", re.I)
CWD = [re.compile(r"<cwd>\s*([^<\n]+?)\s*</cwd>"), re.compile(r"[Ww]orking directory:\s*([^\n]+)")]
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
OUTPUT_TYPES = {"function_call_output", "local_shell_call_output", "custom_tool_call_output"}


# ---------------------------------------------------------------- tool calls -> file reads
def _args(raw):
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def _resolve(path, cwd):
    path = os.path.expanduser(path.strip())
    m = re.match(r"^/([a-zA-Z])(/.*)?$", path)
    if os.name == "nt" and m:                       # Git Bash style /d/foo -> D:/foo
        path = f"{m.group(1).upper()}:{m.group(2) or '/'}"
    if not os.path.isabs(path):
        path = os.path.join(cwd or os.getcwd(), path)
    return os.path.normpath(path)


def _powershell(cmd):
    """('whole', file) | ('range', file, a, b) | None for a PowerShell file read."""
    w = PS_WRAP.match(cmd)
    m = PS_GC.match(w.group(2) if w else cmd)
    if not m:
        return None
    opts = {k.lower(): int(v) for k, v in re.findall(r"-(TotalCount|Head|First)\s+(\d+)", m.group(3) or "", re.I)}
    sel = {k.lower(): int(v) for k, v in re.findall(r"-(Skip|First)\s+(\d+)", m.group(4) or "", re.I)}
    first = next(iter(opts.values()), None)
    if first is None and not sel:
        return "whole", m.group(2)
    a = sel.get("skip", 0) + 1
    b = a - 1 + sel["first"] if "first" in sel else 10 ** 9
    if first is not None:                           # -TotalCount applies before the pipe
        b = min(b, first)
    return "range", m.group(2), a, b


EDITOR_COMMANDS = {"view", "read", "create", "str_replace", "insert", "undo_edit", "write"}
EDITOR_NAME = re.compile(r"edit|write|replace|create|insert|patch", re.I)


def shell_command(call):
    """The command string when a tool call runs a shell command (Bash, shell, PowerShell, exec...), else ''. Editor
    tools that also take a `command` argument (str_replace_editor: view/create/str_replace) are not shell calls."""
    name, args = call
    if name in MCP_WRAPPERS and isinstance(args.get("ToolName"), str):
        name, args = args["ToolName"], _args(args.get("Arguments"))
    cmd = args.get("command", args.get("cmd"))
    if isinstance(cmd, list):
        cmd = " ".join(str(x) for x in cmd)
    if not isinstance(cmd, str) or not cmd.strip() or cmd in EDITOR_COMMANDS or EDITOR_NAME.search(name or ""):
        return ""
    return cmd


def classify(name, args, cwd=None):
    """('whole', path) | ('range', path, a, b) | None for one tool call."""
    if name in MCP_WRAPPERS and isinstance(args.get("ToolName"), str):
        name, args = args["ToolName"], _args(args.get("Arguments"))
    cmd = args.get("command", args.get("cmd"))
    if isinstance(cmd, list):
        cmd = cmd[-1] if cmd else ""
    if isinstance(cmd, str) and cmd and cmd not in ("view", "read"):
        m = SED.match(cmd)
        if m:
            return "range", _resolve(m.group(4), cwd), int(m.group(1)), int(m.group(2))
        m = CAT.match(cmd)
        if m:
            return "whole", _resolve(m.group(2), cwd)
        ps = _powershell(cmd)
        return (ps[0], _resolve(ps[1], cwd)) + ps[2:] if ps else None
    path = next((args[k] for k in PATH_KEYS if isinstance(args.get(k), str) and args[k]), None)
    reads = cmd in ("view", "read") or (READ_NAME.search(name or "") and not NOT_READ.search(name or ""))
    if not path or not reads:
        return None
    path = _resolve(path, cwd)
    if not any(args.get(k) is not None for k in RANGE_KEYS):
        return "whole", path
    if isinstance(args.get("view_range"), list) and len(args["view_range"]) == 2:
        a, b = args["view_range"]
        return "range", path, int(a), int(b) if int(b) > 0 else 10 ** 9
    a = int(args.get("offset") or args.get("start_line") or args.get("line_start") or args.get("StartLine") or 1)
    if args.get("limit") is not None:
        return "range", path, a, a + int(args["limit"]) - 1
    return "range", path, a, int(args.get("end_line") or args.get("line_end") or args.get("EndLine") or 10 ** 9)


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)


def find_cwd(body):
    """Working directory announced in the system prompt (Claude Code) or environment context (Codex)."""
    gem = _gemini(body)
    head = [body.get("system"), body.get("instructions"), gem.get("systemInstruction")]
    for src, key in ((body, "messages"), (body, "input"), (gem, "contents")):
        if isinstance(src.get(key), list):
            head += src[key][:3]
    for s in _strings(head):
        for rx in CWD:
            m = rx.search(s)
            if m:
                return _resolve(m.group(1).strip().strip("\"'"), None)
    return None


def _gemini(body):
    """The generateContent payload: top level (Gemini API) or under "request" (Cloud Code, Antigravity)."""
    inner = body.get("request")
    return inner if isinstance(inner, dict) and "contents" in inner else body


def _text(content):
    """Plain text of a tool result, or None if it holds anything but text (images, documents)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and all(isinstance(b, dict) and isinstance(b.get("text"), str)
                                                     for b in content):
        return "".join(b["text"] for b in content)
    return None


def _set_text(holder, key, new):
    old = holder[key]
    if isinstance(old, str):
        holder[key] = new
        return
    block = dict(old[0], text=new)
    cache = next((b["cache_control"] for b in old if "cache_control" in b), None)
    if cache is not None:
        block["cache_control"] = cache
    holder[key] = [block]


# ---------------------------------------------------------------- the agent's intent, from its own traffic
def _say(content):
    """Visible text and reasoning in one message's content: text, thinking, reasoning summaries, Gemini thoughts."""
    if isinstance(content, str):
        return content
    out = []
    for b in content if isinstance(content, list) else []:
        if isinstance(b, dict):
            for k in ("text", "thinking"):
                if isinstance(b.get(k), str):
                    out.append(b[k])
            out += [s["text"] for s in b.get("summary") or [] if isinstance(s, dict) and isinstance(s.get("text"), str)]
            if b.get("type") in ("message", "reasoning"):
                out.append(_say(b.get("content")))
    return "\n".join(o for o in out if o)


def intents(body):
    """(task, {call id: (snapshot, sibling args)}): the first user message, and for every tool call the reasoning
    and text the agent wrote in the same turn plus the arguments of the other calls in it (grep patterns, names)."""
    task, ctx = "", {}

    def turn(snap, calls):
        for cid, args in calls:
            ctx[cid] = (snap[-4000:], " ".join(json.dumps(a) for c, a in calls if c != cid)[:2000])

    for m in body.get("messages") or []:                              # Anthropic Messages, OpenAI Chat
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user" and not task:
            task = REMINDER.sub("", _say([b for b in m["content"] if isinstance(b, dict) and b.get("type") == "text"]
                                          if isinstance(m.get("content"), list) else m.get("content")))
        elif m.get("role") == "assistant":
            content = m.get("content")
            calls = [(b.get("id"), _args(b.get("input"))) for b in content if isinstance(b, dict)
                     and b.get("type") == "tool_use"] if isinstance(content, list) else []
            calls += [(c.get("id"), _args((c.get("function") or {}).get("arguments"))) for c in m.get("tool_calls") or []]
            snap = _say([b for b in content if isinstance(b, dict) and b.get("type") != "tool_use"]
                        if isinstance(content, list) else content)
            turn("\n".join(s for s in (m.get("reasoning_content"), snap) if isinstance(s, str)), calls)
    snap, calls = [], []                                              # OpenAI Responses: items in order
    for it in body.get("input") if isinstance(body.get("input"), list) else []:
        if not isinstance(it, dict):
            continue
        if it.get("role") == "user" and not task:
            task = REMINDER.sub("", _say(it.get("content")))
        elif it.get("type") in ("reasoning", "message") or it.get("role") == "assistant":
            if calls:
                turn("\n".join(snap), calls)
                snap, calls = [], []
            snap.append(_say([it]))
        elif it.get("type") in ("function_call", "custom_tool_call", "local_shell_call"):
            calls.append((it.get("call_id") or it.get("id"), _args(it.get("arguments", it.get("input")))))
    turn("\n".join(snap), calls)
    for c in _gemini(body).get("contents") or []:                     # Gemini
        if not isinstance(c, dict):
            continue
        parts = [p for p in c.get("parts") or [] if isinstance(p, dict)]
        if c.get("role") == "user" and not task:
            task = "\n".join(p["text"] for p in parts if isinstance(p.get("text"), str))
        elif c.get("role") == "model":
            calls = [(p["functionCall"].get("id"), _args(p["functionCall"].get("args"))) for p in parts
                     if isinstance(p.get("functionCall"), dict) and p["functionCall"].get("id")]
            turn("\n".join(p["text"] for p in parts if isinstance(p.get("text"), str)), calls)
    return task[:3000], ctx


ANNOUNCE = (
    "[cortexnet] Large source-file reads in this session come back outline-first: every symbol with its real line "
    "span, plus the bodies a coprocessor predicts you need, chosen from what you have written about your goal so "
    "far. Line numbers are real. Before reading a file, say in a sentence what you are looking for; it picks the "
    "bodies. If you need code that is not shown, request all of those line ranges in one turn (parallel ranged "
    "reads), not one at a time. Reading the same file whole a second time returns the full file.")


def announce(body):
    """Append ANNOUNCE to the system prompt, the same bytes on every request, so the prefix cache keeps matching."""
    gem = _gemini(body)
    if "system" in body:
        s = body.get("system")                                        # Anthropic Messages
        body["system"] = (s + "\n\n" + ANNOUNCE if isinstance(s, str) and s else
                          (s if isinstance(s, list) else []) + [{"type": "text", "text": ANNOUNCE}])
        return True
    if isinstance(body.get("input"), list) or "instructions" in body:  # OpenAI Responses
        body["instructions"] = ((body.get("instructions") or "") + "\n\n" + ANNOUNCE).strip()
        return True
    msgs = body.get("messages")
    if isinstance(msgs, list):                                        # OpenAI Chat (an Anthropic body without a
        sys_m = next((m for m in msgs if isinstance(m, dict)      # system prompt is left alone: no system role)
                      and m.get("role") in ("system", "developer") and isinstance(m.get("content"), str)), None)
        if sys_m:
            sys_m["content"] += "\n\n" + ANNOUNCE
        return sys_m is not None
    if isinstance(gem.get("contents"), list):                         # Gemini
        si = gem.setdefault("systemInstruction", {"parts": []})
        si.setdefault("parts", []).append({"text": ANNOUNCE})
        return True
    return False


IDENT_KEYS = {"name", "role", "type", "model", "ToolName", "ServerName"}


def skeleton(x, depth=0):
    """Structure of a request with every value replaced by its type and size (for adapting new agents);
    short identifiers (tool names, roles) are kept."""
    if depth > 7:
        return "..."
    if isinstance(x, dict):
        return {k: v if k in IDENT_KEYS and isinstance(v, str) and len(v) <= 64 else skeleton(v, depth + 1)
                for k, v in x.items()}
    if isinstance(x, list):
        kinds, out = set(), []
        for v in x:
            sk = skeleton(v, depth + 1)
            key = json.dumps(sk, sort_keys=True)
            if key not in kinds:
                kinds.add(key)
                out.append(sk)
        return out[:6] + ([f"... {len(x)} items"] if len(x) > 1 else [])
    if isinstance(x, str):
        return f"str[{len(x)}]"
    return type(x).__name__


# ---------------------------------------------------------------- the rewriter
class Rewriter:
    def __init__(self, state_dir, reader=None, drop_tools=None, rewrite=True, keep=0.8, aware=None, shell=None):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.reader = reader or Reader(state_dir=os.path.join(state_dir, "reader"))
        self.drop = re.compile(drop_tools) if drop_tools else None
        self.rewrite_on, self.keep = rewrite, keep
        # aware: tell the agent the proxy exists (system prompt note) and pick bodies from its stated intent
        self.aware = aware if aware is not None else os.environ.get("CORTEXNET_PROXY_AWARE", "0") == "1"
        # shell: outline long outputs of shell commands that are not file reads (cortexnet/shellout.py)
        self.shell = shell if shell is not None else os.environ.get("CORTEXNET_PROXY_SHELL", "0") == "1"
        self.task, self.ctx = "", {}
        self.memo, self.outlined, self.lock = {}, set(), threading.Lock()
        self.tool_calls, self.last_calls = {}, {}
        self.shapes = int(os.environ.get("CORTEXNET_PROXY_SHAPES", "0"))    # debug: log key skeletons, no values
        self.memo_path = os.path.join(state_dir, "rewrites.jsonl")
        self.usage_path = os.path.join(state_dir, "usage.jsonl")
        if os.path.exists(self.memo_path):
            with open(self.memo_path, encoding="utf8") as f:
                for line in f:
                    r = json.loads(line)
                    self.memo[r["id"]] = r
                    if r["path"]:                    # shell rewrites carry no path
                        self.outlined.add(r["path"])

    @staticmethod
    def _key(path):
        return os.path.normcase(os.path.abspath(path))

    def _outline(self, path, text, cid=None):
        if os.path.splitext(path)[1].lower() not in SRC_EXT or not os.path.isfile(path):
            return None
        if self._key(path) in self.outlined:
            return None                              # second whole read: the agent wants all of it
        body = REMINDER.sub("", text)
        if tokens(body) <= self.reader.full_below:
            return None
        if self.aware:                               # the agent's own words pick the bodies
            snap, siblings = self.ctx.get(cid, ("", ""))
            out = self.reader.read(path, query=(self.task + "\n" + siblings).strip(), snapshot=snap)
        else:
            out = self.reader.read(path)
        if tokens(out) >= self.keep * tokens(body):
            return None                              # small file, bypass, or too little saved
        notes = REMINDER.findall(text)
        head = (f"[cortexnet] {path}: outline-first view, {tokens(out)} of {tokens(body)} tokens. Ask for a line "
                f"range for anything not shown; reading the whole file again returns all of it.\n")
        return head + out + ("\n" + "\n".join(notes) if notes else "")

    def decide(self, cid, call, text, fresh, cwd, stats):
        """Replacement text for one tool result (None keeps the original)."""
        if cid in self.memo:
            return self.memo[cid]["text"]
        if not fresh or not cid:
            return None
        new, kind = None, None
        try:
            kind = classify(call[0], call[1], cwd) if call and text else None
            if kind and kind[0] == "range" and self._key(kind[1]) in self.outlined:
                self.reader.range(kind[1], kind[2], min(kind[3], 10 ** 6))   # label what the outline missed
            elif kind and kind[0] == "whole":
                new = self._outline(kind[1], text, cid)
            elif kind is None and self.shell and call and shell_command(call) and \
                    not is_content_command(shell_command(call)):
                snap, siblings = self.ctx.get(cid, ("", ""))
                words = re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}",
                                   " ".join([shell_command(call), snap, siblings, self.task if self.aware else ""]))
                new = outline_output(REMINDER.sub("", text), words)
                if new is not None:
                    stats["shell_rewrites"] = stats.get("shell_rewrites", 0) + 1
        except Exception as e:                       # never break the agent's request
            print(f"cortexnet-proxy: kept original ({type(e).__name__}: {e})", file=sys.stderr)
            new = None
        file_rw = new is not None and bool(kind)
        rec = {"id": cid, "path": self._key(kind[1]) if file_rw else "", "text": new}
        self.memo[cid] = rec
        if new is not None:
            if file_rw:
                self.outlined.add(rec["path"])
            stats["orig_tokens"] += tokens(text)
            stats["new_tokens"] += tokens(new)
            stats["rewrites"] += 1
            with open(self.memo_path, "a", encoding="utf8") as f:
                f.write(json.dumps(rec) + "\n")
        return new

    def _results(self, body):
        """(holder, key, call id, (name, args), fresh) for every tool result, in any of the three wire formats."""
        out, calls = [], {}
        msgs = body.get("messages")
        if isinstance(msgs, list):
            last = max([i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "assistant"],
                       default=-1)
            for m in msgs:
                if not isinstance(m, dict) or m.get("role") != "assistant":
                    continue
                for b in m.get("content") if isinstance(m.get("content"), list) else []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        calls[b.get("id")] = (b.get("name", ""), _args(b.get("input")))
                for c in m.get("tool_calls") or []:
                    f = c.get("function") or {}
                    calls[c.get("id")] = (f.get("name", ""), _args(f.get("arguments")))
            for i, m in enumerate(msgs):
                if not isinstance(m, dict):
                    continue
                if m.get("role") == "tool" and "content" in m:
                    cid = m.get("tool_call_id")
                    out.append((m, "content", cid, calls.get(cid), i > last, False))
                elif m.get("role") == "user" and isinstance(m.get("content"), list):
                    for b in m["content"]:
                        if isinstance(b, dict) and b.get("type") == "tool_result" and "content" in b:
                            cid = b.get("tool_use_id")
                            out.append((b, "content", cid, calls.get(cid), i > last, bool(b.get("is_error"))))
        items = body.get("input")
        if isinstance(items, list):
            last = max([i for i, it in enumerate(items) if isinstance(it, dict) and it.get("type") not in OUTPUT_TYPES],
                       default=-1)
            for it in items:
                if not isinstance(it, dict):
                    continue
                if it.get("type") in ("function_call", "custom_tool_call"):
                    calls[it.get("call_id")] = (it.get("name", ""), _args(it.get("arguments", it.get("input"))))
                elif it.get("type") == "local_shell_call":
                    calls[it.get("call_id") or it.get("id")] = ("shell", {"command": (it.get("action") or {}).get("command")})
            for i, it in enumerate(items):
                if isinstance(it, dict) and it.get("type") in OUTPUT_TYPES and "output" in it:
                    cid = it.get("call_id") or it.get("id")
                    out.append((it, "output", cid, calls.get(cid), i > last, False))
        contents = _gemini(body).get("contents")
        if isinstance(contents, list):                    # Gemini: functionCall / functionResponse parts
            def parts(c):
                return [p for p in c.get("parts") or [] if isinstance(p, dict)] if isinstance(c, dict) else []
            last = max([i for i, c in enumerate(contents) if isinstance(c, dict) and c.get("role") == "model"],
                       default=-1)
            for i, c in enumerate(contents):
                for j, p in enumerate(parts(c)):
                    fc = p.get("functionCall")
                    if isinstance(fc, dict):
                        calls[fc.get("id") or f"{fc.get('name')}@{i}.{j}"] = (fc.get("name", ""), _args(fc.get("args")))
            for i, c in enumerate(contents):
                for j, p in enumerate(parts(c)):
                    fr = p.get("functionResponse")
                    resp = fr.get("response") if isinstance(fr, dict) else None
                    key = next((k for k in ("output", "content", "result") if isinstance(resp, dict) and k in resp),
                               None)
                    if key:
                        # ids pair a response with its call; without them, the call sits one turn earlier
                        cid = fr.get("id") or next((k for k in calls if k.startswith(f"{fr.get('name')}@{i - 1}.")),
                                                   None)
                        out.append((resp, key, cid, calls.get(cid), i > last, False))
        self.last_calls = calls
        return out

    def _tally(self, body):
        """Every tool call seen this session, by id -> tool name (MCP calls as server.tool)."""
        self._results(body)
        for cid, (name, args) in self.last_calls.items():
            if cid and cid not in self.tool_calls:
                if name in MCP_WRAPPERS and isinstance(args.get("ToolName"), str):
                    name = f"{args.get('ServerName')}.{args['ToolName']}"
                self.tool_calls[cid] = name

    def _drop_tools(self, body):
        tools = body.get("tools")
        if not self.drop or not isinstance(tools, list):
            return 0
        keep = [t for t in tools if not (isinstance(t, dict) and self.drop.search(
            t.get("name") or (t.get("function") or {}).get("name") or ""))]
        body["tools"] = keep
        return len(tools) - len(keep)

    def rewrite(self, raw):
        """Request bytes -> (bytes to send upstream, stats). Any failure sends the original bytes."""
        stats = {"rewrites": 0, "orig_tokens": 0, "new_tokens": 0, "dropped_tools": 0, "model": None}
        try:
            body = json.loads(raw)
        except ValueError:
            return raw, stats
        if not isinstance(body, dict):
            return raw, stats
        stats["model"] = body.get("model")
        if self.shapes > 0:
            self.shapes -= 1
            with open(os.path.join(self.state_dir, "shapes.jsonl"), "a", encoding="utf8") as f:
                f.write(json.dumps(skeleton(body)) + "\n")
        try:
            with self.lock:
                self._tally(body)
        except Exception:
            pass
        if not self.rewrite_on:
            return raw, stats
        try:
            with self.lock:
                stats["dropped_tools"] = self._drop_tools(body)
                cwd = find_cwd(body)
                if self.aware:
                    self.task, self.ctx = intents(body)
                    stats["changed"] = announce(body)
                for holder, key, cid, call, fresh, err in self._results(body):
                    text = _text(holder[key])
                    if text is None or err:
                        continue
                    new = self.decide(cid, call, text, fresh, cwd, stats)
                    if new is not None:
                        _set_text(holder, key, new)
                        stats["changed"] = True
        except Exception as e:
            print(f"cortexnet-proxy: forwarded unchanged ({type(e).__name__}: {e})", file=sys.stderr)
            return raw, stats
        if not stats.pop("changed", False) and not stats["dropped_tools"]:
            return raw, stats
        return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf8"), stats

    def log(self, path, status, stats, usage, seconds):
        rec = dict(ts=time.time(), path=path, status=status, seconds=round(seconds, 3), usage=usage, **stats)
        with self.lock, open(self.usage_path, "a", encoding="utf8") as f:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- usage from (streamed) responses
class Meter:
    """Collects every "usage" object in a response (SSE or JSON) and merges them, keeping the largest counts."""
    RX = re.compile(r'"(?:usage|usageMetadata)"\s*:\s*\{')
    CAP = 8 << 20

    def __init__(self):
        self.buf = bytearray()

    def feed(self, chunk):
        if len(self.buf) < self.CAP:
            self.buf += chunk

    @property
    def usage(self):
        text, dec, merged = self.buf.decode("utf8", "replace"), json.JSONDecoder(), {}
        for m in self.RX.finditer(text):
            try:
                u, _ = dec.raw_decode(text, m.end() - 1)
            except ValueError:
                continue
            for k, v in u.items():
                for kk, vv in (v.items() if isinstance(v, dict) else [(None, v)]):
                    name = f"{k}.{kk}" if kk else k
                    if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                        merged[name] = max(merged.get(name, 0), vv)
        return merged


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        self._forward()

    do_GET = do_PUT = do_PATCH = do_DELETE = do_POST

    def _body(self):
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():     # Go clients stream bodies
            parts = []
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline().strip():                     # trailers
                        pass
                    return b"".join(parts)
                parts.append(self.rfile.read(size))
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _forward(self):
        px, t0 = self.server.px, time.time()
        raw = self._body()
        stats = {}
        if raw and "json" in (self.headers.get("Content-Type") or "json"):
            raw, stats = px.rewriter.rewrite(raw)
        up = urlsplit(px.upstream)
        path = up.path.rstrip("/") + self.path
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers["Accept-Encoding"] = "identity"
        if raw or self.command in ("POST", "PUT", "PATCH"):
            headers["Content-Length"] = str(len(raw))
        conn = (HTTPSConnection if up.scheme == "https" else HTTPConnection)(up.netloc, timeout=900)
        try:
            conn.request(self.command, path, body=raw or None, headers=headers)
            resp = conn.getresponse()
        except OSError as e:
            msg = json.dumps({"error": {"type": "proxy_error", "message": f"cortexnet-proxy: upstream {e}"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        meter = Meter()
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                meter.feed(chunk)
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        finally:
            conn.close()
            if stats:
                px.rewriter.log(self.path, resp.status, stats, meter.usage, time.time() - t0)
                if stats.get("rewrites") and not px.quiet:
                    print(f"cortexnet-proxy: {stats['rewrites']} read(s) {stats['orig_tokens']} -> "
                          f"{stats['new_tokens']} tokens", file=sys.stderr)


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):  # clients hanging up are normal
            super().handle_error(request, client_address)


class Proxy:
    def __init__(self, upstream, rewriter, host="127.0.0.1", port=8787):
        self.upstream, self.rewriter, self.quiet = upstream, rewriter, False
        self.server = Server((host, port), Handler)
        self.server.px = self
        self.port = self.server.server_address[1]
        self.url = f"http://{host}:{self.port}"

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


# ---------------------------------------------------------------- CLI
def totals(usage_path, since=0.0):
    """Summed usage fields, rewrite sizes and the request count ("requests") from a usage log."""
    tot = {"requests": 0}
    if os.path.exists(usage_path):
        with open(usage_path, encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                if r["ts"] < since:
                    continue
                tot["requests"] += 1
                for k in ("rewrites", "orig_tokens", "new_tokens"):
                    tot[k] = tot.get(k, 0) + r.get(k, 0)
                for k, v in (r.get("usage") or {}).items():
                    tot[k] = tot.get(k, 0) + v
    return tot


def summarize(usage_path, since=0.0):
    tot = totals(usage_path, since)
    n, g = tot["requests"], tot.get
    weighted = g("input_tokens", 0) + 1.25 * g("cache_creation_input_tokens", 0) + 0.1 * g("cache_read_input_tokens", 0)
    lines = [f"requests {n}, rewrites {g('rewrites', 0)}: {g('orig_tokens', 0)} -> {g('new_tokens', 0)} tokens "
             f"({g('orig_tokens', 0) - g('new_tokens', 0)} saved at the read)"]
    if "cache_read_input_tokens" in tot or "cache_creation_input_tokens" in tot:
        lines.append(f"anthropic: input {g('input_tokens', 0)}, cache write {g('cache_creation_input_tokens', 0)}, "
                     f"cache read {g('cache_read_input_tokens', 0)}, output {g('output_tokens', 0)}; "
                     f"cache-weighted input {weighted:,.0f}")
    elif "prompt_tokens" in tot:
        lines.append(f"openai chat: prompt {g('prompt_tokens', 0)} (cached {g('prompt_tokens_details.cached_tokens', 0)}),"
                     f" completion {g('completion_tokens', 0)}")
    elif "input_tokens" in tot:
        lines.append(f"responses: input {g('input_tokens', 0)} (cached {g('input_tokens_details.cached_tokens', 0)}), "
                     f"output {g('output_tokens', 0)}")
    if "promptTokenCount" in tot:
        lines.append(f"gemini: prompt {g('promptTokenCount', 0)} (cached {g('cachedContentTokenCount', 0)}), "
                     f"output {g('candidatesTokenCount', 0)}, thoughts {g('thoughtsTokenCount', 0)}")
    return "\n".join(lines)


def _rewriter(a):
    from .mcp_server import options
    state = os.path.join(a.state, "proxy")
    reader = Reader(state_dir=os.path.join(state, "reader"), **options())
    on = not a.passthrough and os.environ.get("CORTEXNET_PROXY_REWRITE", "1") != "0"
    return Rewriter(state, reader=reader, drop_tools=a.drop_tools, rewrite=on, aware=a.aware or None,
                    shell=a.shell or None)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cortexnet-proxy", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state", default=os.environ.get("CORTEXNET_STATE") or os.path.join(os.getcwd(), ".cortexnet"))
    for name in ("serve", "run"):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--upstream", default=os.environ.get("CORTEXNET_UPSTREAM", "https://api.anthropic.com"))
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8787 if name == "serve" else 0)
        p.add_argument("--drop-tools", help="regex: tool definitions whose name matches are removed (static)")
        p.add_argument("--passthrough", action="store_true", help="forward unchanged, log usage only (A/B baseline)")
        p.add_argument("--aware", action="store_true", help="tell the agent about outline-first reads (system "
                       "prompt note) and choose bodies from its own reasoning and task (CORTEXNET_PROXY_AWARE=1)")
        p.add_argument("--shell", action="store_true", help="outline long shell outputs: keep start, end, errors and "
                       "lines matching the agent's intent (CORTEXNET_PROXY_SHELL=1)")
    sub.choices["run"].add_argument("agent", nargs=argparse.REMAINDER, help="-- <agent command line>")
    sub.add_parser("stats", parents=[common])
    a = ap.parse_args(argv)
    if a.cmd == "stats":
        print(summarize(os.path.join(a.state, "proxy", "usage.jsonl")))
        return 0
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print("cortexnet-proxy: warning, listening beyond loopback; request bodies hold code and keys",
              file=sys.stderr)
    px = Proxy(a.upstream, _rewriter(a), a.host, a.port)
    mode = "passthrough" if not px.rewriter.rewrite_on else "rewriting"
    print(f"cortexnet-proxy: {px.url} -> {a.upstream} ({mode})", file=sys.stderr)
    if a.cmd == "serve":
        try:
            px.server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    agent = a.agent[1:] if a.agent[:1] == ["--"] else a.agent
    if not agent:
        ap.error("run needs an agent command after --")
    env = dict(os.environ, ANTHROPIC_BASE_URL=px.url, OPENAI_BASE_URL=px.url + "/v1", CLOUD_CODE_URL=px.url)
    if os.path.splitext(os.path.basename(agent[0]))[0].lower() == "claude":
        # an env block in Claude Code's settings outranks the process env; --settings outranks user settings
        override = os.path.join(px.rewriter.state_dir, "claude-settings.json")
        with open(override, "w", encoding="utf8") as f:
            json.dump({"env": {"ANTHROPIC_BASE_URL": px.url}}, f)
        agent = agent[:1] + ["--settings", override] + agent[1:]
    t0, px.quiet = time.time(), True             # the agent owns the terminal; summary on exit
    px.start()
    try:
        code = subprocess.call(agent, env=env, shell=os.name == "nt")
    except KeyboardInterrupt:
        code = 130
    finally:
        px.stop()
    print(summarize(px.rewriter.usage_path, since=t0), file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
