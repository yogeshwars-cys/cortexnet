"""Cortex CLI: a terminal coding agent with CortexNET built in.

Every file read goes through CortexNET (outline plus the bodies the include model and the coprocessor pick, a
connected-symbol menu), the conversation fades on a watermark near the context limit, and long log-like command
output is outlined. Edits and commands ask before they run unless --yes. Paths stay inside the repository.

  cortex                                            # REPL in the current repo, llama-server on :8080
  cortex "make the failing test pass"               # one task, then exit
  cortex --api anthropic --base https://api.meta.ai --model muse-spark-1.3 --key-env LLAMA_API_KEY

Backends: --api openai (any OpenAI-compatible /v1/chat/completions: llama-server, vLLM, Ollama, OpenAI) or
--api anthropic (/v1/messages: Anthropic, or the Meta Model API's Anthropic-compatible endpoint).
"""
import argparse
import contextlib
import difflib
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from .card import TaskCard
from .fade import MARK as FADE_MARK, preview as fade_preview, watermark_fade
from .prefetch import Prefetcher
from .outline import outline, symbols, tokens
from .reader import Reader, SRC_EXT
from .cli_ui import PlainUI, make_ui
from .shellout import is_content_command, outline_output

SYSTEM = """You are a coding agent working in the repository at {root}. Every token you read or write stays in a small
context window, so be economical.
- If a message does not need the code (a greeting, a general question), answer it directly, without tools.
- Otherwise work in small steps with the tools and never guess file contents. Reads are outline-first: read_file
  returns the file's outline (every function/class with its line range) plus the bodies most likely relevant, so say
  in `query` what you are looking for. Use read_lines for other lines, and cortex_approve for symbols offered under
  [connected]. Do not read a file again while its earlier read is still above; use read_lines instead.
- An outline shows names and signatures, not behavior. Never state what code does, which dependencies or commands
  exist, or cite file:line, unless you saw those lines (read_lines, a shown body, or a file read whole). If you have
  not read it, read it or say you have not checked.
- Before an edit, read the exact lines you change. edit_file replaces one exact snippet. Use run for tests/commands.
- Before tool calls, write at most one short sentence, or nothing. Do not repeat what you already said.
- Old reads may be shortened later ([faded ...]); then read the file again if you need it.
- Keep your task card with `note`: plan steps for multi-step tasks, each finding with its file:line once you have
  seen it, and `want` (what you will look at next) whenever you change direction. The card is never faded, so
  write down what you will need later instead of relying on old reads.
- Use `scout` for broad questions (where is X handled, which files touch Y) to keep raw reads out of your context.
- When the task is done, reply with a short summary of what you changed and why.
Top-level entries: {listing}"""


def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


S, I = {"type": "string"}, {"type": "integer"}
TOOLS = [
    _fn("read_file", "Read a file: small files whole, large files as an outline plus the bodies most likely relevant "
        "to `query`.", {"path": S, "query": S}, ["path"]),
    _fn("read_lines", "Lines start..end of a file, with line numbers.", {"path": S, "start": I, "end": I},
        ["path", "start", "end"]),
    _fn("cortex_approve", "Pull the bodies of symbols offered in a [connected] menu, by id (e.g. [\"#3\"]).",
        {"ids": {"type": "array", "items": S}}, ["ids"]),
    _fn("grep", "Search files for a regular expression; returns path:line: text (max 50).",
        {"pattern": S, "path": {"type": "string", "description": "file or directory, default the repository"}},
        ["pattern"]),
    _fn("list_dir", "List a directory.", {"path": S}, []),
    _fn("edit_file", "Replace one exact occurrence of `old` with `new` in a file.", {"path": S, "old": S, "new": S},
        ["path", "old", "new"]),
    _fn("write_file", "Create or overwrite a file with `content`.", {"path": S, "content": S}, ["path", "content"]),
    _fn("run", "Run a shell command in the repository (tests, scripts, git). Returns exit code and output.",
        {"command": S}, ["command"]),
    _fn("note", "Update your task card (your working memory; it is never faded, raw reads are): record a finding with "
        "file:line you have seen, what you did, plan steps, check off steps, the phase, and what you want next. "
        "`want` names the files/symbols you will look at next; CortexNET prepares them in advance.",
        {"finding": S, "done": S, "want": S, "plan": {"type": "array", "items": S},
         "check": {"type": "array", "items": I}, "phase": {"type": "string", "enum": ["explore", "plan", "edit",
                                                                                    "verify"]}}, []),
    _fn("scout", "Ask a sub-agent a focused question about the code (where is X, how does Y work). It reads in its "
        "own fresh context and returns a short answer with file:line; its reads never enter your context.",
        {"question": S}, ["question"]),
]
SCOUT_TOOLS = [t for t in TOOLS if t["function"]["name"] in ("read_file", "read_lines", "grep", "list_dir",
                                                               "cortex_approve")]
SCOUT_SYSTEM = """You are a scout for a coding agent, in the repository at {root}. Answer the question by reading the
code with the tools (outline-first reads; read_lines for bodies). Reply with a short answer: at most 12 lines, each
claim with the file:line you saw. Never state what you did not read."""
PLAN_LINE = """
Plan mode (read-only): you cannot edit files or run commands. Investigate, then answer with a concrete plan: the files
and lines to change, what to change, and how to verify it."""
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".cortexnet", ".mypy_cache", ".pytest_cache"}
READ_TOOLS = ("read_file", "read_lines", "grep", "list_dir")     # results that can be pointed back to
MUTATING = ("edit_file", "write_file", "run")                     # may change files: grep/list answers go stale
SECTION = re.compile(r"^(?=--- (?:lines|x-ray)|\[connected\])", re.M)


def sections(out):
    """A read's parts: [header + outline (or the whole small file), body/x-ray sections..., connected menu]."""
    return [p for p in SECTION.split(out) if p] or [out]


# ---------------------------------------------------------------- tools

class Workspace:
    """The repository the agent may touch. confirm(question, detail) -> bool decides edits and commands."""

    def __init__(self, root, reader=None, confirm=None, log_cases=True):
        self.root = os.path.realpath(root)
        self.reader = reader or Reader(os.path.join(self.root, ".cortexnet", "agent"), split_tokens=600)
        self.confirm = confirm or (lambda q, d: False)
        self.task, self.snapshot, self.intent = "", "", ""   # intent: the agent's latest `want` (its task card)
        self.saved = 0                                  # tokens CortexNET reads left out, vs whole files
        self.lock = threading.RLock()                   # the prefetcher warms the reader from a background thread
        self.log_cases, self.last_read = log_cases, {}  # training cases for `cortex learn`: path -> read id
        self.cases_dir = os.path.join(self.root, ".cortexnet", "agent")
        self.undo_stack = []                            # [(rel path, text before the change, or None: it was new)]
        self.map_cache = {}                             # path -> (mtime, top-level names) for /map
        self._edits = _takes_edit(self.confirm)          # the confirm can return revised text (y/n/e)

    def _log(self, ev):
        """Training data for the coprocessor: each read (what was asked, with which words) and the ranges that
        followed it (what was actually needed). File texts are stored once by hash. Local only (.cortexnet/)."""
        if not self.log_cases:
            return
        try:
            os.makedirs(os.path.join(self.cases_dir, "texts"), exist_ok=True)
            with open(os.path.join(self.cases_dir, "cases.jsonl"), "a", encoding="utf8") as f:
                f.write(json.dumps(dict(ev, t=time.time())) + "\n")
        except OSError:
            pass

    def _log_read(self, p, query):
        with open(p, encoding="utf8", errors="replace") as f:
            text = f.read()
        h = hashlib.sha1(text.encode("utf8", "replace")).hexdigest()[:16]
        tp = os.path.join(self.cases_dir, "texts", h + ".txt")
        if self.log_cases and not os.path.exists(tp):
            try:
                os.makedirs(os.path.dirname(tp), exist_ok=True)
                with open(tp, "w", encoding="utf8", newline="") as f:
                    f.write(text)
            except OSError:
                pass
        rid = f"{time.time():.6f}"
        self.last_read[p] = rid
        self._log(dict(ev="read", id=rid, path=self.rel(p), text=h, q=self.task, query=query, h=self.snapshot,
                       intent=self.intent))

    def path(self, p):
        p = os.path.realpath(os.path.join(self.root, (p or ".").strip()))
        if p != self.root and not p.startswith(self.root + os.sep):
            raise ValueError("path outside the repository")
        return p

    def rel(self, p):
        return os.path.relpath(p, self.root).replace("\\", "/")

    def call(self, name, a):
        try:
            return getattr(self, "t_" + name)(**a) if hasattr(self, "t_" + name) else f"unknown tool {name}"
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}"
        except (OSError, ValueError, re.error, UnicodeDecodeError) as e:
            return f"error: {type(e).__name__}: {e}"

    def t_read_file(self, path, query=""):
        p = self.path(path)
        q = "\n".join(x for x in (self.task, self.intent, query or "") if x).strip()
        with self.lock:
            out = self.reader.read(p, q, snapshot=self.snapshot)
        with open(p, encoding="utf8", errors="replace") as f:
            self.saved += max(0, tokens(f.read()) - tokens(out))
        self._log_read(p, query or "")
        if " Outline (" in out.split("\n", 1)[0] and not re.search(r"^--- (?:lines|x-ray)", out, re.M):
            # outline-only reads invited confident summaries of code that was never read (first user session)
            out += ("\n[only the outline is shown, no bodies: use read_lines on the symbols you need before saying "
                    "what they do]")
        return out

    def t_read_lines(self, path, start, end):
        p = self.path(path)
        with self.lock:
            out = self.reader.range(p, int(start), int(end))
        if p in self.last_read:                       # the label: what the agent needed after that read
            self._log(dict(ev="range", read=self.last_read[p], path=self.rel(p), start=int(start), end=int(end)))
        return out

    def t_cortex_approve(self, ids):
        with self.lock:
            return self.reader.approve([ids] if isinstance(ids, str) else list(ids))

    def files(self, base):
        if os.path.isfile(base):
            yield base
            return
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for f in sorted(fns):
                yield os.path.join(dp, f)

    def t_grep(self, pattern, path="."):
        rx, out = re.compile(pattern), []
        for f in self.files(self.path(path)):
            if os.path.getsize(f) > 2_000_000:
                continue
            try:
                with open(f, encoding="utf8") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            out.append(f"{self.rel(f)}:{i}: {line.rstrip()[:200]}")
                            if len(out) >= 50:
                                return "\n".join(out) + "\n[50 matches shown; narrow the pattern or the path]"
            except UnicodeDecodeError:
                continue                                 # binary file
        return "\n".join(out) or "no matches"

    def t_list_dir(self, path="."):
        p = self.path(path)
        items = sorted(os.listdir(p))
        return "\n".join(f"{n}/" if os.path.isdir(os.path.join(p, n)) else n for n in items if n not in SKIP_DIRS)

    def ask(self, question, detail, editable=None):
        """The user's answer: True, False, or (when `editable` is given and the UI can open an editor) revised text."""
        if editable is not None and self._edits:
            return self.confirm(question, detail, editable)
        return self.confirm(question, detail)

    def _keep_undo(self, p):
        old = None
        if os.path.exists(p):
            with open(p, encoding="utf8", errors="replace", newline="") as f:
                old = f.read()
        self.undo_stack.append((self.rel(p), old))
        del self.undo_stack[:-20]

    def t_edit_file(self, path, old, new):
        p = self.path(path)
        with open(p, encoding="utf8", newline="") as f:
            text = f.read()
        crlf = "\r\n" in text                            # the model writes \n: keep a CRLF file CRLF
        fix = (lambda s: s.replace("\r\n", "\n").replace("\n", "\r\n")) if crlf else (lambda s: s)
        old, new = fix(old), fix(new)
        n = text.count(old) if old else 0
        if n != 1:
            return f"error: `old` occurs {n} times in {self.rel(p)}; it must occur exactly once (read the lines first)"
        revised = False
        for _ in range(4):                                # e: the user rewrites `new`, then sees the diff again
            after = text.replace(old, new, 1)
            diff = "".join(difflib.unified_diff(text.splitlines(True), after.splitlines(True), self.rel(p),
                                                self.rel(p)))
            ans = self.ask(f"apply this edit to {self.rel(p)}?", diff, editable=new)
            if not isinstance(ans, str):
                break
            new, revised = fix(ans), True
        if ans is not True:
            return "the user declined this edit"
        self._keep_undo(p)
        with open(p, "w", encoding="utf8", newline="") as f:
            f.write(after)
        note = " (the user revised your change before applying it)" if revised else ""
        return f"edited {self.rel(p)}{note}:\n{diff}"

    def t_write_file(self, path, content):
        p = self.path(path)
        what = "overwrite" if os.path.exists(p) else "create"
        revised = False
        for _ in range(4):
            ans = self.ask(f"{what} {self.rel(p)} ({len(content.splitlines())} lines)?", content[:2000],
                           editable=content)
            if not isinstance(ans, str):
                break
            content, revised = ans, True
        if ans is not True:
            return "the user declined this write"
        self._keep_undo(p)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf8", newline="") as f:
            f.write(content)
        note = ", revised by the user before writing" if revised else ""
        return f"wrote {self.rel(p)} ({len(content.splitlines())} lines{note})"

    def undo(self):
        """Put back the file the last edit_file/write_file changed. A message, or None when there is nothing to undo."""
        if not self.undo_stack:
            return None
        rel, old = self.undo_stack.pop()
        p = self.path(rel)
        if old is None:
            if os.path.exists(p):
                os.remove(p)
            return f"removed {rel} (the agent had created it)"
        with open(p, "w", encoding="utf8", newline="") as f:
            f.write(old)
        return f"restored {rel} to before the agent's last change to it"

    def undo_diff(self):
        """What /undo would change, as a unified diff (None when there is nothing to undo)."""
        if not self.undo_stack:
            return None
        rel, old = self.undo_stack[-1]
        p = self.path(rel)
        now = ""
        if os.path.exists(p):
            with open(p, encoding="utf8", errors="replace", newline="") as f:
                now = f.read()
        return "".join(difflib.unified_diff(now.splitlines(True), (old or "").splitlines(True), f"{rel} (now)",
                                            f"{rel} (after /undo)")) or f"{rel}: no difference"

    def symbol_list(self, path):
        """[(start, end, label)] in the outline's order; /outline numbers them from 1 for `read N`."""
        head = self.outline_of(path)
        if head is None:
            raise ValueError(f"cannot outline {path}")
        rows = []
        for line in head.split("\n")[1:]:
            m = re.match(r"\s*L(\d+)-(\d+) ", line)
            if m:
                rows.append((int(m[1]), int(m[2]), line.rstrip()))
        return rows

    def repo_map(self, base=".", max_files=40, max_names=8):
        """[(rel path, number of top-level symbols, first names)] of source files under base, for /map."""
        rows = []
        for f in self.files(self.path(base)):
            if os.path.splitext(f)[1].lower() not in SRC_EXT:
                continue
            try:
                mt = os.path.getmtime(f)
                if os.path.getsize(f) > 1_000_000:
                    continue
                hit = self.map_cache.get(f)
                if hit is None or hit[0] != mt:
                    with open(f, encoding="utf8", errors="replace") as fh:
                        names = [s.name for s in symbols(fh.read(), f) if s.depth == 0
                                 and s.kind in ("def", "class", "const", "heading")]
                    hit = self.map_cache[f] = (mt, names)
            except (OSError, ValueError, SyntaxError, RecursionError):
                continue
            rows.append((self.rel(f), len(hit[1]), hit[1][:max_names]))
            if len(rows) >= max_files:
                break
        return rows

    def t_run(self, command, timeout=300):
        if not self.confirm(f"run: {command}", ""):
            return "the user declined this command"
        try:
            r = subprocess.run(command, shell=True, cwd=self.root, capture_output=True, text=True, timeout=timeout,
                               encoding="utf8", errors="replace")
            text, code = (r.stdout or "") + (r.stderr or ""), r.returncode
        except subprocess.TimeoutExpired as e:
            text, code = (e.stdout or "") if isinstance(e.stdout, str) else "", f"timeout after {timeout}s"
        short = None if is_content_command(command) else outline_output(text, re.findall(r"\w{3,}", command +
                                                                                         " " + self.task))
        if short is None and len(text) > 12000:              # content output: keep it, but bounded
            text = text[:6000] + f"\n[... {len(text) - 12000} characters omitted ...]\n" + text[-6000:]
        return f"exit code {code}\n{short or text}"

    def outline_of(self, path):
        try:
            p = self.path(path)
            with open(p, encoding="utf8", errors="replace") as f:
                text = f.read()
        except (OSError, ValueError):
            return None
        return outline(text, self.rel(p), symbols(text, p, split_tokens=600))


# ---------------------------------------------------------------- model backends

def _post(url, body, headers, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(), dict({"Content-Type": "application/json"},
                                                                              **headers))
            return json.loads(urllib.request.urlopen(req, timeout=600).read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:400]
            if e.code not in (429, 502, 503, 504) or i == tries - 1:
                raise RuntimeError(f"HTTP {e.code}: {msg}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if i == tries - 1:
                raise RuntimeError(f"cannot reach the model server: {e}") from None
        time.sleep(3 * (i + 1))


class OpenAIChat:
    def __init__(self, base, model, key=None):
        self.url, self.model, self.key = base.rstrip("/") + "/chat/completions", model, key

    def chat(self, msgs, tools):
        r = _post(self.url, {"model": self.model, "messages": msgs, "tools": tools, "temperature": 0},
                  {"Authorization": "Bearer " + self.key} if self.key else {})
        m = r["choices"][0]["message"]
        calls = []
        for c in m.get("tool_calls") or []:
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except ValueError:
                args = {}
            calls.append({"id": c.get("id") or f"call_{len(calls)}", "name": c["function"]["name"], "args": args})
        u = r.get("usage") or {}
        sent = u.get("prompt_tokens", 0)
        t = r.get("timings") or {}                      # llama-server: prompt_n = tokens actually computed
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or \
            (max(0, sent - t["prompt_n"]) if "prompt_n" in t else 0)
        self.last = dict(sent=sent, cached=cached, out=u.get("completion_tokens", 0))
        return m.get("content") or "", calls, sent


class AnthropicMessages:
    """Messages API. The conversation is kept in Chat form and converted on each call."""

    def __init__(self, base, model, key=None, max_tokens=4096):
        self.url, self.model, self.key, self.max_tokens = base.rstrip("/") + "/v1/messages", model, key, max_tokens

    @staticmethod
    def convert(msgs):
        system, out = "", []
        for m in msgs:
            if m["role"] == "system":
                system = m["content"]
            elif m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                            "input": json.loads(c["function"]["arguments"] or "{}")} for c in m.get("tool_calls") or []]
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no text)"}]})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)            # results of one turn go in one user message
                else:
                    out.append({"role": "user", "content": [block]})
        merged = []                                             # roles must alternate: join same-role neighbours
        for m in out:                                           # (tool results then a user message, after `#3`)
            if merged and merged[-1]["role"] == m["role"]:
                blocks = lambda c: c if isinstance(c, list) else [{"type": "text", "text": c}]  # noqa: E731
                merged[-1] = {"role": m["role"], "content": blocks(merged[-1]["content"]) + blocks(m["content"])}
            else:
                merged.append(m)
        return system, merged

    @staticmethod
    def cacheable(system, conv, tools):
        """Prompt-cache breakpoints: the tool list, the system prompt and the newest message. The conversation is
        append-only between fades, so each request's prefix up to the previous breakpoint is read from the cache."""
        eph = {"type": "ephemeral"}
        tools = [dict(t) for t in tools]
        if tools:
            tools[-1]["cache_control"] = eph
        sys_blocks = [{"type": "text", "text": system, "cache_control": eph}] if system else []
        conv = [dict(m) for m in conv]
        if conv:
            last = conv[-1]
            blocks = [{"type": "text", "text": last["content"]}] if isinstance(last["content"], str) else \
                [dict(b) for b in last["content"]]
            blocks[-1]["cache_control"] = eph
            last["content"] = blocks
        return sys_blocks, conv, tools

    def chat(self, msgs, tools):
        system, conv = self.convert(msgs)
        tl = [{"name": t["function"]["name"], "description": t["function"]["description"],
               "input_schema": t["function"]["parameters"]} for t in tools]
        system, conv, tl = self.cacheable(system, conv, tl)
        r = _post(self.url, {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": conv,
                             "tools": tl}, {"x-api-key": self.key or "", "anthropic-version": "2023-06-01"})
        text = "\n".join(b.get("text", "") for b in r.get("content") or [] if b.get("type") == "text")
        calls = [{"id": b["id"], "name": b["name"], "args": b.get("input") or {}}
                 for b in r.get("content") or [] if b.get("type") == "tool_use"]
        u = r.get("usage") or {}
        cached = u.get("cache_read_input_tokens", 0)
        sent = u.get("input_tokens", 0) + cached + u.get("cache_creation_input_tokens", 0)
        self.last = dict(sent=sent, cached=cached, out=u.get("output_tokens", 0))
        return text, calls, sent


# ---------------------------------------------------------------- the loop

def msg_tokens(msgs):
    return sum(tokens(str(m.get("content") or "")) + sum(tokens(c["function"]["arguments"]) + 8
                                                         for c in m.get("tool_calls") or []) for m in msgs)


class Agent:
    def __init__(self, backend, ws, ctx=8192, mark=0.85, low=0.5, max_steps=30, ui=None, prefetch=True, depth=0,
                 system=SYSTEM, tools=None, hints=True):
        self.backend, self.ws, self.ctx, self.mark, self.low = backend, ws, ctx, mark, low
        self.fade_kw = {}                                   # fade order / on_fade hook
        self.files = None                                   # spiking file manager (cortexnet/filesnn.py)
        self.files_indexed = False                          # the repository index has neurons
        self.max_steps, self.ui, self.depth, self.tools = max_steps, ui or PlainUI(), depth, tools or TOOLS
        listing = ", ".join(ws.t_list_dir(".").split("\n")[:40])
        self.msgs = [{"role": "system", "content": system.format(root=ws.root, listing=listing)}]
        self.card = TaskCard()
        self.prefetch = Prefetcher(ws) if prefetch else None
        self.scouts, self.said = 0, ""
        self.ratio, self.faded, self.last_in = 1.0, 0, 0
        self.recs, self.epoch, self.reused = [], 0, 0     # results still in context; edits/commands bump epoch
        self.usage = dict(sent=0, cached=0, out=0)        # totals over the session, as the server reports them
        self.pinned = set()                                # paths whose reads fade never shrinks (/pin)
        self.pending = []                                  # notes for the model, sent with the next user message
        self.injected = 0                                  # tool calls the user made at the prompt (#3, read 2,4)
        self.hints, self.extracted = hints, None           # user extract: note for the model; the last Extract
        self.primed, self.bridge_threads = 0, []           # request texts the coprocessor encoded ahead of reads

    # ------------------------------------------------ results already in context: pointers and deltas

    def _mtime(self, path):
        try:
            return os.path.getmtime(self.ws.path(path))
        except (OSError, ValueError):
            return None

    def _live(self, rec):
        """The earlier result is still in the conversation, not faded."""
        m = rec["msg"]
        return any(x is m for x in self.msgs) and not (m.get("content") or "").startswith(FADE_MARK)

    def _valid(self, rec):
        if rec["name"] in ("grep", "list_dir"):
            return rec["epoch"] == self.epoch                 # any edit or command may have changed the answer
        return rec["mtime"] is not None and rec["mtime"] == self._mtime(rec["args"].get("path"))

    def run_tool(self, name, args, user=False):
        """(output, record or None). An identical read/grep/list whose earlier result is still in context and still
        true returns a pointer; a re-read of an unchanged file returns only the parts not shown before.
        user: called by the user at the prompt, not the model (any tool, whatever this agent may call)."""
        if not user and name not in {t["function"]["name"] for t in self.tools}:
            return f"error: {name} is not available here", None   # a scout cannot edit, run or nest
        if name == "note":
            keys = ("finding", "done", "want", "plan", "check", "phase")
            out = self.card.apply(**{k: v for k, v in args.items() if k in keys})
            self.ws.intent = self.card.want                  # the latent request steers the next reads
            return out, None
        if name == "scout":
            return self.scout(str(args.get("question", ""))), None
        if name in MUTATING:
            self.epoch += 1
        if name == "read_file" and self.prefetch is not None:
            self.prefetch.observe(args.get("path", ""))
        if self.files is not None and args.get("path") and name in FILE_TOOLS:
            self._file_used(args["path"])
        key = name + json.dumps(args, sort_keys=True)
        if name in READ_TOOLS:
            for rec in reversed(self.recs):
                if rec["key"] == key and self._live(rec) and self._valid(rec):
                    self.reused += 1
                    return f"[same result as your earlier {name} call above; nothing has changed since]", None
        out = self.ws.call(name, args)
        if name not in READ_TOOLS or out.startswith("error"):
            return out, None
        rec = dict(name=name, key=key, args=args, epoch=self.epoch, sections=sections(out),
                   mtime=self._mtime(args["path"]) if args.get("path") else None)
        if name == "read_file":
            prev = [r for r in self.recs if r["name"] == "read_file" and r["args"].get("path") == args.get("path")
                    and self._live(r) and self._valid(r)]
            if prev:
                shown = {s for r in prev for s in r["sections"][1:]} | {r["sections"][0] for r in prev}
                new = [s for s in rec["sections"][1:] if s not in shown]
                head = rec["sections"][0]
                if head not in shown:                         # the file changed shape: show it all
                    return out, rec
                self.reused += 1
                self.ws.saved += tokens(out)
                note = f"[{args['path']} is unchanged since your read above, and its outline is there"
                if not new:
                    return note + "; nothing new for this query: use read_lines for other lines]", rec
                delta = note + "; the new parts for this query:]\n" + "\n".join(new)
                self.ws.saved -= tokens(delta)
                return delta, rec
        return out, rec

    def scout(self, question, max_steps=8, cap_chars=4800):
        """A sub-agent with read-only tools and its own fresh context; only its short answer comes back."""
        if self.depth > 0:
            return "error: a scout cannot start another scout"
        if not question.strip():
            return "error: scout needs a question"
        saved = (self.ws.task, self.ws.intent, self.ws.snapshot)
        sub = Agent(self.backend, self.ws, ctx=self.ctx, mark=self.mark, low=self.low, max_steps=max_steps,
                    ui=self.ui, prefetch=False, depth=1, system=SCOUT_SYSTEM, tools=SCOUT_TOOLS)
        sub.prefetch = self.prefetch                       # same index and warm cache, no second builder thread
        self.scouts += 1
        enter = getattr(self.ui, "nested", None)
        try:
            with (enter(question) if enter else contextlib.nullcontext()):
                answer = sub.turn(question)
        finally:
            self.ws.task, self.ws.intent, self.ws.snapshot = saved
            for k, v in sub.usage.items():
                self.usage[k] = self.usage.get(k, 0) + v
            self.reused += sub.reused
        if len(answer) > cap_chars:
            answer = answer[:cap_chars] + "\n[scout answer cut]"
        return "scout answer:\n" + answer

    def compact(self):
        """/compact: one fade batch now, aiming at half the current context (returns the messages changed)."""
        used = self.context()
        n = watermark_fade(self.msgs, used, self.ctx, self.ws.outline_of, mark=0.0, low=0.5 * used / self.ctx,
                           scale=self.ratio, pinned=self.pinned, **self.fade_kw)
        self.faded += n
        return n

    def fade_preview(self):
        """What the next watermark fade would shrink, in order (nothing changes)."""
        return fade_preview(self.msgs, self.context(), self.ctx, self.ws.outline_of, mark=self.mark, low=self.low,
                            scale=self.ratio, pinned=self.pinned)

    def inject(self, name, args):
        """Run a tool the user asked for at the prompt (`#3`, `read 2,4 calc.py`, /test) without a model call. The
        request, the call and its result are appended to the conversation (append-only, roles alternate), so the next
        turn sees them and fade treats them like the agent's own reads."""
        self.injected += 1
        cid = f"user_{self.injected}"
        self.msgs.append({"role": "user", "content": f"(I ran {name} myself; the result is below)"})
        self.msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        out, rec = self.run_tool(name, args, user=True)
        self.ui.step(name, args, out)
        m = {"role": "tool", "tool_call_id": cid, "name": name, "content": out}
        self.msgs.append(m)
        if rec:
            self.recs.append(dict(rec, msg=m))
        return out

    def undo(self):
        """/undo: put back the last edited file and tell the model with the next message."""
        msg = self.ws.undo()
        if msg:
            self.epoch += 1                                # grep/list answers above may be stale now
            self.pending.append(f"[the user reverted a change with /undo: {msg}]")
        return msg

    def read_rows(self):
        """/reads: [(path, what was shown, state, tokens)] for each read still in the conversation, oldest first."""
        from .cli_ui import summarize
        rows = []
        for r in self.recs:
            if r["name"] not in ("read_file", "read_lines"):
                continue
            content = r["msg"].get("content") or ""
            alive = any(x is r["msg"] for x in self.msgs)
            if not alive:
                continue
            state = "faded" if content.startswith(FADE_MARK) else "stale" if not self._valid(r) else "live"
            what = summarize(r["name"], r["args"], content) if state != "faded" else content[len(FADE_MARK):-1][:60]
            path = r["args"].get("path", "")
            if r["name"] == "read_lines":
                path += f":{r['args'].get('start')}-{r['args'].get('end')}"
            rows.append((path + (" (pinned)" if r["args"].get("path") in self.pinned else ""), what, state,
                         tokens(content)))
        return rows

    def clear(self):
        """/clear: a fresh conversation (the system prompt, a new card); usage totals and the reader's state stay."""
        del self.msgs[1:]
        self.card, self.recs, self.said, self.pending = TaskCard(), [], "", []
        self.ws.intent = self.ws.snapshot = ""

    # ------------------------------------------------ sessions (--resume, --fork)

    def state(self):
        """Everything needed to continue this conversation later, as JSON-safe data."""
        index = {id(m): i for i, m in enumerate(self.msgs)}
        recs = [dict({k: v for k, v in r.items() if k != "msg"}, msg=index[id(r["msg"])]) for r in self.recs
                if id(r["msg"]) in index]
        return dict(version=1, msgs=self.msgs, recs=recs, card=vars(self.card), pinned=sorted(self.pinned),
                    pending=self.pending, usage=self.usage, epoch=self.epoch, faded=self.faded, reused=self.reused,
                    scouts=self.scouts, ratio=self.ratio, injected=self.injected, saved=self.ws.saved,
                    undo=self.ws.undo_stack, time=time.time())

    def restore(self, st):
        """Continue a saved conversation. The system prompt is this run's (tools or mode may differ); faded reads
        stay faded (pointers, not refetched)."""
        if st.get("version") != 1:
            raise ValueError("unknown session format")
        system = self.msgs[0]
        self.msgs = [system] + [m for m in st["msgs"][1:]]
        self.recs = [dict(r, msg=self.msgs[r["msg"]]) for r in st["recs"] if 0 < r["msg"] < len(self.msgs)]
        self.card = TaskCard()
        for k, v in st["card"].items():
            setattr(self.card, k, v)
        self.pinned, self.pending = set(st["pinned"]), list(st["pending"])
        self.usage.update(st["usage"])
        self.epoch, self.faded, self.reused = st["epoch"], st["faded"], st["reused"]
        self.scouts, self.ratio, self.injected = st["scouts"], st["ratio"], st["injected"]
        self.ws.saved, self.ws.undo_stack = st["saved"], [tuple(u) for u in st["undo"]]
        self.ws.intent = self.card.want

    def reads(self):
        """[(path, read_file calls, read_lines calls)] in this conversation, most read first."""
        from .fade import tool_results
        n = {}
        for _, name, args in tool_results(self.msgs):
            if name in ("read_file", "read_lines") and args.get("path"):
                r = n.setdefault(args["path"], [0, 0])
                r[name == "read_lines"] += 1
        return sorted(((p, a, b) for p, (a, b) in n.items()), key=lambda x: -(x[1] + x[2]))

    def close(self):
        for t in self.bridge_threads:
            t.join(10)
        if self.prefetch is not None and self.depth == 0:
            self.prefetch.close()
        if self.files is not None and self.depth == 0:
            self.files.save()

    def context(self):
        return int(self.ratio * msg_tokens(self.msgs))

    def bridge(self, user):
        """User extract: what the request names, handed on before the first model call and in parallel with it.
        The prefetcher (the cache manager) warms the files it points at, the coprocessor encodes the request into its
        query cache in a background thread, and the returned note (locations only) goes on the user message."""
        from .extract import extract
        x = self.extracted = extract(user, self.prefetch, self.ws)
        picks = []
        if self.files is not None:                               # files the request names get neurons up front, and
            for rel in x.files:                                  # the request is the manager's first input
                try:
                    self._know(self.ws.path(rel), rel)
                except ValueError:
                    continue
            self._index_files()
            picks = self.files.predict(user)
        if self.prefetch is not None and (picks or x.files):
            self.prefetch.prime(picks + [r for r in x.files if r not in picks])
        cp = getattr(self.ws.reader, "coproc", None)
        if cp is not None and getattr(cp, "encoder", None) is not None:
            def encode():
                with self.ws.lock:                         # the encoder and cache are shared with reads
                    self.primed += cp.prime(user)
            t = threading.Thread(target=encode, daemon=True)
            t.start()
            self.bridge_threads.append(t)
        if x.summary():
            self.ui.note(x.summary())
        return x.note() if self.hints else ""

    def turn(self, user):
        """One user request through the tool loop; returns the agent's final text."""
        self.ws.task = user
        self.card.goal = user
        hint = self.bridge(user) if self.depth == 0 else ""
        notes, self.pending = self.pending, []
        self.msgs.append({"role": "user", "content": "\n".join(notes + [user] + ([hint] if hint else []))})
        for _ in range(self.max_steps):
            n = watermark_fade(self.msgs, self.context(), self.ctx, self.ws.outline_of, mark=self.mark, low=self.low,
                               scale=self.ratio, pinned=self.pinned, **self.fade_kw)
            if n:
                self.faded += n
                self.ui.note(f"faded {n} old reads to keep the context under {self.mark:.0%} of {self.ctx:,}")
            sent = msg_tokens(self.msgs)
            try:
                with self.ui.thinking():
                    text, calls, n_in = self.backend.chat(self.msgs, self.tools)
            except RuntimeError as e:
                if not ("expected" in str(e) and "format" in str(e)):
                    raise
                # llama-server could not parse the model's tool call (small models answer as a call to a tool that
                # does not exist): say so and let it try again, as one step
                self.ui.note("the model's tool call did not parse; asking it to retry")
                self.msgs.append({"role": "user", "content": "Your last reply was not a valid tool call. Call one of "
                                  "the listed tools with JSON arguments, or answer in plain text."})
                continue
            if n_in:
                self.ratio, self.last_in = n_in / max(sent, 1), n_in
            for k, v in (getattr(self.backend, "last", None) or {}).items():
                self.usage[k] = self.usage.get(k, 0) + v
            if not calls:
                self.msgs.append({"role": "assistant", "content": text})
                return text
            self.msgs.append({"role": "assistant", "content": text, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                for c in calls]})
            if text.strip() and text.strip() != self.said:      # small models repeat one narration line per step
                self.ui.thought(text.strip())
            self.said = text.strip() or self.said
            self.ws.snapshot = text
            for c in calls:
                out, rec = self.run_tool(c["name"], c["args"])
                self.ui.step(c["name"], c["args"], out)
                m = {"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": out}
                self.msgs.append(m)
                if rec:
                    self.recs.append(dict(rec, msg=m))
            if self.prefetch is not None and self.depth == 0:   # warm the next files while the model thinks
                words = "\n".join((text, self.card.want))
                if self.files is not None:                       # the spiking manager's picks, then the prefetcher's
                    self._index_files()
                    picks = self.files.predict(words + "\n" + " ".join(json.dumps(c["args"]) for c in calls))
                    self.prefetch.prime(picks + [r for r in self.prefetch.predict(words) if r not in picks])
                else:
                    self.prefetch.step(words)
        msg = f"(stopped after {self.max_steps} tool steps; ask me to continue)"
        self.msgs.append({"role": "assistant", "content": msg})
        return msg

    def _file_used(self, path):
        """A file tool ran: the spiking manager scores its last prediction, learns, and spikes the file's synapses."""
        try:
            p = self.ws.path(path)
            rel = self.ws.rel(p)
        except ValueError:
            return
        self.files.used(rel)
        self._know(p, rel)

    def _know(self, p, rel):
        """A neuron for the file, with the symbols it defines (they feed its symbol channel)."""
        try:
            with open(p, encoding="utf8", errors="replace") as f:
                text = f.read()
            self.files.know(rel, [x.name for x in symbols(text, p) if not x.children][:60])
        except (OSError, ValueError, SyntaxError, RecursionError):
            pass

    def _index_files(self):
        """Once the prefetcher's repository index is built, every indexed file gets a neuron, so a name in the
        model's words can fire a file it has not opened yet (the only files warming speeds up)."""
        pf = self.prefetch
        if self.files_indexed or pf is None or not pf.ready.is_set():
            return
        defs = {}
        for name, where in list(pf.index.items()):
            for rel in where:
                defs.setdefault(rel, []).append(name)
        for rel in list(pf.rels):
            if rel not in self.files.idx:
                self.files.know(rel, defs.get(rel, [])[:60])
        self.files_indexed = True

    def status(self):
        pf = self.prefetch
        return dict(context=self.context(), ctx=self.ctx, faded=self.faded, saved=self.ws.saved,
                    reused=self.reused, scouts=self.scouts, card=self.card.version,
                    prefetch_hits=pf.hits if pf else 0, prefetch_misses=pf.misses if pf else 0,
                    warmed=pf.warmed if pf else 0, primed=self.primed,
                    files=self.files.agreement() if self.files is not None and self.files.steps else None, **self.usage)

    def interrupted(self):
        """After Ctrl-C mid-task: every tool call gets a result (servers reject a call without one), and the
        conversation records that the user stopped the task."""
        done = {m.get("tool_call_id") for m in self.msgs if m["role"] == "tool"}
        i = 0
        while i < len(self.msgs):
            for c in self.msgs[i].get("tool_calls") or []:
                if c["id"] not in done:
                    i += 1
                    self.msgs.insert(i, {"role": "tool", "tool_call_id": c["id"], "name": c["function"]["name"],
                                         "content": "interrupted by the user"})
                    done.add(c["id"])
            i += 1
        self.msgs.append({"role": "assistant", "content": "(stopped by the user)"})


def server_ctx(base, api, timeout=3):
    """The context window to fade against when --ctx is not given: llama-server reports its own (n_ctx at /props);
    otherwise 8192 for a local OpenAI-compatible server and 32768 for a hosted API. The first user session ran a
    hosted model with the 8k laptop default and faded 27 times, discarding evidence the model had read."""
    if api == "openai":
        root = re.sub(r"/v1/?$", "", base.rstrip("/"))
        try:
            d = json.loads(urllib.request.urlopen(root + "/props", timeout=timeout).read())
            n = d.get("n_ctx") or (d.get("default_generation_settings") or {}).get("n_ctx")
            if isinstance(n, int) and n > 0:
                return n
        except (OSError, ValueError, urllib.error.URLError):
            pass
        return 8192
    return 32768


FILE_TOOLS = ("read_file", "read_lines", "edit_file", "write_file")


COMMANDS = [("/help", "list the commands"), ("/card", "show the task card"),
            ("/context", "context use, what fades next, cache, savings, prefetch"),
            ("/reads", "reads in the conversation: shown, live / stale / faded"),
            ("/why", "why a read showed those bodies: /why [file]"),
            ("/outline", "numbered outline of a file, for `read N`: /outline file"),
            ("/map", "top-level symbols per source file: /map [dir]"),
            ("/pin", "keep a file's reads from fading: /pin file (no file: list)"),
            ("/unpin", "let a pinned file fade again: /unpin file"),
            ("/compact", "fade old reads now"), ("/undo", "revert the agent's last edit or write"),
            ("/diff", "what /undo would revert"), ("/test", "run the tests and show the agent: /test [command]"),
            ("/lint", "run the linter and show the agent: /lint [command]"),
            ("/learn", "training cases logged for the coprocessor"), ("/clear", "start a fresh conversation"),
            ("/files", "the spiking file manager: next-file guesses, agreement, learned weights"),
            ("/exit", "quit")]
TYPED = [("#3 #7", "pull [connected] menu symbols into the conversation"),
         ("read 2,5 [file]", "pull outline symbols 2 and 5 (numbers from /outline; the last file if omitted)")]
TEST_RX = re.compile(r"\b(pytest|unittest|go test|cargo test|npm (?:run )?test|yarn test|ctest|mvn test|gradle\w* test)\b")
LINT_RX = re.compile(r"\b(ruff|flake8|pylint|mypy|eslint|golangci-lint|go vet|cargo clippy|clippy)\b")


def preset(agent, kind, given=""):
    """The command /test or /lint runs: the one given, else the last such command run this session, else a default
    from the repository's files."""
    if given:
        return given
    rx = TEST_RX if kind == "test" else LINT_RX
    for m in reversed(agent.msgs):
        for c in m.get("tool_calls") or []:
            if c["function"]["name"] == "run":
                cmd = json.loads(c["function"]["arguments"] or "{}").get("command", "")
                if rx.search(cmd):
                    return cmd
    has = lambda f: os.path.exists(os.path.join(agent.ws.root, f))  # noqa: E731
    if kind == "test":
        for f, cmd in (("pyproject.toml", "python -m pytest -q"), ("setup.py", "python -m pytest -q"),
                       ("package.json", "npm test"), ("go.mod", "go test ./..."), ("Cargo.toml", "cargo test")):
            if has(f):
                return cmd
    else:
        for f, cmd in (("pyproject.toml", "python -m ruff check ."), ("setup.py", "python -m ruff check ."),
                       ("package.json", "npx eslint ."), ("go.mod", "go vet ./..."), ("Cargo.toml", "cargo clippy")):
            if has(f):
                return cmd
    return ""


def last_path(agent):
    """The file most recently read in the conversation (the default for /why and `read N`)."""
    for r in reversed(agent.recs):
        if r["name"] == "read_file" and r["args"].get("path"):
            return r["args"]["path"]
    return getattr(agent, "outlined", "")


def typed(agent, ui, line):
    """`#3 #7` (approve menu ids) and `read 2,5 [file]` (outline symbols by number): tool calls made at the prompt,
    no model call. Returns True when the line was one of them. A `cortex ...` shell command typed here is answered
    locally too: the model cannot change the CLI's options, and sending it would cost a call."""
    if re.match(r"(?:cortex(?:\.exe)?|python3? -m cortexnet\.cli)(?:\s|$)", line):
        ui.say("That is a shell command, and this is already a Cortex session. To change options, /exit and run it in "
               "the shell; add --resume to continue this conversation with the new options.")
        return True
    if re.fullmatch(r"#\w+(?:[ ,]+#?\w+)*", line):
        ids = ["#" + x.lstrip("#") for x in re.split(r"[ ,]+", line) if x]
        agent.inject("cortex_approve", {"ids": ids})
        return True
    m = re.fullmatch(r"read\s+(\d+(?:\s*[,-]\s*\d+)*)(?:\s+(\S+))?", line)
    if not m:
        return False
    path = m[2] or last_path(agent)
    if not path:
        ui.say("which file? `read 2,5 path/to/file` (numbers from /outline)")
        return True
    try:
        rows = agent.ws.symbol_list(path)
    except ValueError as e:
        ui.say(str(e))
        return True
    nums = set()
    for part in re.split(r"\s*,\s*", m[1]):
        a, _, b = part.partition("-")
        nums.update(range(int(a), int(b or a) + 1))
    bad = sorted(n for n in nums if not 1 <= n <= len(rows))
    if bad:
        ui.say(f"{path} has symbols 1-{len(rows)} (/outline {path}); no {', '.join(map(str, bad))}")
        return True
    agent.outlined = path
    for n in sorted(nums):
        a, b, _ = rows[n - 1]
        agent.inject("read_lines", {"path": path, "start": a, "end": b})
    return True


def why_lines(ws, path):
    """/why: the include decision of the last read of `path`."""
    try:
        p = ws.path(path)
    except ValueError as e:
        return str(e)
    d = ws.reader.last.get(p)
    if d is None:
        return f"{path} has not been read with read_file in this session"
    if d["mode"] != "outline":
        why = "it is small" if d["mode"] == "whole" else "Rule A judged the whole file cheaper than outline + bodies"
        return f"{path} was shown whole ({d['tokens']:,} tokens): {why}"
    rows = d["rows"]
    inc = [r for r in rows if r["included"]]
    out = [f"{path}: {d['tokens']:,} tokens, shown {d['shown']:,} (outline + {len(inc)} bodies"
           + (f" + {d['menu']} menu entries" if d["menu"] else "") + f"); policy {d['policy']}"]
    cause = dict(rule_ab="Rule B: lowers expected cost", budget=f"P >= {d['tau']:.2f}, within the token budget",
                 coproc_top="the coprocessor's top pick")

    def row(r):
        pc = f", coprocessor {r['pc']:.2f}" if r["pc"] is not None else ""
        return f"  L{r['start']}-{r['end']} {r['name']}: P {r['p']:.2f}{pc}"
    if inc:
        out.append("shown:")
        out += [row(r) + f"  <- {cause.get(r['cause'], r['cause'])}" for r in inc]
    rest = sorted((r for r in rows if not r["included"]), key=lambda r: -max(r["p"], r["pc"] or 0))[:3]
    if rest:
        out.append("closest left out:")
        out += [row(r) for r in rest]
    return "\n".join(out)


def slash(agent, ui, line):
    """A /command, handled here without a model call. Returns "exit", True (handled) or False (not a command)."""
    if not line.startswith("/"):
        return False
    cmd = line.split()[0].lower()
    if cmd in ("/exit", "/quit"):
        return "exit"
    arg = line[len(cmd):].strip()
    if cmd == "/help":
        ui.say("\n".join(f"  {n:<16} {d}" for n, d in COMMANDS + TYPED))
    elif cmd == "/card":
        ui.say(agent.card.render() if not agent.card.empty() else "no task card yet (the agent keeps one with `note`)")
    elif cmd == "/files":
        fm = agent.files
        if fm is None:
            ui.say("spiking file manager off (--no-snn)")
        else:
            a = fm.agreement()
            guess = ", ".join((fm.pending or {}).get("pred", [])) or "(nothing fired)"
            ui.say(f"  spiking file manager for {fm.model}: {len(fm.files)} file neurons, {a['steps']} predictions "
                   f"scored\n"
                   f"  agreement       hit@1 {a['hit@1']:.0%}, hit@3 {a['hit@3']:.0%}; files not opened yet "
                   f"{a['fresh_hits']:.0%} of {a['fresh']}\n"
                   f"  next files      {guess}\n"
                   f"  learned weights {fm.habits()}; driving: {'learned' if a['plastic'] else 'innate'}\n"
                   f"  network         {a['spikes']:.1f} spikes per prediction, {a['silent']:.0%} of neurons silent")
    elif cmd == "/context":
        s = agent.status()
        hits, miss = s["prefetch_hits"], s["prefetch_misses"]
        rows = [("context", f"{s['context']:,} of {s['ctx']:,} tokens ({s['context'] / max(s['ctx'], 1):.0%}); "
                            f"fades at {agent.mark:.0%} down to {agent.low:.0%}"),
                ("faded", f"{s['faded']} messages"),
                ("reads", f"CortexNET saved {s['saved']:,} tokens vs whole-file reads; "
                          f"{s['reused']} repeat calls answered from context"),
                ("model", f"sent {s['sent']:,}, cached {s['cached']:,}"
                          + (f" ({s['cached'] / s['sent']:.0%})" if s["sent"] else "") + f", output {s['out']:,}"),
                ("prefetch", f"{hits} of {hits + miss} reads predicted, {s['warmed']} files warmed"
                             if agent.prefetch else "off"),
                ("extract", (agent.extracted.summary() or "nothing named in the last request")
                            + (f"; {s['primed']} request texts encoded ahead" if s["primed"] else "")
                            if agent.extracted else "no request yet"),
                ("card", f"v{s['card']}, scouts used {s['scouts']}")]
        if agent.pinned:
            rows.append(("pinned", ", ".join(sorted(agent.pinned))))
        ui.say("\n".join(f"  {k:<9} {v}" for k, v in rows))
        nxt = agent.fade_preview()
        if nxt:
            gain = sum(x["saved"] for x in nxt)
            ui.table(f"next fade at {agent.mark:.0%} of {agent.ctx:,}: {len(nxt)} messages, about {gain:,} tokens",
                     ("file", "becomes", "saves"), [(x["path"], x["what"], f"{x['saved']:,}") for x in nxt[:10]])
        else:
            ui.say("  next fade: nothing old enough to shrink yet")
    elif cmd == "/reads":
        rows = agent.read_rows()
        if rows:
            ui.table("reads in the conversation (oldest first)", ("read", "shown", "state", "tokens"),
                     [(p, w, s, f"{t:,}") for p, w, s, t in rows])
        else:
            ui.say("no reads in the conversation")
    elif cmd == "/why":
        path = arg or last_path(agent)
        ui.say(why_lines(agent.ws, path) if path else "no file read yet: /why path/to/file")
    elif cmd == "/outline":
        path = arg or last_path(agent)
        try:
            rows = agent.ws.symbol_list(path) if path else None
        except ValueError as e:
            rows = None
            ui.say(str(e))
        if rows:
            agent.outlined = path
            ui.say(f"{path} (`read 2,5` pulls symbols by number):\n"
                   + "\n".join(f"  {i:>3}. {label}" for i, (_, _, label) in enumerate(rows, 1)))
        elif rows is not None or not path:
            ui.say("no outline: /outline path/to/file")
    elif cmd == "/map":
        try:
            rows = agent.ws.repo_map(arg or ".")
        except ValueError as e:
            rows = None
            ui.say(str(e))
        if rows:
            ui.say("\n".join(f"  {p} ({n}): {', '.join(names)}{', ...' if n > len(names) else ''}"
                              for p, n, names in rows) + ("\n  [first 40 files]" if len(rows) >= 40 else ""))
        elif rows is not None:
            ui.say("no source files there")
    elif cmd == "/pin":
        if arg:
            try:
                agent.ws.path(arg)
            except ValueError as e:
                ui.say(str(e))
                return True
            agent.pinned.add(arg)
            ui.say(f"pinned {arg}: fade leaves its reads as they are")
        else:
            ui.say("pinned: " + ", ".join(sorted(agent.pinned)) if agent.pinned else "nothing pinned (/pin file)")
    elif cmd == "/unpin":
        if arg in agent.pinned:
            agent.pinned.discard(arg)
            ui.say(f"unpinned {arg}")
        else:
            ui.say(f"{arg or '(no file)'} is not pinned")
    elif cmd == "/undo":
        msg = agent.undo()
        ui.say(msg + " (the agent is told with your next message)" if msg else "nothing to undo")
    elif cmd == "/diff":
        d = agent.ws.undo_diff()
        ui.say(d if d else "nothing to undo")
    elif cmd in ("/test", "/lint"):
        kind = cmd[1:]
        command = preset(agent, kind, arg)
        if not command:
            ui.say(f"no {kind} command known yet: {cmd} <command>")
        else:
            agent.inject("run", {"command": command})
    elif cmd == "/compact":
        before = agent.context()
        n = agent.compact()
        ui.say(f"faded {n} messages: context {before:,} -> {agent.context():,} tokens")
    elif cmd == "/learn":
        from .cases import MIN_CASES, build, summary
        _, _, st = build(agent.ws.cases_dir)
        ui.say(summary(st) + f". `cortex learn` writes the dataset (training wants {MIN_CASES}+ cases).")
    elif cmd == "/clear":
        agent.clear()
        ui.say("fresh conversation (usage totals and CortexNET's read state are kept)")
    else:
        ui.say(f"unknown command {cmd}; /help lists them")
    return True


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["learn"]:
        from .cases import main as learn
        return learn(argv[1:])
    ap = argparse.ArgumentParser(prog="cortex", description=__doc__.split("\n\n")[0])
    ap.add_argument("task", nargs="*", help="a task to run once (otherwise an interactive session)")
    env = os.environ.get                                  # per-machine defaults: CORTEX_API, _BASE, _MODEL, _KEY_ENV
    ap.add_argument("--api", choices=("openai", "anthropic"), default=env("CORTEX_API") or "openai",
                    help="default $CORTEX_API, else openai")
    ap.add_argument("--base", default=env("CORTEX_BASE"), help="server base URL (default $CORTEX_BASE, else "
                    "http://127.0.0.1:8080/v1 for openai and https://api.anthropic.com for anthropic)")
    ap.add_argument("--model", default=env("CORTEX_MODEL") or "local", help="default $CORTEX_MODEL, else local")
    ap.add_argument("--key-env", default=env("CORTEX_KEY_ENV"),
                    help="name of the environment variable holding the API key (default $CORTEX_KEY_ENV)")
    ap.add_argument("--ctx", type=int, default=None, help="the model's context window, for the fade watermark "
                    "(default: asked from llama-server, else 8192 locally and 32768 for hosted APIs)")
    ap.add_argument("--coproc", default=os.environ.get("CORTEXNET_COPROC", ""), help="Cortex-SubNN head (.npz)")
    ap.add_argument("--no-graph", action="store_true", help="with --coproc: no symbol graph / connected menu")
    ap.add_argument("--no-snn", action="store_true", default=os.environ.get("CORTEXNET_SNN") == "0",
                    help="no spiking file manager (cortexnet/filesnn.py): the prefetcher alone picks the files to warm")
    ap.add_argument("--yes", action="store_true", help="apply edits and run commands without asking")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--repo", default=".", help="repository root (default: the current directory)")
    ap.add_argument("--plain", action="store_true", help="plain text output (no colors, no prompt toolbar)")
    ap.add_argument("--plan", action="store_true", help="read-only: the agent cannot edit files or run commands")
    ap.add_argument("--no-hints", action="store_true", help="do not add the user extract's note (where the files and "
                    "symbols your request names are defined) to your messages; prefetch still runs")
    ap.add_argument("--resume", nargs="?", const="last", default=None, metavar="SESSION",
                    help="continue the last session in this repository (or a saved session file)")
    ap.add_argument("--fork", action="store_true", help="with --resume: continue in a new session file and leave "
                    "the resumed one as it was")
    a = ap.parse_args(argv)
    key = os.environ.get(a.key_env) if a.key_env else None
    base = a.base or ("http://127.0.0.1:8080/v1" if a.api == "openai" else "https://api.anthropic.com")
    backend = OpenAIChat(base, a.model, key) if a.api == "openai" else AnthropicMessages(base, a.model, key)
    a.ctx = a.ctx or server_ctx(base, a.api)
    root = os.path.realpath(a.repo)
    kw = dict(split_tokens=600)
    if a.coproc:
        from .coproc import MiniLM
        from .coproc import Coprocessor
        cp = Coprocessor(a.coproc)                                  # refuses a bad head before MiniLM loads
        cp.encoder = MiniLM()
        kw.update(coproc=cp, ctx_base=2000)
        if not a.no_graph:
            from .graph import CoNeedGraph
            kw.update(graph=CoNeedGraph(), menu=True)
    ui = make_ui(a.plain, history=os.path.join(root, ".cortexnet", "cli_history"), commands=COMMANDS)
    ws = Workspace(root, Reader(os.path.join(root, ".cortexnet", "agent"), **kw),
                   confirm=(lambda q, d: True) if a.yes else ui.confirm)
    tools = [t for t in TOOLS if t["function"]["name"] not in MUTATING] if a.plan else TOOLS
    agent = Agent(backend, ws, ctx=a.ctx, max_steps=a.max_steps, ui=ui, tools=tools,
                  system=SYSTEM + PLAN_LINE if a.plan else SYSTEM, hints=not a.no_hints)
    if not a.no_snn:
        from .filesnn import SpikingFileManager
        agent.files = SpikingFileManager(model=f"{a.api}:{a.model}", state_dir=os.path.join(root, ".cortexnet", "agent"))
    sessions = Sessions(root)
    resumed = None
    if a.resume:
        try:
            resumed = sessions.load(a.resume, agent)
        except (OSError, ValueError, KeyError) as e:
            ui.error(f"cannot resume: {e}")
            agent.close()
            return 1
    sessions.start(resumed, fork=a.fork)
    mode = ", ".join(x for x in ("plan (read-only)" if a.plan else "",
                                 "coprocessor + graph + menu" if a.coproc else "",
                                 "auto-approve" if a.yes else "",
                                 f"resumed {os.path.basename(resumed)}" + (" as a fork" if a.fork else "")
                                 if resumed else "") if x)
    ui.header(dict(repo=root, backend=f"{a.api} {a.model} @ {base}", ctx=a.ctx, mode=mode))
    if resumed:
        ui.status(agent.status())
    tasks = [" ".join(a.task)] if a.task else None
    while True:
        if tasks is not None:
            if not tasks:
                break
            user = tasks.pop()
        else:
            user = ui.read()
            if user is None or user in ("exit", "quit"):
                break
            if not user:
                continue
            try:
                done = slash(agent, ui, user) or typed(agent, ui, user)
            except KeyboardInterrupt:              # e.g. a long /test: no tool call is left without a result
                done = True
                agent.interrupted()
                ui.note("stopped")
            if done == "exit":
                break
            if done:
                sessions.save(agent)
                continue
        try:
            ui.answer(agent.turn(user))
        except KeyboardInterrupt:
            agent.interrupted()
            ui.note("stopped; the conversation so far is kept")
        except RuntimeError as e:
            ui.error(str(e))
            if tasks is not None:
                sessions.save(agent)
                agent.close()
                return 1
        sessions.save(agent)
        ui.status(agent.status())
    agent.close()
    return 0


class Sessions:
    """Saved conversations under .cortexnet/: session.json is the latest; older ones and forks go to sessions/.
    Saved after every turn and every state-changing command, with an atomic replace."""

    def __init__(self, root, keep=10):
        self.dir = os.path.join(root, ".cortexnet")
        self.archive, self.keep = os.path.join(self.dir, "sessions"), keep
        self.latest = os.path.join(self.dir, "session.json")
        self.path = self.latest

    def load(self, which, agent):
        p = self.latest if which == "last" else which
        with open(p, encoding="utf8") as f:
            agent.restore(json.load(f))
        return p

    def start(self, resumed, fork=False):
        """Where this run saves: a fork gets its own file; a fresh run archives the previous latest session."""
        os.makedirs(self.archive, exist_ok=True)
        if resumed and fork:
            self.path = os.path.join(self.archive, time.strftime("fork-%Y%m%d-%H%M%S.json"))
        elif resumed:
            self.path = resumed
        elif os.path.exists(self.latest):
            os.replace(self.latest, os.path.join(self.archive, time.strftime(
                "%Y%m%d-%H%M%S.json", time.localtime(os.path.getmtime(self.latest)))))
            old = sorted(f for f in os.listdir(self.archive) if not f.startswith("fork-"))
            for f in old[:-self.keep]:
                os.remove(os.path.join(self.archive, f))

    def save(self, agent):
        if len(agent.msgs) < 2:
            return
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf8") as f:
                json.dump(agent.state(), f)
            os.replace(tmp, self.path)
        except (OSError, TypeError, ValueError):
            pass


def _takes_edit(fn):
    """Whether a confirm callback accepts a third argument (the text the user may revise)."""
    try:
        ps = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    return any(p.kind == p.VAR_POSITIONAL for p in ps) or \
        sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in ps) >= 3


if __name__ == "__main__":
    sys.exit(main())
