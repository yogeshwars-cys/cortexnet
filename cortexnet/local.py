"""CortexNET locally: from nothing to asking questions about a repository, with a model on your own GPU, in one command.

  cortexnet try   [--repo DIR]           no model, no key: what CortexNET's first read of each source file shows,
                                         against reading the whole file
  cortexnet local [--repo DIR] [...]     fetch llama.cpp's server and Llama 3.1 8B Instruct (Q4_K_M, ~4.9 GB) once,
                                         start it with an 8k context, then ask questions about DIR in one session

The session is the laptop A/B's setup (bench/live_ab.py --session): every read goes through CortexNET's Reader
(outline first, the bodies the include model picks, read_lines for the rest) and the history fades on a watermark,
85% of the context down to 50% in one batch, so a long session stays inside the window. After each answer one line
reports the context in use, the reads faded, and the file tokens shown against whole-file reads.

Downloads go to $CORTEXNET_HOME (default: the user cache dir + cortexnet). The llama.cpp build is pinned (BUILD):
Vulkan on Windows and Linux x64 (NVIDIA, AMD and Intel GPUs, no CUDA runtime needed), Metal on Apple silicon, --cpu
for machines without a usable GPU. --gguf uses a model file you already have; --server uses a server you already run.
"""
import argparse
import json
import os
import platform
import re

import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

from .fade import watermark_fade
from .outline import outline, symbols, tokens
from .reader import SRC_EXT, Reader

BUILD = "b12015"
MODEL_HF = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
SKIP = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".cortexnet", "dist", "build", ".tox",
        ".mypy_cache", ".pytest_cache", ".idea", ".vscode"}
SYSTEM = ("You answer questions about the code in this repository. Always look the answer up with the tools; never "
          "answer from memory. Paths are relative to the repository root. read_file returns small files whole and large "
          "files as an outline (each function and class with its line range) plus the parts most likely relevant; use "
          "read_lines for any other lines. Answer in a few sentences and cite file:line. Files:\n{files}\n"
          'Use a path exactly as listed, e.g. read_file with path "{example}". If a tool returns an error, fix the call '
          "and try again; do not answer from an error message.")


def home():
    root = os.environ.get("CORTEXNET_HOME")
    if not root:
        base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else (
            os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"))
        root = os.path.join(base or os.path.expanduser("~"), "cortexnet")
    os.makedirs(root, exist_ok=True)
    return root


def source_files(root, exts=SRC_EXT):
    """Source files under root. In a git repository: what git lists (tracked, plus untracked files no ignore rule
    covers, .git/info/exclude included), so ignored and private files never reach the model or the report."""
    try:
        out = subprocess.run(["git", "-C", root, "ls-files", "-co", "--exclude-standard", "-z", "--", "."],
                             capture_output=True, check=True, timeout=30).stdout.decode("utf8", "replace")
        return sorted(os.path.join(root, f) for f in out.split("\0")
                      if f and os.path.splitext(f)[1].lower() in exts and os.path.isfile(os.path.join(root, f)))
    except (OSError, subprocess.SubprocessError):
        pass
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in SKIP and not d.startswith("."))
        out += [os.path.join(dp, f) for f in sorted(fns) if os.path.splitext(f)[1].lower() in exts]
    return out


def rel(p, root):
    return os.path.relpath(p, root).replace("\\", "/")


# ---- cortexnet try --------------------------------------------------------------------------------------------------

def cmd_try(a):
    root = os.path.abspath(a.repo)
    files = source_files(root)
    if not files:
        raise SystemExit(f"no source files under {root}")
    rows = []
    with tempfile.TemporaryDirectory() as state:              # a throwaway include model: the shipped priors only
        r = Reader(state, split_tokens=600)
        for p in files:
            try:
                with open(p, encoding="utf8", errors="replace") as fh:
                    whole = tokens(fh.read())
                shown = tokens(r.read(p))
            except (OSError, ValueError):
                continue
            rows.append((rel(p, root), whole, shown))
    big = [x for x in rows if x[2] < x[1]]
    w, s = sum(x[1] for x in rows), sum(x[2] for x in rows)
    print(f"{root}\n{len(rows)} source files, {w:,} tokens read whole; CortexNET's first reads show {s:,} ({s / max(w, 1):.0%})."
          f"\n{len(rows) - len(big)} small files come back whole; {len(big)} large ones as outline + likely bodies.\n")
    print(f"{'largest files':<56} {'whole':>8} {'cortexnet':>10} {'share':>6}")
    for path, whole, shown in sorted(big, key=lambda x: -x[1])[:a.top]:
        print(f"{path[-56:]:<56} {whole:>8,} {shown:>10,} {shown / whole:>6.0%}")
    print("\nAnything not shown is one read_lines away, and each such request is a miss that trains the include model."
          "\nNext: `cortexnet local` asks questions about this repository with a model on your own GPU.")


# ---- llama.cpp server -----------------------------------------------------------------------------------------------

def asset(cpu):
    system, machine = platform.system(), platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if system == "Windows":
        return f"llama-{BUILD}-bin-win-{'cpu' if cpu else 'vulkan'}-{'arm64' if arm else 'x64'}.zip"
    if system == "Darwin":
        return f"llama-{BUILD}-bin-macos-{'arm64' if arm else 'x64'}.tar.gz"
    if system == "Linux":
        return f"llama-{BUILD}-bin-ubuntu-{'' if cpu else 'vulkan-'}{'arm64' if arm else 'x64'}.tar.gz"
    raise SystemExit(f"no llama.cpp build for {system}; start a server yourself and pass --server URL")


def fetch(url, dest):
    part = dest + ".part"
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
        total, got, shown = int(resp.headers.get("Content-Length") or 0), 0, -1
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            got += len(chunk)
            pct = got * 100 // total if total else -1
            if pct // 10 != shown // 10:
                print(f"  {got / 1048576:,.0f} MB" + (f" ({pct}%)" if total else ""), flush=True)
                shown = pct
    os.replace(part, dest)


def server_binary(cpu):
    name = asset(cpu)
    d = os.path.join(home(), "llama.cpp", name.rsplit(".", 2 if name.endswith(".tar.gz") else 1)[0])
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    found = [os.path.join(dp, exe) for dp, _, fs in os.walk(d) if exe in fs] if os.path.isdir(d) else []
    if not found:
        os.makedirs(d, exist_ok=True)
        archive = os.path.join(d, name)
        print(f"downloading llama.cpp {BUILD} ({name})", flush=True)
        fetch(f"https://github.com/ggml-org/llama.cpp/releases/download/{BUILD}/{name}", archive)
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(d)
        else:
            with tarfile.open(archive) as t:
                t.extractall(d, filter="data")
        os.remove(archive)
        found = [os.path.join(dp, exe) for dp, _, fs in os.walk(d) if exe in fs]
        if not found:
            raise SystemExit(f"{exe} not found in {name}")
        if os.name != "nt":
            os.chmod(found[0], 0o755)
    return found[0]


def get(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def healthy(base):
    try:
        return get(base + "/health")[0] == 200
    except (OSError, ValueError):
        return False


def start_server(a):
    """Start llama-server and wait until the model is loaded. Returns (process, log path)."""
    exe = server_binary(a.cpu)
    log = os.path.join(home(), "llama-server.log")
    cmd = [exe, "--host", "127.0.0.1", "--port", str(a.port), "-c", str(a.ctx), "-np", "1", "--jinja",
           "-ngl", "0" if a.cpu else "99"] + (["-m", a.gguf] if a.gguf else ["-hf", a.model])
    env = dict(os.environ, LLAMA_CACHE=os.path.join(home(), "models"))
    env.pop("LLAMA_API_KEY", None)        # llama-server takes it as its own key; it usually holds a Meta API key
    print(f"starting llama-server on :{a.port} ({'model ' + a.gguf if a.gguf else a.model}); log: {log}")
    if not a.gguf:
        print("the first run downloads the model (~4.9 GB) into " + env["LLAMA_CACHE"])
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    base, t0 = f"http://127.0.0.1:{a.port}", time.time()
    note = t0 + 30
    try:
        while not healthy(base):
            if proc.poll() is not None:
                with open(log, encoding="utf8", errors="replace") as fh:
                    tail = fh.read().splitlines()[-15:]
                raise SystemExit("llama-server exited:\n  " + "\n  ".join(tail) +
                                 "\n(no usable GPU? try --cpu; a model you already have: --gguf PATH)")
            time.sleep(2)
            if time.time() > note:
                print(f"  still loading ({time.time() - t0:.0f} s)", flush=True)
                note += 30
    except KeyboardInterrupt:
        proc.terminate()
        raise SystemExit("stopped")
    print(f"ready in {time.time() - t0:.0f} s\n")
    return proc


# ---- question-and-answer session ------------------------------------------------------------------------------------

def post(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None


def msg_tokens(msgs):
    return sum(tokens(str(m.get("content") or "")) + sum(tokens(c["function"]["arguments"]) + 8
                                                         for c in m.get("tool_calls") or []) for m in msgs)


def tool(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": req}}}


S = {"type": "string"}
TOOLS = [tool("read_file", "Read a file. Small files come back whole with line numbers; large files come back as an "
              "outline (each function/class with its line range) plus the parts most likely relevant to `query`. "
              "Use read_lines to see any other lines.", {"path": S, "query": S}, ["path"]),
         tool("read_lines", "Return lines start..end of a file with line numbers.",
              {"path": S, "start": {"type": "integer"}, "end": {"type": "integer"}}, ["path", "start", "end"]),
         tool("grep", "Search files for a regular expression; returns path:line: text for each match (max 40).",
              {"pattern": S, "path": {"type": "string", "description": "file or directory, relative"}}, ["pattern"])]


class Session:
    """One conversation over a repository: CortexNET reads, a watermark fade, and running totals for the report."""

    def __init__(self, root, base, ctx, state):
        self.root, self.base, self.ctx = root, base, ctx
        self.reader = Reader(state, split_tokens=600)
        files = [rel(p, root) for p in source_files(root)]
        listing = "\n".join(files[:200]) + (f"\n(+{len(files) - 200} more; use grep)" if len(files) > 200 else "")
        self.msgs = [{"role": "system", "content": SYSTEM.format(files=listing, example=files[0] if files else "")}]
        self.ratio, self.faded, self.shown, self.whole = 1.0, 0, 0, 0

    def path(self, p):
        full = os.path.normpath(os.path.join(self.root, (p or ".").lstrip("/\\")))
        if not full.startswith(os.path.normpath(self.root)):
            raise ValueError("path outside the repository")
        if not os.path.exists(full):          # "reader.py" for cortexnet/reader.py: the one source file it ends with
            want = "/" + (p or "").replace("\\", "/").lstrip("/")
            hits = [f for f in source_files(self.root) if ("/" + rel(f, self.root)).endswith(want)]
            if len(hits) == 1:
                return hits[0]
        return full

    def outline_of(self, p):
        try:
            full = self.path(p)
            with open(full, encoding="utf8", errors="replace") as fh:
                text = fh.read()
        except (OSError, ValueError):
            return None
        return outline(text, full, symbols(text, full, split_tokens=600))

    def call(self, name, a, question):
        try:
            if name == "read_file":
                full = self.path(a["path"])
                out = self.reader.read(full, (question + "\n" + (a.get("query") or "")).strip())
                with open(full, encoding="utf8", errors="replace") as fh:
                    self.whole += tokens(fh.read())
                self.shown += tokens(out)
                return out
            if name == "read_lines":
                out = self.reader.range(self.path(a["path"]), int(a["start"]), int(a["end"]))
                self.shown += tokens(out)
                return out
            if name == "grep":
                rx, base = re.compile(a["pattern"]), self.path(a.get("path", "."))
                files = [base] if os.path.isfile(base) else source_files(base, SRC_EXT | {".md"})
                hits = []
                for f in files:
                    with open(f, encoding="utf8", errors="replace") as fh:
                        hits += [f"{rel(f, self.root)}:{i}: {l.rstrip()[:200]}" for i, l in enumerate(fh, 1)
                                 if rx.search(l)]
                    if len(hits) >= 40:
                        break
                return "\n".join(hits[:40]) or "no matches"
            return f"unknown tool {name}"
        except (OSError, KeyError, ValueError, re.error, TypeError) as e:
            return f"error: {type(e).__name__}: {e}"

    def ask(self, question, max_calls=8):
        """Answer one question in the running conversation. Returns (answer, tool calls, seconds)."""
        self.msgs.append({"role": "user", "content": question})
        t0, ncalls, answer = time.time(), 0, ""
        for turn in range(max_calls + 1):
            self.faded += watermark_fade(self.msgs, self.ratio * msg_tokens(self.msgs), self.ctx, self.outline_of,
                                         scale=self.ratio)
            sent = msg_tokens(self.msgs)
            body = {"messages": [{k: v for k, v in m.items() if not k.startswith("_")} for m in self.msgs],
                    "temperature": 0}
            if ncalls < max_calls:
                body["tools"] = TOOLS
            r = post(self.base + "/v1/chat/completions", body)
            n_in = (r.get("usage") or {}).get("prompt_tokens", 0)
            if n_in:
                self.ratio = n_in / max(sent, 1)          # server tokens per estimated token (tokenizer + template)
            msg = r["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                answer = msg.get("content") or ""
                break
            self.msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": [
                {"id": c.get("id") or f"call_{turn}_{j}", "type": "function", "function": c["function"]}
                for j, c in enumerate(calls)]})
            for j, c in enumerate(calls):
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except ValueError:                # small models sometimes emit broken JSON; the tool says what is missing
                    args = {}
                print(f"  {c['function']['name']}({', '.join(f'{k}={v!r}' for k, v in args.items())})", flush=True)
                self.msgs.append({"role": "tool", "tool_call_id": c.get("id") or f"call_{turn}_{j}",
                                  "name": c["function"]["name"], "content": self.call(c["function"]["name"], args,
                                                                                      question)})
                ncalls += 1
        self.msgs.append({"role": "assistant", "content": answer})
        return answer, ncalls, time.time() - t0

    def report(self, ncalls, secs):
        used = int(self.ratio * msg_tokens(self.msgs))
        share = f"{self.shown / self.whole:.0%}" if self.whole else "n/a"
        return (f"[context {used:,} of {self.ctx:,} | reads faded {self.faded} | {ncalls} tool calls | file tokens shown "
                f"{self.shown:,} vs {self.whole:,} whole ({share}) | {secs:.1f} s]")


def cmd_local(a):
    root = os.path.abspath(a.repo)
    base = re.sub(r"/v1/?$", "", (a.server or f"http://127.0.0.1:{a.port}").rstrip("/"))
    proc = None
    if not a.server:
        if healthy(base):
            print(f"using the server already on :{a.port}")
        else:
            proc = start_server(a)
    try:
        ctx = a.ctx
        try:
            props = json.loads(get(base + "/props")[1])
            ctx = props.get("n_ctx") or (props.get("default_generation_settings") or {}).get("n_ctx") or ctx
        except (OSError, ValueError):
            pass
        sess = Session(root, base, int(ctx), os.path.join(root, ".cortexnet"))
        print(f"CortexNET, local: questions about {root}, {ctx:,}-token context. Empty line or 'exit' quits.\n")
        while True:
            try:
                q = input("ask> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q.lower() in ("", "exit", "quit"):
                break
            try:
                answer, n, secs = sess.ask(q)
            except RuntimeError as e:
                print(f"server error: {e}\n")
                continue
            except KeyboardInterrupt:
                print("\n(stopped)\n")
                continue
            print(f"\n{answer.strip()}\n{sess.report(n, secs)}\n")
    finally:
        if proc:
            proc.terminate()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cortexnet", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("try", help="no model, no key: CortexNET's first reads vs whole files on a repository")
    t.add_argument("--repo", default=".", help="repository to read (default: the current directory)")
    t.add_argument("--top", type=int, default=12, help="largest files to list")
    loc = sub.add_parser("local", help="a model on your own GPU: download once, then ask questions about a repository")
    loc.add_argument("--repo", default=".", help="repository to ask about (default: the current directory)")
    loc.add_argument("--ctx", type=int, default=8192, help="context window (default 8192, the 6 GB laptop setup)")
    loc.add_argument("--port", type=int, default=8080)
    loc.add_argument("--model", default=MODEL_HF, help=f"Hugging Face repo:quant for llama-server -hf (default {MODEL_HF})")
    loc.add_argument("--gguf", default="", help="use a GGUF file you already have instead of downloading")
    loc.add_argument("--cpu", action="store_true", help="CPU build and no GPU offload")
    loc.add_argument("--server", default="", help="an OpenAI-compatible server you already run (skips the download)")
    a = ap.parse_args(argv)
    (cmd_try if a.cmd == "try" else cmd_local)(a)


if __name__ == "__main__":
    sys.exit(main())
