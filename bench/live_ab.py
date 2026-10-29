"""Live A/B: the same agent answers the same code questions with whole-file reads vs CortexNET tools.

Arms
  baseline  read_file(path) returns the whole file (numbered), grep(pattern, path)
  cortex    read_file(path, query) = outline-first CortexNET read, read_lines(path, start, end), grep(pattern, path)
Metrics per arm: accuracy (answer regex), input tokens the model processed summed over turns (what an API bills
without caching), tool-output tokens, tool calls. Files are copied into a fixture directory first, so answers stay
fixed while the repo changes.

Backends (same tool loop):
  --hf <model id>            local transformers model with a chat template that supports tools (default Qwen/Qwen3-1.7B)
  --openai <base url>        any OpenAI-compatible server, e.g. llama-server running Muse Glimmer
  --meta                     Meta Model API (Responses endpoint, default muse-spark-1.3); key from MODEL_API_KEY or
                             LLAMA_API_KEY. Reasoning summaries are logged per turn to results/traces/ (gitignored).
Usage: python -I bench/live_ab.py [--hf Qwen/Qwen3-1.7B | --openai http://127.0.0.1:8080/v1 | --meta] [--tasks N]
                                  [--budget USD]

Paid backends stop before a call would push total spend past --budget. Spend is kept in a ledger across runs
(results/traces/spend.jsonl), so the budget covers every run, not just this one. Only public code goes into the
fixture (this repo's package and Python standard library files), because hosted APIs may keep what they are sent.
"""
import argparse
import json
import os
import re
import shutil
import sys
import sysconfig
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortexnet import Reader, outline, symbols, tokens  # noqa: E402
from cortexnet.fade import watermark_fade  # noqa: E402
from cortexnet.reader import numbered  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STDLIB = sysconfig.get_paths()["stdlib"]
# Public code only: this repo's package and big Python standard library files (fixed for a given Python version).
SOURCES = [(os.path.join(REPO, "cortexnet"), "cortexnet")] + [
    (os.path.join(STDLIB, f), "stdlib/" + f) for f in ("argparse.py", "textwrap.py", "shlex.py", "configparser.py",
                                                     "difflib.py", "logging/handlers.py")]
TRACES = os.path.join(HERE, "results", "traces")
TASKS = [
    ("What is the default value of full_below in the Reader class?", r"\b800\b"),
    ("What value does the DECAY constant in reader.py have?", r"\b0?\.7\b"),
    ("In learn.py, what prior weight does the 'changed' feature get?", r"\b4(\.0)?\b"),
    ("What is the default learning rate (lr) of IncludeModel?", r"\b0?\.1\b"),
    ("What is the default section size used by sections() in outline.py?", r"\b40\b"),
    ("Which protocolVersion does the MCP server return by default when the client sends none?", r"2025-06-18"),
    ("Which Reader method records whether a request was a miss?", r"\brange\b"),
    ("In learn.py, which function computes the symbol-graph edge weight between two symbols?", r"\blink\b"),
    ("In stdlib/argparse.py, what is the default max_help_position of HelpFormatter?", r"\b24\b"),
    ("In stdlib/argparse.py, which ArgumentParser method expands arguments that start with fromfile_prefix_chars?",
     r"_read_args_from_files"),
    ("In stdlib/argparse.py, which ArgumentParser method raises the 'invalid choice' error?", r"_check_value"),
    ("In stdlib/textwrap.py, what is the default placeholder of TextWrapper?", r"\[\.\.\.\]"),
    ("In stdlib/shlex.py, what does the shlex class set as its default commenters?", r"#"),
    ("In stdlib/configparser.py, what is MAX_INTERPOLATION_DEPTH?", r"\b10\b"),
    ("In stdlib/configparser.py, which interpolation class implements the ${section:option} syntax?",
     r"ExtendedInterpolation"),
    ("In stdlib/difflib.py, above what sequence length does SequenceMatcher's autojunk heuristic apply?", r"\b200\b"),
    ("In stdlib/logging/handlers.py, what is DEFAULT_SOAP_LOGGING_PORT?", r"\b9023\b"),
    ("In stdlib/logging/handlers.py, what is the default `when` of TimedRotatingFileHandler?", r"""['"`]?h['"`]?\b"""),
    ("In stdlib/logging/handlers.py, what is the default flushLevel of MemoryHandler?", r"\bERROR\b"),
    ("In stdlib/logging/handlers.py, how many seconds does _MIDNIGHT hold?", r"86,?400|24\s*\*\s*60\s*\*\s*60"),
]
PRICES = {"muse-spark-1.3": (1.25, 0.15, 4.25), "muse-spark-1.2": (1.25, 0.15, 4.25),
          "muse-spark-1.1": (1.25, 0.15, 4.25)}  # USD per 1M tokens: input, cached input, output (incl. reasoning)


class BudgetExceeded(Exception):
    pass


class Ledger:
    """Spend across all paid runs, so --budget is a cap on the total, not per run."""

    def __init__(self, cap):
        os.makedirs(TRACES, exist_ok=True)
        self.path, self.cap = os.path.join(TRACES, "spend.jsonl"), cap
        self.spent = 0.0
        if os.path.exists(self.path):
            with open(self.path, encoding="utf8") as f:
                self.spent = sum(json.loads(l)["usd"] for l in f if l.strip())

    def check(self, estimate):
        if self.spent + estimate > self.cap:
            raise BudgetExceeded(f"spent ${self.spent:.4f}, next call up to ${estimate:.4f}, cap ${self.cap:.2f}")

    def add(self, usd, **info):
        self.spent += usd
        with open(self.path, "a", encoding="utf8") as f:
            f.write(json.dumps(dict(usd=usd, t=time.time(), **info)) + "\n")
SYSTEM = ("You answer questions about the code in this workspace. Always look the answer up with the tools before "
          "answering; never answer from memory. Paths are relative to the workspace root. Files:\n{files}\n"
          "When you know the answer, reply with one short sentence that starts with 'ANSWER:'.")
# --hint (for small local models): one concrete call shape; graph.py is in no task, so it gives nothing away
HINT = ('\nUse a path exactly as listed above, e.g. read_file with path "cortexnet/graph.py". If a tool returns an '
        "error, fix the call and try again; do not answer from an error message.")


def fixture(ref=""):
    """The workspace the agent reads. ref: take cortexnet/ from that git commit instead of the working tree, so the
    questions' corpus (and the file list in the system prompt) stays fixed while the code under test changes."""
    d = tempfile.mkdtemp(dir=os.environ.get("TMP"))
    for src, name in SOURCES:
        dst = os.path.join(d, name)
        if ref and name == "cortexnet":
            import io
            import subprocess
            import tarfile
            tar = subprocess.run(["git", "-C", REPO, "archive", "--format=tar", ref, "cortexnet"], check=True,
                                 capture_output=True).stdout
            with tarfile.open(fileobj=io.BytesIO(tar)) as t:
                t.extractall(d, filter="data")
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.json", "*.npz"))
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
    return d


def tool_specs(arm):
    def fn(name, desc, props, req):
        return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
            "type": "object", "properties": props, "required": req}}}
    s = {"type": "string"}
    grep = fn("grep", "Search files for a regular expression; returns path:line: text for each match (max 40).",
              {"pattern": s, "path": {"type": "string", "description": "file or directory, relative"}}, ["pattern"])
    if base_arm(arm) == "baseline":
        return [fn("read_file", "Return the whole file with line numbers.", {"path": s}, ["path"]), grep]
    # same tool name as the baseline, so the A/B measures what the tool returns, not how it is named
    return [fn("read_file", "Read a file. Small files come back whole with line numbers; large files come back as an "
               "outline (each function/class with its line range) plus the parts most likely relevant to `query`. "
               "Use read_lines to see any other lines.", {"path": s, "query": s}, ["path"]),
            fn("read_lines", "Return lines start..end of a file with line numbers.",
               {"path": s, "start": {"type": "integer"}, "end": {"type": "integer"}}, ["path", "start", "end"]), grep] \
        + ([fn("cortex_approve", "Pull the bodies of connected symbols offered in a [connected] menu, by id "
               "(e.g. [\"#3\"]).", {"ids": {"type": "array", "items": s}}, ["ids"])]
           if (ARMS[base_arm(arm)] or {}).get("menu") else [])


# Reader settings per arm (split 600 = the MCP server's default). "coproc" arms need --coproc <exported head>.
# "full" = every CortexNET part: coprocessor (additive + rank pick), symbol graph, connected-symbol menu (with the
# cortex_approve tool), on top of outline-first reads, sub-block splits and the trail.
# Any arm + "+fade" (e.g. cortex+fade) also fades stale reads to outlines in one batch near the context limit.
# "fulldyn" adds the refractory bodies and the charge-ordered fade (cortexnet/reader.py, fade.py). The "neuro*" arms of
# results/live_ab_llama_3_1_8b_session8k_neuro.md ran the coprocessor as a neuromorphic twin, since removed (it gave
# the same picks as the conventional head).
ARMS = {"baseline": None, "cortex": {}, "coproc": dict(coproc=True), "coproc_ab": dict(coproc=True, policy="rule_ab"),
        "full": dict(coproc=True, menu=True, graph=True),
        "fulldyn": dict(coproc=True, menu=True, graph=True, refractory=True, fade_order="charge"),
        "coprocdyn": dict(coproc=True, refractory=True, fade_order="charge")}


def base_arm(arm):
    return arm.split("+")[0]


class Tools:
    """task: the user's question; every cortex arm reads with query = task + the call's own query and the agent's
    words of this turn as the snapshot, the same views the proxy (aware mode) and the coprocessor's training use."""

    def __init__(self, root, arm, state, coproc=None, task="", ctx_base=2000):
        self.root, self.arm, self.task = root, arm, task
        cfg = ARMS[base_arm(arm)]
        kw = dict(cfg or {})
        self.fade_order = kw.pop("fade_order", "age")
        which = kw.pop("coproc", False)
        if which:
            cp = coproc.get(which) if isinstance(coproc, dict) else coproc
            if cp is None:
                raise SystemExit(f"arm {arm} needs --coproc <head.npz>")
            kw.update(coproc=cp, ctx_base=ctx_base)
        if kw.pop("graph", False):
            from cortexnet.graph import CoNeedGraph
            kw["graph"] = CoNeedGraph()
        self.reader = None if cfg is None else Reader(state, split_tokens=600, **kw)

    def outline(self, path):
        """The file's outline (what a faded read leaves behind), or None if the path does not resolve."""
        try:
            p = self._p(path)
            with open(p, encoding="utf8", errors="replace") as fh:
                text = fh.read()
        except (OSError, ValueError):
            return None
        return outline(text, p, symbols(text, p, split_tokens=600))

    def _p(self, path):
        p = os.path.normpath(os.path.join(self.root, (path or ".").lstrip("/\\")))
        if not p.startswith(os.path.normpath(self.root)):
            raise ValueError("path outside the workspace")
        if not os.path.exists(p):      # "reader.py" for cortexnet/reader.py: take the one listed file it ends with
            want = "/" + (path or "").replace("\\", "/").lstrip("/")
            hits = [os.path.join(dp, f) for dp, _, fs in os.walk(self.root) if ".state" not in dp for f in fs
                    if ("/" + os.path.relpath(os.path.join(dp, f), self.root).replace("\\", "/")).endswith(want)]
            if len(hits) == 1:
                return hits[0]
        return p

    def call(self, name, a, snapshot=""):
        try:
            if name == "grep":
                rx = re.compile(a["pattern"])
                base = self._p(a.get("path", "."))
                files = [base] if os.path.isfile(base) else [os.path.join(dp, f) for dp, _, fs in os.walk(base)
                                                            for f in sorted(fs) if f.endswith((".py", ".md"))]
                out = []
                for f in files:
                    with open(f, encoding="utf8", errors="replace") as fh:
                        for i, l in enumerate(fh, 1):
                            if rx.search(l):
                                out.append(f"{os.path.relpath(f, self.root)}:{i}: {l.rstrip()[:200]}")
                return "\n".join(out[:40]) or "no matches"
            if name == "read_file" and self.reader:
                return self.reader.read(self._p(a["path"]), (self.task + "\n" + (a.get("query") or "")).strip(),
                                        snapshot=snapshot)
            if name == "read_file":
                with open(self._p(a["path"]), encoding="utf8", errors="replace") as fh:
                    lines = fh.read().split("\n")
                return numbered(lines, 1, len(lines))
            if name == "cortex_approve" and self.reader is not None and self.reader.menu_on:
                ids = a.get("ids") or []
                return self.reader.approve([ids] if isinstance(ids, str) else list(ids))
            if name == "read_lines" and self.reader:
                return self.reader.range(self._p(a["path"]), a["start"], a["end"])
            return f"unknown tool {name}"
        except (OSError, KeyError, ValueError, re.error, TypeError) as e:
            return f"error: {type(e).__name__}: {e}"


def load_low_ram(model_id):
    """Build the model on the GPU and fill it tensor by tensor from the safetensors shards with plain reads.

    from_pretrained memory-maps whole shards, which on Windows reserves page-file commit for the full file; with
    little free RAM and a full system drive that fails (os error 1455). Peak host RAM here is one tensor.
    """
    import glob as _glob
    import struct

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        from transformers.initialization import no_init_weights      # transformers >= 5
    except ImportError:
        from transformers.modeling_utils import no_init_weights

    path = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    cfg = AutoConfig.from_pretrained(path)
    with no_init_weights(), torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    dt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    loaded = set()
    for shard in sorted(_glob.glob(os.path.join(path, "*.safetensors"))):
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            base = 8 + n
            for name, meta in header.items():
                if name == "__metadata__" or name not in params:
                    continue
                a, b = meta["data_offsets"]
                f.seek(base + a)
                buf = bytearray(f.read(b - a))
                t = torch.frombuffer(buf, dtype=dt[meta["dtype"]]).reshape(meta["shape"])
                with torch.no_grad():
                    params[name].copy_(t.to("cuda", non_blocking=False))
                loaded.add(name)
                del buf, t
    if getattr(cfg, "tie_word_embeddings", False):
        model.tie_weights()
    missing = [k for k in dict(model.named_parameters()) if k not in loaded and not (
        getattr(cfg, "tie_word_embeddings", False) and k.endswith("lm_head.weight"))]
    if missing:
        raise RuntimeError(f"weights not found for {missing[:5]}")
    return model.eval()


class HFBackend:
    def __init__(self, model_id, think=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = load_low_ram(model_id)
        self.name, self.think = model_id + (" (thinking)" if think else ""), think

    def chat(self, messages, tools):
        import torch
        enc = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True, enable_thinking=self.think)
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=1500 if self.think else 400, do_sample=False)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        text = text.split("</think>", 1)[-1]          # reasoning is not kept in the history (Qwen3 guidance)
        calls = []
        for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
            try:
                calls.append(json.loads(m.group(1)))
            except ValueError:
                pass
        if not calls:
            for m in re.finditer(r"\{\s*\"name\"\s*:\s*\"(\w+)\"\s*,\s*\"arguments\"\s*:\s*(\{.*?\})\s*\}", text, re.S):
                try:
                    calls.append({"name": m.group(1), "arguments": json.loads(m.group(2))})
                except ValueError:
                    pass
        content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
        return content, calls, int(ids.shape[1])


def post(url, body, key=None, tries=4, timeout=600):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    for i in range(tries):
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(), headers)
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            # llama-server's 500 for a tool call it cannot parse is the model's output at temperature 0: a retry
            # returns the same thing, so fail at once
            unparsed = e.code == 500 and "expected" in msg and "format" in msg
            if e.code not in (429, 500, 502, 503, 504) or unparsed or i == tries - 1:
                raise RuntimeError(f"HTTP {e.code}: {msg}") from None
        except (urllib.error.URLError, TimeoutError):
            if i == tries - 1:
                raise
        time.sleep(2 ** i * 5)


class OpenAIBackend:
    """Chat completions. Messages are kept in the standard shape: tool_calls with ids and JSON-string arguments."""

    def __init__(self, base, model="local", key=None):
        self.base, self.name, self.key = base.rstrip("/"), model, key
        self.last = {}

    def chat(self, messages, tools):
        msgs = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
        r = post(self.base + "/chat/completions",
                 {"model": self.name, "messages": msgs, "tools": tools, "temperature": 0}, self.key)
        msg = r["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except ValueError:     # small local models sometimes emit broken JSON; the tool reports the missing args
                args = {}
            calls.append({"id": c.get("id"), "name": c["function"]["name"], "arguments": args})
        u = r.get("usage", {})
        # llama-server adds timings: prompt_n (tokens actually prefilled, after its prefix cache), prompt_ms,
        # predicted_n, predicted_ms
        self.last = dict(usage=u, timings=r.get("timings") or {})
        return msg.get("content") or "", calls, u.get("prompt_tokens", 0)


class MetaBackend:
    """Meta Model API, Responses endpoint. Turns are chained with previous_response_id, so each request sends only
    the new tool outputs; the usage still reports (and bills) the whole context, with repeats as cached input."""

    def __init__(self, ledger, model="muse-spark-1.3", base="https://api.meta.ai/v1", effort="medium",
                 summary="detailed", max_output=8000):
        self.key = os.environ.get("MODEL_API_KEY") or os.environ.get("LLAMA_API_KEY")
        if not self.key:
            raise SystemExit("set MODEL_API_KEY or LLAMA_API_KEY")
        if model not in PRICES:
            raise SystemExit(f"no price for {model}; add it to PRICES before spending money")
        self.ledger, self.base, self.name = ledger, base.rstrip("/"), model
        self.effort, self.summary, self.max_output = effort, summary, max_output
        self.last, self.last_in = {}, 20000

    def cost(self, u):
        p_in, p_cached, p_out = PRICES[self.name]
        cached = (u.get("input_tokens_details") or {}).get("cached_tokens", 0)
        return ((u.get("input_tokens", 0) - cached) * p_in + cached * p_cached + u.get("output_tokens", 0) * p_out) / 1e6

    @staticmethod
    def full_items(messages):
        """The whole tool history as Responses input items (used when a stored response cannot be chained)."""
        items = []
        for m in messages:
            if m["role"] == "user":
                items.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                items += [{"type": "function_call", "call_id": c["id"], "name": c["function"]["name"],
                           "arguments": c["function"]["arguments"]} for c in m.get("tool_calls") or []]
            elif m["role"] == "tool":
                items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
        return items

    def chat(self, messages, tools):
        prev = next((m for m in reversed(messages) if m.get("_resp_id")), None)
        if prev:
            new = messages[messages.index(prev) + 1:]          # tool outputs, and in session mode the next question
            items = [{"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]}
                     if m["role"] == "tool" else {"role": "user", "content": m["content"]}
                     for m in new if m["role"] in ("tool", "user")]
        else:
            items = self.full_items(messages)
        body = {"model": self.name, "input": items, "max_output_tokens": self.max_output, "store": True,
                "reasoning": {"effort": self.effort, "summary": self.summary},
                "instructions": next(m["content"] for m in messages if m["role"] == "system"),
                "tools": [dict(type="function", **t["function"]) for t in tools]}
        if prev:
            body["previous_response_id"] = prev["_resp_id"]
        # worst case for the next call: the whole history uncached, plus a full output
        p_in, _, p_out = PRICES[self.name]
        grow = sum(tokens(i.get("output") or i.get("content") or "") for i in self.full_items(messages))
        self.ledger.check(((max(self.last_in, grow) + grow) * p_in + self.max_output * p_out) / 1e6)
        try:
            r = post(self.base + "/responses", body, self.key)
        except RuntimeError as e:
            if not prev or re.match(r"HTTP 40[13]\b", str(e)):   # a broken chain: resend the whole history
                raise
            self.chained_fallbacks = getattr(self, "chained_fallbacks", 0) + 1
            body.pop("previous_response_id")
            body["input"] = self.full_items(messages)
            r = post(self.base + "/responses", body, self.key)
        u = r.get("usage") or {}
        usd = self.cost(u)
        self.ledger.add(usd, model=self.name, input=u.get("input_tokens", 0), output=u.get("output_tokens", 0))
        self.last_in = u.get("input_tokens", 0)
        text, calls, reasoning = [], [], []
        for it in r.get("output") or []:
            if it.get("type") == "reasoning":
                reasoning += [s.get("text", "") for s in it.get("summary") or []]
            elif it.get("type") == "function_call":
                try:
                    args = json.loads(it.get("arguments") or "{}")
                except ValueError:
                    args = {}
                calls.append({"id": it["call_id"], "name": it["name"], "arguments": args})
            elif it.get("type") == "message":
                text += [c.get("text", "") for c in it.get("content") or [] if c.get("type") == "output_text"]
        self.last = dict(usage=u, usd=usd, reasoning=reasoning, resp_id=r.get("id"), status=r.get("status"))
        return "\n".join(text), calls, u.get("input_tokens", 0)


def msg_tokens(msgs):
    return sum(tokens(str(m.get("content") or "")) + sum(tokens(c["function"]["arguments"]) + 8
                                                         for c in m.get("tool_calls") or []) for m in msgs)


def run_task(backend, root, arm, q, state, max_calls=8, trace=None, count_errors=False, hint=False, coproc=None,
             session=None, fade=None):
    """One question through the tool loop. With count_errors, a server error (e.g. the context overflowed) ends the
    task as a failure of this arm, recorded in st["error"], instead of raising.
    session: a dict kept across questions (one conversation per arm: its messages and tools); None = fresh per task.
    fade: dict(limit, mark, keep) for "+fade" arms; the token estimate is scaled by the server's own prompt counts."""
    specs = tool_specs(arm)
    files = sorted(os.path.relpath(os.path.join(dp, f), root).replace("\\", "/") for dp, _, fs in os.walk(root)
                   for f in fs if f.endswith(".py") and ".state" not in dp)
    system = SYSTEM.format(files="\n".join(files)) + (HINT if hint else "")
    sess = session if session is not None else {}
    if "msgs" not in sess:
        sess["msgs"] = [{"role": "system", "content": system}]
        sess["tools"] = Tools(root, arm, state, coproc, task=q, ctx_base=tokens(system) + tokens(q))
        sess["ratio"] = 1.0
    msgs, tools = sess["msgs"], sess["tools"]
    tools.task = q
    msgs.append({"role": "user", "content": q})
    st = dict(input_tokens=0, cached_tokens=0, output_tokens=0, reasoning_tokens=0, usd=0.0, tool_tokens=0, calls=0,
              answer="", prefill_tokens=0, prefill_ms=0.0, gen_ms=0.0, peak_context=0, seconds=0.0, error="",
              fades=0, context_end=0)
    t0 = time.time()
    for turn in range(max_calls + 1):
        if fade:
            forget = tools.reader.forget if tools.reader is not None and tools.reader.refractory else None
            st["fades"] += watermark_fade(msgs, sess["ratio"] * msg_tokens(msgs), fade["limit"], tools.outline,
                                          mark=fade["mark"], low=fade["low"], keep=fade["keep"],
                                          keep_frac=fade["keep_frac"], scale=sess["ratio"], order=tools.fade_order,
                                          on_fade=forget)
        sent = msg_tokens(msgs)
        try:
            content, calls, n_in = backend.chat(msgs, specs)
        except RuntimeError as e:
            if not count_errors or re.match(r"HTTP 40[13]\b", str(e)):   # auth problems abort the run
                raise
            st["error"] = f"{str(e)[:240]} (rejected request ~{int(sess['ratio'] * sent):,} tokens)"
            break
        if n_in:
            sess["ratio"] = n_in / max(sent, 1)       # server tokens per estimated token (tokenizer + template)
        last = getattr(backend, "last", {})
        u = last.get("usage") or {}
        out_n = u.get("output_tokens", u.get("completion_tokens", 0))
        st["input_tokens"] += n_in
        st["cached_tokens"] += (u.get("input_tokens_details") or u.get("prompt_tokens_details") or {}).get(
            "cached_tokens", 0)                                 # Responses or chat-completions field names
        st["output_tokens"] += out_n
        st["reasoning_tokens"] += (u.get("output_tokens_details") or u.get("completion_tokens_details") or {}).get(
            "reasoning_tokens", 0)
        st["usd"] += last.get("usd", 0.0)
        tm = last.get("timings") or {}
        st["prefill_tokens"] += tm.get("prompt_n", 0)
        st["prefill_ms"] += tm.get("prompt_ms", 0.0)
        st["gen_ms"] += tm.get("predicted_ms", 0.0)
        st["peak_context"] = max(st["peak_context"], n_in + out_n)
        for j, c in enumerate(calls):
            c.setdefault("id", f"call_{turn}_{j}")
        if not calls or st["calls"] >= max_calls:
            st["answer"] = content
        else:
            msgs.append({"role": "assistant", "content": content, "_resp_id": last.get("resp_id"), "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}} for c in calls]})
            snap = "\n".join([content or ""] + list(last.get("reasoning") or [])).strip()
            for c in calls:
                out = tools.call(c["name"], c.get("arguments") or {}, snapshot=snap)
                st["calls"] += 1
                st["tool_tokens"] += tokens(out)
                c["output_tokens"] = tokens(out)
                msgs.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": out})
        if trace:
            trace.write(json.dumps(dict(q=q, arm=arm, turn=turn, reasoning=last.get("reasoning"), content=content,
                                        calls=calls, usage=u, usd=last.get("usd")), default=str) + "\n")
        if st["answer"] or not calls or st["calls"] >= max_calls:
            break
    if st["answer"]:
        # the conversation goes on in session mode; the answer's response id is the chain's next link (--meta)
        # (only when that response has no unanswered tool calls, i.e. the cap did not end the task)
        rid = getattr(backend, "last", {}).get("resp_id") if not calls else None
        msgs.append({"role": "assistant", "content": st["answer"], "_resp_id": rid})
    st["seconds"] = time.time() - t0
    st["context_end"] = int(sess["ratio"] * msg_tokens(msgs))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default=None)
    ap.add_argument("--openai", default=None)
    ap.add_argument("--model", default="local")
    ap.add_argument("--tasks", type=int, default=len(TASKS))
    ap.add_argument("--think", action="store_true", help="Qwen3 thinking mode for the local backend")
    ap.add_argument("--meta", action="store_true", help="Meta Model API (Responses endpoint)")
    ap.add_argument("--effort", default="medium", help="reasoning effort for --meta")
    ap.add_argument("--budget", type=float, default=5.0, help="cap on total USD across all paid runs (ledger)")
    ap.add_argument("--first", type=int, default=0, help="skip the first N tasks")
    ap.add_argument("--tag", default="", help="suffix for the results file")
    ap.add_argument("--count-errors", action="store_true",
                    help="a server error (e.g. context overflow) fails that arm instead of dropping the task")
    ap.add_argument("--hint", action="store_true", help="add a call example to the system prompt (small models)")
    ap.add_argument("--arms", default="baseline,cortex", help=f"comma-separated, first is the reference: {list(ARMS)}")
    ap.add_argument("--coproc", default="", help="exported Cortex-SubNN head (.npz) for the coproc arms")
    ap.add_argument("--session", action="store_true",
                    help="all questions in one conversation per arm (context accumulates, as in a real session)")
    ap.add_argument("--ctx", type=int, default=8192, help="server context size, for the +fade watermark")
    ap.add_argument("--fade-mark", type=float, default=0.85, help="+fade arms fade once the context passes this share")
    ap.add_argument("--fade-low", type=float, default=0.5, help="a fade batch shrinks the context below this share")
    ap.add_argument("--fade-keep-frac", type=float, default=0.25,
                    help="the kept newest results may hold at most this share of the context (the newest is always kept)")
    ap.add_argument("--fade-keep", type=int, default=4, help="reads of files touched in the last N tool calls stay")
    ap.add_argument("--fixture-ref", default="", help="git commit for the cortexnet/ sources the agent reads "
                    "(default: the working tree)")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if base_arm(x) not in ARMS or x not in (base_arm(x), base_arm(x) + "+fade")]
    if bad or len(arms) < 2:
        raise SystemExit(f"--arms: need 2+ of {list(ARMS)} (optionally +fade), got {arms}")
    if a.meta and any(x.endswith("+fade") for x in arms):
        raise SystemExit("+fade needs a backend that resends the history (--openai/--hf); --meta chains server-side")
    if a.session and not a.count_errors:
        raise SystemExit("--session needs --count-errors (an overflow must fail that arm, not drop the task)")
    fade = dict(limit=a.ctx, mark=a.fade_mark, low=a.fade_low, keep=a.fade_keep, keep_frac=a.fade_keep_frac)
    coproc = None
    if any((ARMS[base_arm(x)] or {}).get("coproc") for x in arms):
        from cortexnet.coproc import Coprocessor, MiniLM
        cp = Coprocessor(a.coproc, encoder=MiniLM())           # raises on a missing head or unusable calibration
        coproc = {True: cp}
        print(f"coprocessor {a.coproc}: calib {[round(float(x), 3) for x in cp.calib]}", flush=True)
    if a.meta:
        backend = MetaBackend(Ledger(a.budget), a.model if a.model != "local" else "muse-spark-1.3", effort=a.effort)
    elif a.openai:
        backend = OpenAIBackend(a.openai, a.model, os.environ.get("OPENAI_API_KEY"))
    else:
        backend = HFBackend(a.hf or "Qwen/Qwen3-1.7B", a.think)
    root = fixture(a.fixture_ref)
    res = {arm: [] for arm in arms}
    t0, stopped, skipped = time.time(), "", []
    os.makedirs(TRACES, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9]+", "_", backend.name + (f"_{a.effort}" if a.meta else "") + a.tag)
    trace = open(os.path.join(TRACES, f"live_ab_{name}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"), "w", encoding="utf8")
    sessions = {arm: {} for arm in arms} if a.session else None
    try:
        for i, (q, rx) in enumerate(TASKS[a.first:a.tasks], a.first):
            pair = {}
            try:
                for arm in res:            # a task counts only when both arms finished it
                    st = run_task(backend, root, arm, q, os.path.join(root, ".state", arm.replace("+", "_")),
                                  trace=trace, count_errors=a.count_errors, hint=a.hint, coproc=coproc,
                                  session=sessions[arm] if sessions else None,
                                  fade=fade if arm.endswith("+fade") else None)
                    st["ok"] = not st["error"] and bool(re.search(rx, st["answer"], re.I))
                    pair[arm] = st
            except RuntimeError as e:      # API error after retries: drop this task from both arms, keep going
                skipped.append(f"task {i + 1}: {e}")
                print(skipped[-1], flush=True)
                continue
            for arm in res:
                res[arm].append(pair[arm])
            print(f"task {i + 1}: " + "  ".join(
                f"{arm} ok={p['ok']} in={p['input_tokens']} calls={p['calls']} ${p['usd']:.4f} {p['seconds']:.0f}s"
                + (f" ctx={p['context_end']}" if a.session else "") + (f" faded={p['fades']}" if p["fades"] else "")
                + (f" ERROR {p['error'][:80]}" if p["error"] else "") for arm, p in pair.items()), flush=True)
    except BudgetExceeded as e:
        stopped = f"Stopped at the budget cap: {e}."
        print(stopped, flush=True)
    finally:
        trace.close()
        shutil.rmtree(root, ignore_errors=True)
    ref = arms[0]
    n = len(res[ref])
    L = [f"# Live A/B: {backend.name}{f' (reasoning effort {a.effort})' if a.meta else ''}, {n} code questions "
         f"({time.time() - t0:.0f}s)", "",
         "| arm | accuracy | input tokens (sum over turns) | cached | output (reasoning) | tool-output tokens | "
         "tool calls | cost |", "|---|---|---|---|---|---|---|---|"]
    for arm, rows in res.items():
        S = lambda k: sum(r[k] for r in rows)  # noqa: E731
        L.append(f"| {arm} | {S('ok')}/{n} | {S('input_tokens'):,} | {S('cached_tokens'):,} | "
                 f"{S('output_tokens'):,} ({S('reasoning_tokens'):,}) | {S('tool_tokens'):,} | {S('calls')} | "
                 f"${S('usd'):.3f} |")
    if n:
        b = sum(r["input_tokens"] for r in res[ref])
        L += [""] + [f"{arm} input tokens = {sum(r['input_tokens'] for r in res[arm]) / max(b, 1):.0%} of {ref}."
                     for arm in arms[1:]]
    if any(r["prefill_ms"] for rows in res.values() for r in rows):   # local server (llama-server timings)
        L += ["", "Local run: prefill tokens are what the server actually computed (after its prefix cache); peak "
              "context is the largest single request (prompt + completion).", "",
              "| arm | answered | server errors | prefill tokens | prefill tok/s | prefill s | generation s | "
              "wall s | peak context (max / mean) |", "|---|---|---|---|---|---|---|---|---|"]
        for arm, rows in res.items():
            S = lambda k: sum(r[k] for r in rows)  # noqa: E731
            pk = [r["peak_context"] for r in rows]
            L.append(f"| {arm} | {S('ok')}/{n} | {sum(bool(r['error']) for r in rows)} | {S('prefill_tokens'):,} | "
                     f"{S('prefill_tokens') / max(S('prefill_ms') / 1e3, 1e-9):,.0f} | {S('prefill_ms') / 1e3:.1f} | "
                     f"{S('gen_ms') / 1e3:.1f} | {S('seconds'):.0f} | {max(pk, default=0):,} / "
                     f"{sum(pk) / max(len(pk), 1):,.0f} |")
    if a.session:
        L += ["", f"Session mode: one conversation per arm, {n} questions in a row, server context {a.ctx:,}; +fade arms "
              f"shrink stale reads in one batch above {a.fade_mark:.0%}, down to {a.fade_low:.0%} (newest {a.fade_keep} results kept, at most {a.fade_keep_frac:.0%} of the context).", ""]
        for arm, rows in res.items():
            first = next((i for i, r in enumerate(rows, a.first + 1) if "context" in r["error"]), None)
            L.append(f"- {arm}: reads faded {sum(r['fades'] for r in rows)}, context at the end "
                     f"{rows[-1]['context_end'] if rows else 0:,}, first context overflow at question "
                     f"{first if first else 'none'}")
    errs = [(i, arm, r["error"]) for arm, rows in res.items() for i, r in enumerate(rows, a.first + 1) if r["error"]]
    if errs:
        L += ["", "Server errors counted as failures:"] + [f"- task {i} {arm}: {e[-200:]}" for i, arm, e in errs]
    if stopped:
        L += ["", stopped]
    if skipped:
        L += ["", "Skipped after API errors (both arms dropped):"] + [f"- {s}" for s in skipped]
    if getattr(backend, "chained_fallbacks", 0):
        L += ["", f"{backend.chained_fallbacks} turns resent the full history because the chained response was gone."]
    L += ["", f"Per task ({' / '.join(arms)}): ok, input tokens, tool calls", ""]
    for i, rows in enumerate(zip(*(res[x] for x in arms)), a.first):
        L.append(f"- {i + 1}. {TASKS[i][0]}  {'/'.join(str(r['ok']) for r in rows)}, "
                 f"{'/'.join(format(r['input_tokens'], ',') for r in rows)}, {'/'.join(str(r['calls']) for r in rows)}")
    out = os.path.join(HERE, "results", f"live_ab_{name}.md")
    open(out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
