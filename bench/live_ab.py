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
from cortexnet import Reader, tokens  # noqa: E402
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


def fixture():
    d = tempfile.mkdtemp(dir=os.environ.get("TMP"))
    for src, name in SOURCES:
        dst = os.path.join(d, name)
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
    if arm == "baseline":
        return [fn("read_file", "Return the whole file with line numbers.", {"path": s}, ["path"]), grep]
    # same tool name as the baseline, so the A/B measures what the tool returns, not how it is named
    return [fn("read_file", "Read a file. Small files come back whole with line numbers; large files come back as an "
               "outline (each function/class with its line range) plus the parts most likely relevant to `query`. "
               "Use read_lines to see any other lines.", {"path": s, "query": s}, ["path"]),
            fn("read_lines", "Return lines start..end of a file with line numbers.",
               {"path": s, "start": {"type": "integer"}, "end": {"type": "integer"}}, ["path", "start", "end"]), grep]


class Tools:
    def __init__(self, root, arm, state):
        self.root, self.arm = root, arm
        self.reader = Reader(state, split_tokens=600) if arm == "cortex" else None   # the MCP server's default

    def _p(self, path):
        p = os.path.normpath(os.path.join(self.root, (path or ".").lstrip("/\\")))
        if not p.startswith(os.path.normpath(self.root)):
            raise ValueError("path outside the workspace")
        return p

    def call(self, name, a):
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
                return self.reader.read(self._p(a["path"]), a.get("query", ""))
            if name == "read_file":
                with open(self._p(a["path"]), encoding="utf8", errors="replace") as fh:
                    lines = fh.read().split("\n")
                return numbered(lines, 1, len(lines))
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


def post(url, body, key=None, tries=4):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    for i in range(tries):
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(), headers)
            return json.loads(urllib.request.urlopen(req, timeout=600).read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            if e.code not in (429, 500, 502, 503, 504) or i == tries - 1:
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
        calls = [{"id": c["id"], "name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}
                 for c in msg.get("tool_calls") or []]
        u = r.get("usage", {})
        self.last = dict(usage=u)
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
            new = messages[messages.index(prev) + 1:]
            items = [{"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]}
                     for m in new if m["role"] == "tool"]
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
            if not prev or "not found" not in str(e):
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


def run_task(backend, root, arm, q, state, max_calls=8, trace=None):
    tools = Tools(root, arm, state)
    specs = tool_specs(arm)
    files = sorted(os.path.relpath(os.path.join(dp, f), root).replace("\\", "/") for dp, _, fs in os.walk(root)
                   for f in fs if f.endswith(".py") and ".state" not in dp)
    msgs = [{"role": "system", "content": SYSTEM.format(files="\n".join(files))}, {"role": "user", "content": q}]
    st = dict(input_tokens=0, cached_tokens=0, output_tokens=0, reasoning_tokens=0, usd=0.0, tool_tokens=0, calls=0,
              answer="")
    for turn in range(max_calls + 1):
        content, calls, n_in = backend.chat(msgs, specs)
        last = getattr(backend, "last", {})
        u = last.get("usage") or {}
        st["input_tokens"] += n_in
        st["cached_tokens"] += (u.get("input_tokens_details") or {}).get("cached_tokens", 0)
        st["output_tokens"] += u.get("output_tokens", 0)
        st["reasoning_tokens"] += (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        st["usd"] += last.get("usd", 0.0)
        for j, c in enumerate(calls):
            c.setdefault("id", f"call_{turn}_{j}")
        if not calls or st["calls"] >= max_calls:
            st["answer"] = content
        else:
            msgs.append({"role": "assistant", "content": content, "_resp_id": last.get("resp_id"), "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}} for c in calls]})
            for c in calls:
                out = tools.call(c["name"], c.get("arguments") or {})
                st["calls"] += 1
                st["tool_tokens"] += tokens(out)
                c["output_tokens"] = tokens(out)
                msgs.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": out})
        if trace:
            trace.write(json.dumps(dict(q=q, arm=arm, turn=turn, reasoning=last.get("reasoning"), content=content,
                                        calls=calls, usage=u, usd=last.get("usd")), default=str) + "\n")
        if st["answer"] or not calls or st["calls"] >= max_calls:
            break
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
    a = ap.parse_args()
    if a.meta:
        backend = MetaBackend(Ledger(a.budget), a.model if a.model != "local" else "muse-spark-1.3", effort=a.effort)
    elif a.openai:
        backend = OpenAIBackend(a.openai, a.model, os.environ.get("OPENAI_API_KEY"))
    else:
        backend = HFBackend(a.hf or "Qwen/Qwen3-1.7B", a.think)
    root = fixture()
    res = {arm: [] for arm in ("baseline", "cortex")}
    t0, stopped, skipped = time.time(), "", []
    os.makedirs(TRACES, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9]+", "_", backend.name + (f"_{a.effort}" if a.meta else "") + a.tag)
    trace = open(os.path.join(TRACES, f"live_ab_{name}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"), "w", encoding="utf8")
    try:
        for i, (q, rx) in enumerate(TASKS[a.first:a.tasks], a.first):
            pair = {}
            try:
                for arm in res:            # a task counts only when both arms finished it
                    st = run_task(backend, root, arm, q, os.path.join(root, ".state", arm), trace=trace)
                    st["ok"] = bool(re.search(rx, st["answer"], re.I))
                    pair[arm] = st
            except RuntimeError as e:      # API error after retries: drop this task from both arms, keep going
                skipped.append(f"task {i + 1}: {e}")
                print(skipped[-1], flush=True)
                continue
            for arm in res:
                res[arm].append(pair[arm])
            print(f"task {i + 1}: " + "  ".join(f"{arm} ok={p['ok']} in={p['input_tokens']} calls={p['calls']} "
                                                f"${p['usd']:.4f}" for arm, p in pair.items()), flush=True)
    except BudgetExceeded as e:
        stopped = f"Stopped at the budget cap: {e}."
        print(stopped, flush=True)
    finally:
        trace.close()
        shutil.rmtree(root, ignore_errors=True)
    n = len(res["baseline"])
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
        b, c = (sum(r["input_tokens"] for r in res[k]) for k in ("baseline", "cortex"))
        L += ["", f"Cortex input tokens = {c / max(b, 1):.0%} of baseline."]
    if stopped:
        L += ["", stopped]
    if skipped:
        L += ["", "Skipped after API errors (both arms dropped):"] + [f"- {s}" for s in skipped]
    if getattr(backend, "chained_fallbacks", 0):
        L += ["", f"{backend.chained_fallbacks} turns resent the full history because the chained response was gone."]
    L += ["", "Per task (baseline / cortex): ok, input tokens, tool calls", ""]
    for i, (rb, rc) in enumerate(zip(res["baseline"], res["cortex"]), a.first):
        L.append(f"- {i + 1}. {TASKS[i][0]}  {rb['ok']}/{rc['ok']}, {rb['input_tokens']:,}/{rc['input_tokens']:,}, "
                 f"{rb['calls']}/{rc['calls']}")
    out = os.path.join(HERE, "results", f"live_ab_{name}.md")
    open(out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
