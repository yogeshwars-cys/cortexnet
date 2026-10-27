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
Usage: python -I bench/live_ab.py [--hf Qwen/Qwen3-1.7B | --openai http://127.0.0.1:8080/v1] [--tasks N]
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortexnet import Reader, tokens  # noqa: E402
from cortexnet.reader import numbered  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
META = os.path.join(os.path.dirname(REPO), "meta", "cortex")
SOURCES = [(os.path.join(REPO, "cortexnet"), "cortexnet"), (META, "probe")]
TASKS = [
    ("What is the default value of full_below in the Reader class?", r"\b800\b"),
    ("What value does the DECAY constant in reader.py have?", r"\b0?\.7\b"),
    ("In learn.py, what prior weight does the 'changed' feature get?", r"\b4(\.0)?\b"),
    ("What is the default learning rate (lr) of IncludeModel?", r"\b0?\.1\b"),
    ("What is the default section size used by sections() in outline.py?", r"\b40\b"),
    ("Which protocolVersion does the MCP server return by default when the client sends none?", r"2025-06-18"),
    ("In the probe's shift.py, how many questions long is the teach window W?", r"\b10\b"),
    ("In the probe's shift.py, what is the prefetch threshold TAU?", r"\b0?\.15\b"),
    ("In the probe's models.py, how many latent dimensions (LATENTS) does the ridge ingress use?", r"\b24\b"),
    ("Which Reader method records whether a request was a miss?", r"\brange\b"),
    ("In learn.py, which function computes the symbol-graph edge weight between two symbols?", r"\blink\b"),
    ("In the probe's sim.py, which policy evicts the resident turn whose next use is furthest away?", r"belady", ),
]
SYSTEM = ("You answer questions about the code in this workspace. Always look the answer up with the tools before "
          "answering; never answer from memory. Paths are relative to the workspace root. Files:\n{files}\n"
          "When you know the answer, reply with one short sentence that starts with 'ANSWER:'.")


def fixture():
    d = tempfile.mkdtemp(dir=os.environ.get("TMP"))
    for src, name in SOURCES:
        shutil.copytree(src, os.path.join(d, name), ignore=shutil.ignore_patterns("__pycache__", "*.json", "*.npz"))
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
        self.reader = Reader(state) if arm == "cortex" else None

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


class OpenAIBackend:
    def __init__(self, base, model="local"):
        self.base, self.name = base.rstrip("/"), model

    def chat(self, messages, tools):
        import urllib.request
        body = json.dumps({"model": self.name, "messages": messages, "tools": tools, "temperature": 0}).encode()
        req = urllib.request.Request(self.base + "/chat/completions", body, {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        msg = r["choices"][0]["message"]
        calls = [{"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}
                 for c in msg.get("tool_calls") or []]
        return msg.get("content") or "", calls, r.get("usage", {}).get("prompt_tokens", 0)


def run_task(backend, root, arm, q, state, max_calls=8):
    tools = Tools(root, arm, state)
    specs = tool_specs(arm)
    files = sorted(os.path.relpath(os.path.join(dp, f), root).replace("\\", "/") for dp, _, fs in os.walk(root)
                   for f in fs if f.endswith(".py") and ".state" not in dp)
    msgs = [{"role": "system", "content": SYSTEM.format(files="\n".join(files))}, {"role": "user", "content": q}]
    st = dict(input_tokens=0, tool_tokens=0, calls=0, answer="")
    for _ in range(max_calls + 1):
        content, calls, n_in = backend.chat(msgs, specs)
        st["input_tokens"] += n_in
        if not calls or st["calls"] >= max_calls:
            st["answer"] = content
            break
        msgs.append({"role": "assistant", "content": content, "tool_calls": [
            {"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments", {})}} for c in calls]})
        for c in calls:
            out = tools.call(c["name"], c.get("arguments") or {})
            st["calls"] += 1
            st["tool_tokens"] += tokens(out)
            msgs.append({"role": "tool", "name": c["name"], "content": out})
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default=None)
    ap.add_argument("--openai", default=None)
    ap.add_argument("--model", default="local")
    ap.add_argument("--tasks", type=int, default=len(TASKS))
    ap.add_argument("--think", action="store_true", help="Qwen3 thinking mode for the local backend")
    a = ap.parse_args()
    backend = OpenAIBackend(a.openai, a.model) if a.openai else HFBackend(a.hf or "Qwen/Qwen3-1.7B", a.think)
    root = fixture()
    res = {arm: [] for arm in ("baseline", "cortex")}
    t0 = time.time()
    for i, (q, rx) in enumerate(TASKS[:a.tasks]):
        for arm in res:
            st = run_task(backend, root, arm, q, os.path.join(root, ".state", arm))
            st["ok"] = bool(re.search(rx, st["answer"], re.I))
            res[arm].append(st)
        print(f"task {i + 1}: " + "  ".join(f"{arm} ok={res[arm][-1]['ok']} in={res[arm][-1]['input_tokens']} "
                                            f"calls={res[arm][-1]['calls']}" for arm in res), flush=True)
    shutil.rmtree(root, ignore_errors=True)
    L = [f"# Live A/B: {backend.name}, {len(res['baseline'])} code questions ({time.time() - t0:.0f}s)", "",
         "| arm | accuracy | input tokens (sum over turns) | tool-output tokens | tool calls |", "|---|---|---|---|---|"]
    for arm, rows in res.items():
        L.append(f"| {arm} | {sum(r['ok'] for r in rows)}/{len(rows)} | {sum(r['input_tokens'] for r in rows):,} | "
                 f"{sum(r['tool_tokens'] for r in rows):,} | {sum(r['calls'] for r in rows)} |")
    out = os.path.join(HERE, "results", f"live_ab_{re.sub(r'[^A-Za-z0-9]+', '_', backend.name)}.md")
    open(out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
