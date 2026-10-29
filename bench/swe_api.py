"""SWE-bench Lite with a hosted model as the agent: whole-file reads vs CortexNET vs CortexNET + coprocessor.

The same 40 instances, tools, call cap and scoring as the Glimmer 30B A/B (bench/swe_agent.py, results/swe_ab.md),
driven through a chat API instead of Glimmer's raw token format:
  --baseten   Muse Spark Open on Baseten (OpenAI-compatible); key from BASETEN_API_KEY
  --meta      Muse Spark 1.3 on the Meta Model API (Responses, turns chained server-side); key from LLAMA_API_KEY or
              MODEL_API_KEY; spend capped by --budget through the shared ledger (results/traces/spend.jsonl)
  --openai    any OpenAI-compatible server (e.g. vLLM serving Glimmer or a 70B model on Modal) with --model; key from
              the variable named by --key-env, if any
Arms:
  whole    read_file returns the whole file, numbered (at most 2,000 lines)
  cortex   outline-first CortexNET reads (v1 include model), query = the issue + the call's query, snapshot = the
           agent's words and reasoning of that turn
  coproc   as cortex, plus the Cortex-SubNN coprocessor's top-3 picks up front (the read_eval.md configuration that
           showed 0.497 gold-line recall against 0.312 for v1 on these tasks' reads)
--require-read: submit is accepted only for a file the agent has read (all arms alike). Llama 3.3 70B otherwise submits
a guess from the issue text on its first turn, so its reads, the thing under test, never happen.
--protocol v2 fixes all of the rules at once for any model (arms whole and cortex; 40 calls, 120k context, read before
submit, submit forced after one reminder on chat-completions backends); every episode records the rules it ran under.
Success, as in the Glimmer A/B: the submitted file is in the gold patch / the submitted symbol overlaps its changed
lines (no tests are run). Tokens: what the API reports per call (input, cached input, output, reasoning).

  python -I bench/swe_api.py prepare                      # SWE-bench Lite snapshots for the 40 ids -> D:/dev-cache/swe
  python -I bench/swe_api.py run --baseten --arms whole,cortex,coproc --coproc <head.npz>
  python -I bench/swe_api.py report --tag <tag>           # paired table -> results/swe_api_<tag>.md
"""
import argparse
import http.client
import io
import json
import os
import re
import sys
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import live_ab  # noqa: E402
import swe_agent as sa  # noqa: E402
from cortexnet import Reader  # noqa: E402
from cortexnet.outline import tokens  # noqa: E402

SWE = os.environ.get("CORTEXNET_SWE", "D:/dev-cache/swe")
RESULTS = os.path.join(HERE, "results")
IDS = os.path.join(RESULTS, "swe_ab_cortex.json")          # the Glimmer A/B's 40 instances
SYSTEM = sa.SYSTEM.replace("\n\nReasoning strength: low", "")
SYSTEM_WHOLE = sa.SYSTEM_WHOLE.replace("\n\nReasoning strength: low", "")
ARMS = ("whole", "cortex", "coproc")
# after the reminder, chat-completions backends must answer with a submit call (Muse Spark Open otherwise writes its
# patch as text and stops, more often the longer the episode; measured: bench/results/swe_api_*_wide.md)
SUBMIT_CHOICE = {"type": "function", "function": {"name": "submit"}}
# fixed rule sets, so a new model is measured exactly like the last one; v2 came out of the 2026-10-29 runs: the
# Glimmer limits (20 calls, 60k) decided outcomes before the model did, Muse Spark Open wrote its fix as text instead of
# calling submit, and Llama 3.3 70B submitted a guess before reading anything (bench/results/debut.md)
PROTOCOLS = {"v2": dict(arms="whole,cortex", max_calls=40, max_context=120000, require_tools=True)}
# require_tools: every turn must be a tool call (tool_choice "required"; the episode ends with submit). Llama 4 Maverick
# otherwise answered in prose on its first turn and never opened a file.


# ---------------------------------------------------------------- data
def prepare():
    """Snapshots of the 40 instances' repositories at their base commits: GitHub's tarball of each commit (a clone
    pulls the whole history, and git archive on a partial clone fetches blobs one by one)."""
    from datasets import load_dataset
    want = {e["instance_id"] for e in json.load(open(IDS))["episodes"]}
    keep = ("instance_id", "repo", "base_commit", "patch", "problem_statement")
    insts = [{k: r[k] for k in keep} for r in load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
             if r["instance_id"] in want]
    insts.sort(key=lambda r: r["instance_id"])
    assert len(insts) == len(want), (len(insts), len(want))
    for r in insts:
        snap = os.path.join(SWE, "snap", r["instance_id"])
        if os.path.exists(os.path.join(snap, ".done")):
            continue
        url = f"https://codeload.github.com/{r['repo']}/tar.gz/{r['base_commit']}"
        data = urllib.request.urlopen(url, timeout=600).read()
        skipped = 0
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for m in tf.getmembers():
                if not (m.isfile() or m.isdir()) or "/" not in m.name:
                    continue
                m.name = m.name.split("/", 1)[1]                 # drop the <repo>-<commit>/ top directory
                try:
                    tf.extract(m, snap, filter="data")
                except (OSError, tarfile.TarError):        # names Windows cannot hold; never source the patch needs
                    skipped += 1
        open(os.path.join(snap, ".done"), "w").write(str(skipped))
        print(f"  {r['instance_id']}" + (f" ({skipped} entries skipped)" if skipped else ""), flush=True)
    json.dump(insts, open(os.path.join(SWE, "instances.json"), "w"))
    print(f"{len(insts)} instances in {SWE}")


# ---------------------------------------------------------------- agent
class Baseten(live_ab.OpenAIBackend):
    """OpenAI-compatible chat completions; keeps the model's reasoning_content as the turn's reasoning."""
    require_tools = False

    def chat(self, messages, tools, tool_choice=None):
        msgs = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
        body = {"model": self.name, "messages": msgs, "tools": tools, "temperature": 0}
        if self.require_tools and not tool_choice:              # protocol v2: every turn acts through a tool
            body["tool_choice"] = "required"
        if tool_choice:                                         # a forced submit is short; capped so a degenerate
            body["tool_choice"], body["max_tokens"] = tool_choice, 4096   # constrained decode cannot run on
        r = live_ab.post(self.base + "/chat/completions", body, self.key, timeout=300)
        msg = r["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except ValueError:
                args = {}
            calls.append({"id": c.get("id"), "name": c["function"]["name"], "arguments": args})
        u = r.get("usage", {})
        self.last = dict(usage=u, reasoning=[msg.get("reasoning_content") or ""])
        return msg.get("content") or "", calls, u.get("prompt_tokens", 0)


class Workspace(sa.Workspace):
    """swe_agent's tools; the cortex arms read with the issue as the task and the agent's words as the snapshot."""

    def __init__(self, root, state, arm, issue, coproc=None, lock=None, require_read=False):
        super().__init__(root, state, "whole" if arm == "whole" else "cortex")
        self.issue, self.snapshot, self.lock = issue, "", lock or threading.Lock()
        self.require_read, self.opened = require_read, set()      # files read with read_file / read_lines
        self.seen = set()                                         # ... in an earlier turn (their output was seen)
        if arm == "coproc":
            self.reader = Reader(state, split_tokens=600, coproc=coproc, coproc_top=3, ctx_base=2000)

    def call(self, i, name, a):
        if name in ("read_file", "read_lines"):
            try:
                self.opened.add(self.rel(self.path(a.get("path", ""))))
            except ValueError:
                pass
        if name == "submit" and self.require_read:
            f = (a.get("file") or "").strip().lstrip("./")
            if f not in self.seen:                              # a read in the same turn does not count
                return (f"submit rejected: read {f or 'the file'} (read_file or read_lines) and look at the result "
                        "before submitting it; it is the file your fix changes")
        if name == "read_file" and self.arm != "whole":
            try:
                with self.lock:                                 # the encoder and its cache are shared
                    return self.reader.read(self.path(a["path"]), self.issue[:2000] + "\n" + a.get("query", ""),
                                            snapshot=self.snapshot)
            except (OSError, KeyError, ValueError, TypeError) as e:
                return f"{type(e).__name__}: {e}"
        try:
            return super().call(i, name, a)
        except (TypeError, AttributeError) as e:              # a malformed argument (e.g. a list as a pattern)
            return f"{type(e).__name__}: {e}"


def episode(backend, inst, arm, coproc, lock, max_calls=20, max_context=60000, require_read=False):
    snap = os.path.join(SWE, "snap", inst["instance_id"])
    state = os.path.join(SWE, "state", backend.tag, arm, inst["instance_id"])
    ws = Workspace(snap, state, arm, inst["problem_statement"], coproc, lock, require_read)
    tools = sa.TOOLS_WHOLE if arm == "whole" else sa.TOOLS
    msgs = [{"role": "system", "content": SYSTEM_WHOLE if arm == "whole" else SYSTEM},
            {"role": "user", "content": "Issue:\n" + inst["problem_statement"]}]
    ep = dict(instance_id=inst["instance_id"], arm=arm, input=0, cached=0, output=0, reasoning=0, usd=0.0, turns=0,
              calls=0, tool_tok={}, file_tok=0, peak=0, error="", hit_limit=False)
    t0 = time.time()
    while ep["calls"] < max_calls:
        err = ""
        for attempt in range(4):                                # a dropped connection is retried; an episode that
            try:                                                # still fails is an error, re-run on resume
                force = ep.get("nudged") and not ep.get("forced") and isinstance(backend, Baseten)
                content, calls, n_in = (backend.chat(msgs, tools, SUBMIT_CHOICE) if force
                                        else backend.chat(msgs, tools))
                ep["forced"] = ep.get("forced") or bool(force)
                err = ""
                break
            except live_ab.BudgetExceeded:
                raise
            except (OSError, http.client.HTTPException) as e:   # RemoteDisconnected, timeouts, resets
                err = f"{type(e).__name__}: {e}"
                time.sleep(10 * (attempt + 1))
            except RuntimeError as e:                           # the server answered with an HTTP error
                err = str(e)
                break
        if err:
            ep["error"] = err[:300]
            break
        last = backend.last
        u = last.get("usage") or {}
        ep["turns"] += 1
        ep["input"] += n_in
        ep["cached"] += ((u.get("input_tokens_details") or u.get("prompt_tokens_details") or {})
                         .get("cached_tokens", 0))
        ep["output"] += u.get("output_tokens", u.get("completion_tokens", 0))
        ep["reasoning"] += ((u.get("output_tokens_details") or u.get("completion_tokens_details") or {})
                            .get("reasoning_tokens", 0))
        ep["usd"] += last.get("usd", 0.0)
        ep["peak"] = max(ep["peak"], n_in)
        if not calls:                                           # a final message without submitting: one reminder,
            ep.setdefault("final", []).append(((content or "") + " | reasoning: "
                                               + " ".join(last.get("reasoning") or []))[-1500:])
            if ep.get("nudged"):                                # the same in every arm
                break
            ep["nudged"] = True
            msgs.append({"role": "assistant", "content": content or "", "_resp_id": last.get("resp_id")})
            msgs.append({"role": "user", "content": "You have not called submit. Call submit now with the file, the "
                                                    "function or class to change, and your proposed patch."})
            continue
        msgs.append({"role": "assistant", "content": content, "_resp_id": last.get("resp_id"), "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for c in calls]})
        ws.snapshot = "\n".join([content or ""] + list(last.get("reasoning") or [])).strip()[-4000:]
        ws.seen = set(ws.opened)                                # reads of earlier turns: their results were seen
        for c in calls:
            ep["calls"] += 1
            out = ws.call(ep["calls"], c["name"], c["arguments"])
            cap = sa.MAX_READ if c["name"] == "read_file" else sa.MAX_OUT
            out = out if len(out) <= cap else out[:cap] + "\n[truncated]"
            if max_calls - ep["calls"] == 3 and c["name"] != "submit":
                out += "\n[3 tool calls left: call submit soon]"
            n = tokens(out)
            ep["tool_tok"][c["name"]] = ep["tool_tok"].get(c["name"], 0) + n
            if c["name"] in ("read_file", "read_lines"):
                ep["file_tok"] += n
            msgs.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": out})
        if ws.submitted is not None:
            break
        if n_in >= max_context:
            ep["hit_limit"] = True
            break
    ep.update(sa.score_submission(inst, ws), submitted=list(ws.submitted) if ws.submitted else None,
              rules=dict(max_calls=max_calls, max_context=max_context, require_read=require_read),
              secs=round(time.time() - t0, 1))
    return ep


def run(a):
    if a.protocol:                                              # the rules are fixed before a run, not after it
        P = PROTOCOLS[a.protocol]
        a.arms, a.max_calls, a.max_context, a.require_read = P["arms"], P["max_calls"], P["max_context"], True
        Baseten.require_tools = P["require_tools"]
        a.tag = f"_{a.protocol}" + a.tag
    insts = json.load(open(os.path.join(SWE, "instances.json")))[:a.n or None]
    if a.ids:
        insts = [i for i in insts if i["instance_id"] in a.ids.split(",")]
    if a.meta:
        backend = live_ab.MetaBackend(live_ab.Ledger(a.budget), a.model or "muse-spark-1.3", effort=a.effort)
        backend.tag = f"meta_{backend.name}_{a.effort}"
    elif a.openai:
        if not a.model:
            raise SystemExit("--openai needs --model")
        backend = Baseten(a.openai, a.model, os.environ.get(a.key_env) if a.key_env else None)
        backend.tag = "openai_" + re.sub(r"[^A-Za-z0-9]+", "_", a.model).strip("_").lower()
    else:
        key = os.environ.get("BASETEN_API_KEY")
        if not key:
            raise SystemExit("set BASETEN_API_KEY")
        backend = Baseten("https://inference.baseten.co/v1", a.model or "meta-models/muse-spark-open-1026", key)
        backend.tag = "baseten_muse_spark_open"
    backend.tag += a.tag
    coproc = None
    arms = [x for x in a.arms.split(",") if x]
    if "coproc" in arms:
        from cortexnet.coproc import Coprocessor, MiniLM
        coproc = Coprocessor(a.coproc, encoder=MiniLM())
    lock = threading.Lock()
    out = os.path.join(RESULTS, "traces", f"swe_api_{backend.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):                                     # resume: skip finished (instance, arm) pairs
        done = {(e["instance_id"], e["arm"]) for e in map(json.loads, open(out, encoding="utf8")) if not e["error"]}
    jobs = [(i, arm) for i in insts for arm in arms if (i["instance_id"], arm) not in done]
    print(f"{backend.tag}: {len(jobs)} episodes ({len(done)} done before)", flush=True)
    wlock = threading.Lock()

    def one(job):
        inst, arm = job
        b = backend if a.meta else Baseten(backend.base, backend.name, backend.key)   # one client per thread
        b.tag = backend.tag
        if a.meta:
            b = live_ab.MetaBackend(backend.ledger, backend.name, effort=a.effort)
            b.tag = backend.tag
        ep = episode(b, inst, arm, coproc, lock, max_calls=a.max_calls, max_context=a.max_context,
                     require_read=a.require_read)
        with wlock:
            with open(out, "a", encoding="utf8") as f:
                f.write(json.dumps(ep) + "\n")
            print(f"  {arm:6s} {inst['instance_id']:40s} file {ep['file_ok']!s:5s} symbol {ep['symbol_ok']!s:5s} "
                  f"in {ep['input']:>7,} out {ep['output']:>6,} calls {ep['calls']:>2} ${ep['usd']:.3f}"
                  + (f" ERROR {ep['error'][:80]}" if ep["error"] else ""), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        for f in [ex.submit(one, j) for j in jobs]:
            f.result()
    print(f"done; episodes in {out}")


# ---------------------------------------------------------------- report
def boot_ratio(x, y, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = [x[i].sum() / max(y[i].sum(), 1e-9) for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return x.sum() / max(y.sum(), 1e-9), np.percentile(r, 2.5), np.percentile(r, 97.5)


def report(a):
    path = os.path.join(RESULTS, "traces", f"swe_api_{a.tag}.jsonl")
    eps = {}
    for e in map(json.loads, open(path, encoding="utf8")):
        if not e["error"]:
            eps[(e["instance_id"], e["arm"])] = e
    arms = [x for x in ARMS if any(k[1] == x for k in eps)]
    ids = sorted({k[0] for k in eps if all((k[0], x) in eps for x in arms)})
    E = {x: [eps[(i, x)] for i in ids] for x in arms}
    L = [f"# SWE-bench Lite, {a.tag}: " + " vs ".join(arms), "",
         f"Paired over the {len(ids)} instances every arm finished (of 40; the Glimmer A/B's set). Success = the "
         "submitted file is in the gold patch / the submitted symbol overlaps its changed lines (no tests run). "
         "Ratios against whole-file reads with 95% intervals from a paired bootstrap over instances.", "",
         "| metric | " + " | ".join(arms) + " | " + " | ".join(f"{x} / whole" for x in arms if x != "whole") + " |",
         "|---|" + "---|" * (2 * len(arms) - 1)]
    for key, label in (("input", "input tokens (sum over turns)"), ("cached", "of which cached"),
                       ("output", "output tokens (incl. reasoning)"), ("usd", "cost (USD)"),
                       ("peak", "peak context"), ("file_tok", "file tokens put into context"), ("turns", "turns"),
                       ("calls", "tool calls")):
        row = [f"{np.mean([e[key] for e in E[x]]):,.4f}" if key == "usd" else f"{np.mean([e[key] for e in E[x]]):,.0f}"
               for x in arms]
        rat = []
        if "whole" in arms:
            for x in arms:
                if x != "whole":
                    m, lo, hi = boot_ratio([e[key] for e in E[x]], [e[key] for e in E["whole"]])
                    rat.append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]")
        L.append(f"| {label} (mean) | " + " | ".join(row) + " | " + " | ".join(rat) + " |")
    for key, label in (("file_ok", "submitted the gold file"), ("symbol_ok", "submitted the gold symbol"),
                       ("hit_limit", "hit the context cap")):
        L.append(f"| {label} | " + " | ".join(f"{np.mean([e[key] for e in E[x]]):.0%}" for x in arms) + " |"
                 + " |" * (len(arms) - 1))
    if "whole" in arms:
        L += [""]
        for x in arms:
            if x == "whole":
                continue
            only_x = sum(e["symbol_ok"] and not w["symbol_ok"] for e, w in zip(E[x], E["whole"]))
            only_w = sum(w["symbol_ok"] and not e["symbol_ok"] for e, w in zip(E[x], E["whole"]))
            from math import comb
            n = only_x + only_w
            p = min(1.0, 2 * sum(comb(n, k) for k in range(min(only_x, only_w) + 1)) / 2 ** n) if n else 1.0
            L.append(f"- gold symbol, {x} vs whole: {only_x} instances only {x} solved, {only_w} only whole "
                     f"(exact McNemar p = {p:.2f})")
    out = os.path.join(RESULTS, f"swe_api_{a.tag}.md")
    open(out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--baseten", action="store_true")
    g.add_argument("--meta", action="store_true")
    g.add_argument("--openai", default="", help="base URL of an OpenAI-compatible server")
    r.add_argument("--key-env", default="", help="environment variable holding the --openai server's key")
    r.add_argument("--model", default="")
    r.add_argument("--effort", default="medium")
    r.add_argument("--budget", type=float, default=40.0, help="cap on total USD across all paid runs (ledger)")
    r.add_argument("--arms", default="whole,cortex,coproc")
    r.add_argument("--coproc", default="D:/dev-cache/cortexnet/subnn/head_minilm_c64_v2.npz")
    r.add_argument("--n", type=int, default=0, help="first N instances only (smoke test)")
    r.add_argument("--ids", default="", help="comma-separated instance ids only")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--tag", default="")
    r.add_argument("--max-calls", type=int, default=20, help="tool calls per episode (the Glimmer A/B: 20)")
    r.add_argument("--max-context", type=int, default=60000, help="stop an episode once a prompt reaches this many "
                   "tokens (the Glimmer A/B: 60k)")
    r.add_argument("--require-read", action="store_true", help="submit is accepted only for a file the agent read "
                   "(for a model that otherwise submits a guess on its first turn, e.g. Llama 3.3 70B)")
    r.add_argument("--protocol", default="", choices=["", "v2"], help="a fixed set of rules for every model: v2 = arms "
                   "whole,cortex, 40 calls, 120k context, read before submit, submit forced after one reminder")
    p = sub.add_parser("report")
    p.add_argument("--tag", required=True, help="e.g. baseten_muse_spark_open or meta_muse-spark-1.3_medium")
    a = ap.parse_args()
    {"prepare": lambda: prepare(), "run": lambda: run(a), "report": lambda: report(a)}[a.cmd]()


if __name__ == "__main__":
    main()
