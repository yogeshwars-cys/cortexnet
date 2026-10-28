"""Proxy A/B: the same agent CLI runs the same task through cortexnet-proxy, rewriting vs passthrough.

Arms (each run gets its own in-process proxy and fresh state, so no rewrite or read memory leaks between runs)
  cortex       fresh whole-file reads of large source files go upstream as CortexNET's outline-first view
  aware        cortex, plus the agent is told about outline-first reads (system prompt note) and the bodies are
               chosen from its own task, reasoning and sibling tool calls (Rewriter aware=True)
  passthrough  requests forwarded unchanged; usage logged the same way (the baseline)
Trials alternate which arm goes first, since agents with cross-session memory (Antigravity) favour the second run.

Agents (non-interactive print mode; the proxy sits on the model-API egress)
  agy      Antigravity CLI   upstream https://daily-cloudcode-pa.googleapis.com, redirected via CLOUD_CODE_URL
  claude   Claude Code       upstream http://127.0.0.1:3458 (the local fallback proxy) or https://api.anthropic.com;
                             --settings overrides an ANTHROPIC_BASE_URL set in user settings
  codex    Codex CLI         upstream https://api.openai.com; only with API-key auth (ChatGPT login ignores
                             OPENAI_BASE_URL)
  codex-meta  Codex CLI on Muse Spark (Meta Model API, Responses wire, key in LLAMA_API_KEY). Each run gets a
              fresh CODEX_HOME holding only the provider config: no user hooks, MCP servers, skills or AGENTS.md,
              so no other token reducer sits between the agent and the proxy. Hard cap: --max-usd per run.
  claude-meta Claude Code on Muse Spark via the Meta API's Anthropic-compatible /v1/messages. --bare (auth only from
              ANTHROPIC_API_KEY, set to LLAMA_API_KEY, so no Anthropic login token can leave), a fresh
              CLAUDE_CONFIG_DIR (no user settings, hooks, MCP or memory), read-only tools. Same --max-usd cap.

Metrics per run: model requests, input tokens as the provider reports them (cache-weighted for Anthropic:
input + 1.25 cache writes + 0.1 cache reads; Gemini: prompt and implicitly cached tokens), output tokens, rewrites
and tokens saved at the read, tool calls by name, wall time. Answers are saved for a quality check; tokens saved
only count if the answer is as good.

Usage: python -I bench/proxy_ab.py --agent agy [--task audit | --task-file F] [--trials 2] [--repo DIR]
       [--upstream URL] [--arms cortex,passthrough]
Writes results/proxy_ab_<agent>.md (table) and results/traces/proxy_ab/<agent>-<time>/ (answers, runs.json).
These runs spend the agent's own quota or credits.
"""
import argparse
import collections
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortexnet.mcp_server import options  # noqa: E402
from cortexnet.proxy import Proxy, Reader, Rewriter, totals  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TASKS = {
    "audit": (
        "Audit the include-model learning loop in this repo end to end. Trace (1) where training labels come from: "
        "range misses, menu approvals, menu items never approved, and edits; (2) where the model is updated and "
        "saved; and (3) how its probabilities choose which bodies Reader.read includes, under the budget policy and "
        "under rule_ab (cortexnet/policy.py), including the coprocessor override. Cite file:line for every claim. "
        "Then list the three most serious bugs or risks you find, each with a concrete failing scenario, and propose "
        "a fix for the top one as a diff. Do not apply it and do not edit any file."),
    "proxy": (
        "Explain how cortexnet/proxy.py keeps the provider's prefix cache valid while rewriting tool results, for "
        "each wire format it supports. Cite file:line for every claim, and list two cases where a rewrite could "
        "still invalidate the cache. Do not edit any file."),
}
UPSTREAM = {"agy": "https://daily-cloudcode-pa.googleapis.com", "claude": "http://127.0.0.1:3458",
            "codex": "https://api.openai.com", "codex-meta": "https://api.meta.ai",
            "claude-meta": "https://api.meta.ai"}
METERED = ("codex-meta", "claude-meta")
META_MODEL = "muse-spark-1.3"
META_PRICE = (1.25, 0.15, 4.25)     # USD per 1M tokens: input, cached input, output (bench/live_ab.py PRICES)
CODEX_META = """model = "{model}"
model_provider = "meta"
model_reasoning_effort = "medium"
approval_policy = "never"
sandbox_mode = "read-only"

[model_providers.meta]
name = "Meta Model API"
base_url = "{url}/v1"
env_key = "LLAMA_API_KEY"
wire_api = "responses"
"""


def usd(r):
    """Spend so far on the Meta API, from the proxy's usage totals: Responses fields (input_tokens includes cached)
    or Messages fields (input_tokens excludes cache reads and writes)."""
    p_in, p_cached, p_out = META_PRICE
    if "cache_read_input_tokens" in r or "cache_creation_input_tokens" in r:
        fresh = r.get("input_tokens", 0) + r.get("cache_creation_input_tokens", 0)
        cached = r.get("cache_read_input_tokens", 0)
    else:
        cached = r.get("input_tokens_details.cached_tokens", 0)
        fresh = r.get("input_tokens", 0) - cached
    return (fresh * p_in + cached * p_cached + r.get("output_tokens", 0) * p_out) / 1e6


def command(agent, task, state, url, template=None):
    if template:
        return [task if w == "{task}" else w for w in shlex.split(template)]
    if agent == "agy":
        return ["agy", "-p", task, "--print-timeout", "0"]
    if agent == "claude":
        override = os.path.join(state, "claude-settings.json")
        with open(override, "w", encoding="utf8") as f:
            json.dump({"env": {"ANTHROPIC_BASE_URL": url}}, f)
        return ["claude", "-p", task, "--settings", override, "--output-format", "text"]
    if agent == "codex":
        return ["codex", "exec", task]
    if agent == "codex-meta":
        home = os.path.join(state, "codex-home")
        os.makedirs(home, exist_ok=True)
        with open(os.path.join(home, "config.toml"), "w", encoding="utf8") as f:
            f.write(CODEX_META.format(model=META_MODEL, url=url))
        return ["codex", "exec", "--skip-git-repo-check", task]
    if agent == "claude-meta":
        return ["claude", "-p", task, "--bare", "--strict-mcp-config", "--model", META_MODEL, "--output-format",
                "text", "--no-session-persistence", "--disallowed-tools", "Edit", "Write", "NotebookEdit", "Bash",
                "PowerShell", "--allowed-tools", "Read", "Grep", "Glob"]
    raise ValueError(agent)


def run_once(agent, arm, task, repo, upstream, state, timeout, template=None, max_usd=None):
    os.makedirs(state, exist_ok=True)
    rw = Rewriter(state, reader=Reader(state_dir=os.path.join(state, "reader"), **options()),
                  rewrite=arm in ("cortex", "aware"), aware=arm == "aware")
    px = Proxy(upstream, rw, port=0).start()
    px.quiet = True
    env = dict(os.environ, ANTHROPIC_BASE_URL=px.url, OPENAI_BASE_URL=px.url + "/v1", CLOUD_CODE_URL=px.url)
    if agent == "codex-meta":
        env["CODEX_HOME"] = os.path.join(state, "codex-home")
    if agent == "claude-meta":
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        env["CLAUDE_CONFIG_DIR"] = os.path.join(state, "claude-home")
        os.makedirs(env["CLAUDE_CONFIG_DIR"], exist_ok=True)
        env.update(ANTHROPIC_API_KEY=os.environ["LLAMA_API_KEY"], ANTHROPIC_MODEL=META_MODEL,
                   ANTHROPIC_DEFAULT_HAIKU_MODEL=META_MODEL, ANTHROPIC_SMALL_FAST_MODEL=META_MODEL,
                   CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_TELEMETRY="1")
    t0 = time.time()
    code = None
    try:
        cmd = command(agent, task, state, px.url, template)
        cmd[0] = shutil.which(cmd[0]) or cmd[0]
        with open(os.path.join(state, "stdout.txt"), "w+", encoding="utf8", errors="replace") as out, \
                open(os.path.join(state, "stderr.txt"), "w+", encoding="utf8", errors="replace") as errf:
            p = subprocess.Popen(cmd, cwd=repo, env=env, stdout=out, stderr=errf, stdin=subprocess.DEVNULL,
                                 text=True, encoding="utf8", errors="replace")
            while p.poll() is None:                 # watch the clock and, for metered APIs, the spend
                time.sleep(2)
                if time.time() - t0 > timeout:
                    code = "timeout"
                elif max_usd and agent in METERED and usd(totals(rw.usage_path)) > max_usd:
                    code = f"spend cap ${max_usd}"
                if code:
                    subprocess.call(["taskkill", "/F", "/T", "/PID", str(p.pid)], stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL) if os.name == "nt" else p.kill()
                    p.wait()
                    break
            code = code or p.returncode
            out.seek(0), errf.seek(0)
            answer, err = out.read(), errf.read()
    finally:
        px.stop()
    tot = totals(rw.usage_path)
    return dict(arm=arm, exit=code, seconds=round(time.time() - t0, 1), answer=answer.strip(), stderr=err[-2000:],
                tools=dict(collections.Counter(rw.tool_calls.values()).most_common()), **tot)


def cost(r):
    """Input-side tokens in the provider's own units: Anthropic cache-weighted, else prompt minus cache discount."""
    if "cache_read_input_tokens" in r or "cache_creation_input_tokens" in r:
        return r.get("input_tokens", 0) + 1.25 * r.get("cache_creation_input_tokens", 0) + \
            0.1 * r.get("cache_read_input_tokens", 0)
    if "promptTokenCount" in r:     # Gemini implicit caching bills cached tokens at a 75% discount
        return r["promptTokenCount"] - 0.75 * r.get("cachedContentTokenCount", 0)
    if "prompt_tokens" in r:
        return r["prompt_tokens"] - 0.5 * r.get("prompt_tokens_details.cached_tokens", 0)
    return r.get("input_tokens", 0) - 0.9 * r.get("input_tokens_details.cached_tokens", 0)


def output_tokens(r):
    return (r.get("output_tokens", 0) + r.get("completion_tokens", 0) + r.get("candidatesTokenCount", 0)
            + r.get("thoughtsTokenCount", 0))


def table(runs, agent, task_name):
    rows = ["| run | arm | exit | requests | rewrites | saved at read | input (weighted) | output | seconds | top tools |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(runs, 1):
        tools = ", ".join(f"{k} {v}" for k, v in list(r["tools"].items())[:4])
        rows.append(f"| {i} | {r['arm']} | {r['exit']} | {r['requests']} | {r.get('rewrites', 0)} | "
                    f"{r.get('orig_tokens', 0) - r.get('new_tokens', 0):,} | {cost(r):,.0f} | {output_tokens(r):,} | "
                    f"{r['seconds']} | {tools} |")
    rows += ["", "| arm | runs | mean requests | mean input (weighted) | mean output | mean seconds |",
             "|---|---|---|---|---|---|"]
    by = collections.defaultdict(list)
    for r in runs:
        by[r["arm"]].append(r)
    for arm, rs in by.items():
        n = len(rs)
        rows.append(f"| {arm} | {n} | {sum(r['requests'] for r in rs) / n:.1f} | "
                    f"{sum(cost(r) for r in rs) / n:,.0f} | {sum(output_tokens(r) for r in rs) / n:,.0f} | "
                    f"{sum(r['seconds'] for r in rs) / n:.0f} |")
    head = (f"# Proxy A/B: {agent}, task `{task_name}`\n\n{time.strftime('%Y-%m-%d %H:%M')}; arms alternate order per "
            f"trial. Input (weighted) is in the provider's units (see bench/proxy_ab.py `cost`). Check the saved "
            f"answers before reading token savings as a win.\n\n")
    return head + "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--agent", required=True, help="agy | claude | codex, or any name with --cmd")
    ap.add_argument("--cmd", help='custom agent command line, e.g. "mycli run {task}" ({task} = one argument)')
    ap.add_argument("--task", choices=sorted(TASKS), default="audit")
    ap.add_argument("--task-file")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--upstream")
    ap.add_argument("--arms", default="cortex,passthrough")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per run")
    ap.add_argument("--max-usd", type=float, default=3.0, help="codex-meta: kill a run past this spend")
    ap.add_argument("--max-total-usd", type=float, default=10.0, help="codex-meta: stop the A/B past this spend")
    a = ap.parse_args()
    task_name = os.path.basename(a.task_file) if a.task_file else a.task
    task = open(a.task_file, encoding="utf8").read().strip() if a.task_file else TASKS[a.task]
    if a.agent not in UPSTREAM and not (a.cmd and a.upstream):
        ap.error("unknown agent: give --cmd and --upstream")
    upstream = a.upstream or UPSTREAM[a.agent]
    arms = a.arms.split(",")
    out = os.path.join(HERE, "results", "traces", "proxy_ab", f"{a.agent}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(os.path.join(out, "answers"), exist_ok=True)
    runs = []
    for t in range(a.trials):
        for arm in (arms if t % 2 == 0 else arms[::-1]):
            print(f"trial {t + 1}/{a.trials}, {arm}: running {a.agent} ...", file=sys.stderr, flush=True)
            r = run_once(a.agent, arm, task, a.repo, upstream, os.path.join(out, "state", f"{t + 1}-{arm}"),
                         a.timeout, a.cmd, a.max_usd)
            r["trial"] = t + 1
            if a.agent in METERED:
                r["usd"] = round(usd(r), 4)
            runs.append(r)
            with open(os.path.join(out, "answers", f"{t + 1}-{arm}.md"), "w", encoding="utf8") as f:
                f.write(r["answer"] + "\n")
            print(f"  exit {r['exit']}, {r['requests']} requests, {r.get('rewrites', 0)} rewrites, "
                  f"input {cost(r):,.0f}, {r['seconds']}s" + (f", ${r['usd']:.3f}" if "usd" in r else ""),
                  file=sys.stderr, flush=True)
            if r["exit"] != 0 and r["stderr"]:
                print("  stderr: " + r["stderr"][-400:], file=sys.stderr)
            spent = sum(x.get("usd", 0) for x in runs)
            if spent > a.max_total_usd:
                print(f"stopping: ${spent:.2f} spent, over --max-total-usd {a.max_total_usd}", file=sys.stderr)
                break
        else:
            continue
        break
    with open(os.path.join(out, "runs.json"), "w", encoding="utf8") as f:
        json.dump(dict(agent=a.agent, task=task, upstream=upstream, runs=runs), f, indent=1)
    md = table(runs, a.agent, task_name)
    with open(os.path.join(HERE, "results", f"proxy_ab_{a.agent}.md"), "w", encoding="utf8") as f:
        f.write(md)
    print(md)
    print(f"answers and per-run state: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
