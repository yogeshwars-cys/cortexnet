# CortexNET

Lets a small local model work through a long coding session on a laptop GPU. A coding agent normally pulls whole files
into its context; on a 6 GB card with an 8k-token window that fails on the first big file. CortexNET sits between the
agent and the code:

- **Outline-first reads.** A read returns the file's outline (symbols, line spans, signatures) plus only the bodies the
  next step is likely to need. Anything else is one `read_lines` away, and every such request is logged as a miss.
- **A coprocessor that picks the bodies** (Cortex-SubNN). A small network on a frozen MiniLM encoder reads the task, the
  agent's own reasoning before the read and the symbols it touched last, and ranks the file's symbols. It adds to a
  feature-based include model; it never takes away a body that model would show.
- **A symbol graph and a connected-symbol menu.** Symbols the agent needs together, or that are semantically close,
  are offered from other files as one-line menu items it can pull with `cortex_approve`.
- **Fade on a watermark.** The conversation stays append-only (so a prefix cache keeps working) until it reaches 85%
  of the context, then one batch shrinks stale reads, oldest first, to 50%: superseded copies of a file become a
  pointer, stale reads become their outline or a pointer.

## Results

### Llama 3.1 8B on a 6 GB laptop GPU

RTX 3050 Laptop (6 GB), Llama 3.1 8B Instruct Q4_K_M in llama.cpp (`-ngl 99 -c 8192`), 20 code questions about this
repo and the Python standard library, asked **in one conversation per arm** so the context accumulates as in a real
session (`bench/live_ab.py --session`, reports `bench/results/live_ab_llama_3_1_8b_session8k_*.md`):

| arm | correct | first context overflow | input tokens | tool calls | tool output | wall |
|---|---|---|---|---|---|---|
| whole-file reads | 0/20 | question 1 (`reader.py` whole = 15.8k tokens) | - | - | - | - |
| CortexNET reads | 6/20 | question 9 (dead after) | - | - | - | - |
| CortexNET + fade | 13/20 | never | 229k | 30 | 63k | 204 s |
| CortexNET + coprocessor + fade | 13/20 | never | 230k | 26 | 49k | 247 s |
| **full CortexNET** (+ graph and menu) + fade | **13/20** | never | **192k** | **20** | **32k** | **199 s** |

- Fade is what keeps a session alive at 8k: every fade arm answered all 20 questions without an overflow, at a median
  5.2k and at most 6.8k tokens of context.
- The full system is the cheapest at equal accuracy: against CortexNET + fade, 16% fewer input tokens, 33% fewer tool
  calls, 49% less tool output. With the right bodies shown up front, the model re-reads less.
- Accuracy between the fade arms is within run-to-run noise: llama.cpp is not bit-reproducible here, and the same arm
  moved by up to 4 answers between runs at temperature 0 (one earlier run had coprocessor + fade 16/20 against 12/20;
  the next run tied at 13). The drop in tool calls with the coprocessor repeated in both runs.
- How it runs: 5.84 of 6.14 GB VRAM (about 4.4 GB weights, 1.0 GB KV cache reserved for the 8k window, 0.4 GB
  compute), all layers on the GPU, prefill 800-870 tokens/s, generation ~23 tokens/s, median 11 s per question.

One question at a time (`live_ab_llama_3_1_8b_8192.md`, before the coprocessor fixes below): whole-file 3/20 (8
overflows), CortexNET 7/20 (none), with 31% of the input tokens.

### Glimmer 30B on SWE-bench Lite

`bench/swe_agent.py`, `bench/modal_glimmer.py::ab`, report `bench/results/swe_ab.md`. One H100 per arm, the same tasks
with whole-file reads, with CortexNET reads (v1 include model, no coprocessor), and with an older per-turn fade.
Paired over the 39 tasks every arm finished (95% CIs from a paired bootstrap over tasks):

| | whole file | CortexNET | CortexNET + per-turn fade |
|---|---|---|---|
| prefill tokens, no prefix cache (the harness re-read the context every turn) | 195k | 0.58x [0.47, 0.71] | 0.56x [0.45, 0.72] |
| prefill tokens with a prefix cache (each context token once) | 18.7k | **0.42x [0.35, 0.50]** | >= 0.36x (lower bound) |
| peak context | 20.4k | 0.47x [0.40, 0.56] | 0.42x [0.36, 0.50] |
| peak KV cache (13 full-attention layers grow, 39 sliding-window layers cap at 2k) | 337 MB | 0.59x [0.53, 0.67] | 0.56x [0.49, 0.63] |
| file tokens put into context | 636k | 33% | 31% |
| turns | 17 | 1.16x [1.02, 1.32] | 1.19x [1.03, 1.38] |
| submitted the gold file / gold symbol | 82% / 41% | 85% / 46% | 82% / 41% |

Task success is unchanged within noise (gold symbol, CortexNET vs whole: 4 tasks only CortexNET solved, 2 only whole,
exact McNemar p = 0.69). The per-turn fade rewrote mid-history on every fade, which forces a re-prefill on a server
with a prefix cache, so its cached cost is only a lower bound; that is why fade was rebuilt as one batch on a
watermark. One whole-file episode ran the 80 GB GPU out of memory (`swe_incident_xarray_4094.md`).

### Hosted Meta models

- **Muse Spark 1.3** (Meta Model API, `bench/live_ab.py --meta`, 14 code questions, one at a time): both arms 14/14;
  CortexNET used 52% of the input tokens and 43% of the cost.
- **Claude Code on Muse Spark 1.3 through the proxy** (`bench/proxy_ab.py`, an audit task): with Meta's prompt cache a
  whole file is paid once and then read at 0.12x, so outline-first reads that add turns cost more per run (round 1:
  1.25x) or about the same (pooled 0.88x, within noise); the intent-aware mode was the cheapest in both its runs
  (n = 2). Outline-first pays where the context is small or uncached, not on a cached hosted API.

### The coprocessor, offline

854 read-time decisions from 91 logged sessions, 5-fold cross-validation grouped by session (`bench/subnn_train.py`,
`bench/results/subnn_recal.md`): MRR 0.414 against 0.349 for frozen MiniLM zero-shot (+0.065 [+0.043, +0.087]), 0.309
for the v1 include model, 0.280 random; symbol vectors stored as 64-byte codes. On replay, every ranker gives the same
realised token cost under the cost-aware include rule: the ranking gain shows up only once the coprocessor is allowed
to act on its ranking (below), which the live session measures.

### What one read shows of the fix (SWE-bench Lite, offline)

`bench/read_eval.py`, report `bench/results/read_eval.md`: the 139 `read_file` calls of the Glimmer episodes (40
tasks), each replayed once per configuration with the issue text and the agent's reasoning at that moment. Gold-line
recall = the share of the patch's lines in that file that are visible with their line numbers (outline labels do not
count). The coprocessor head never saw SWE-bench data. 95% intervals from a bootstrap over tasks:

| configuration | gold-line recall | vs the v1 include model | tokens per read |
|---|---|---|---|
| v1 include model | 0.312 | | 2,385 |
| v1 + x-ray skeletons | 0.341 | +0.029 [+0.006, +0.060] | 2,694 |
| coprocessor, top-1 pick (the default) | 0.396 | +0.084 [+0.014, +0.170] | 2,498 |
| coprocessor + x-ray | 0.444 | +0.133 [+0.050, +0.232] | 2,806 |
| coprocessor, top-3 picks | **0.497** | **+0.185 [+0.081, +0.307]** | 2,724 |
| coprocessor, top-3 picks + MMR | 0.476 | +0.164 [+0.069, +0.279] | 2,722 |

The coprocessor's ranking is what moves recall, and more of its picks beat cheaper skeletons per token. Redundancy-aware
selection (MMR) costs a little: the symbols a fix needs are often related (a function and its helper).

## Read-time options

| option | what it does | measured |
|---|---|---|
| `Reader(coproc_top=1, coproc_max=400)` | the coprocessor's top symbols go up front when small | top-1 +0.084, top-3 +0.185 gold-line recall (above) |
| `Reader(xray_top=2)` | skeletons of big ranked symbols: signature, lines with the query's rare words, control-flow spine (`outline.skeleton`) | +0.029 to +0.048 recall for ~300 tokens per read |
| `Reader(mmr_beta=0.5)` | the coprocessor's picks avoid near-duplicates of each other | -0.021 recall: off by default |
| `cortexnet.fade.watermark_fade` | one fade batch near the context limit, cache-friendly | keeps 8k sessions alive (laptop table) |
| proxy `--shell` | outlines long shell outputs of log-like commands (tests, builds, installs); search results, diffs, listings and printed values are never touched (`cortexnet/shellout.py`) | on 3,203 logged long outputs: the safe rule outlines 1% (saves ~1%); outlining every command saves 36% but hides a third of the words the agent used next (`bench/results/shell_eval.md`), so it is off by default |

## The coprocessor: Cortex-SubNN

The v1 include model only sees features of the file. The coprocessor also sees what the LLM is about to do:

1. **Views.** The task prompt, the model's latest reasoning before the read, and the trail of symbols it looked at
   last, fused by a learned gate (when there is no reasoning, the gate moves weight to the prompt and trail).
2. **Late interaction.** Each symbol is split into up to 4 sub-spans; a symbol's score is its best sub-span match,
   through a low-rank bilinear head on a frozen MiniLM encoder, plus the v1 features as a structural term.
3. **Training.** Cost-weighted contrastive loss on which symbol the agent needed next (edits, line-range requests,
   `sed -n`, git history), hard negatives from the same file.
4. **Calibration.** On each symbol's score relative to the best in its file, the file's best score and its size
   (`coproc.calib_features`). The ordering inside a file depends only on the relative score, whose weight must be
   positive; export and load refuse anything else.
5. **Decision, on top of the include model.** A symbol's probability is `1 - (1 - p_include)(1 - p_coproc)`, so either
   can bring a body up front. Its honest probabilities are low (one or two needed symbols among 100+), so its own top
   symbol also goes up front when it is at most 400 tokens (`Reader(coproc_top=1, coproc_max=400)`). On the 20 laptop
   questions this put the answer in a shown body for 9 of 20 reads instead of 6, for 5.5% more read tokens.

## What did not help, and what was wrong

Measured, see `bench/results/` and the log of every run:

- **Two bugs, fixed.** The first exported coprocessor head had an inverted calibration (slope -0.71, fit on raw scores
  pooled across files), so its probabilities reversed its own ranking, and the reader replaced the include model's
  probabilities instead of adding to them. The coprocessor therefore contributed nothing in the first replays. Both
  are fixed and covered by tests.
- **Attention labels** (`bench/subnn_attn.py`, `subnn_attn.md`): training the coprocessor toward Glimmer 30B's own
  layer-28 attention over 57 whole-file reads (KL to a soft target, besides the hard labels) gives no held-out gain:
  sessions MRR -0.010 [-0.024, +0.003], SWE-bench reads +0.008 [-0.018, +0.034]. Attention ranks the needed symbols
  well (MRR 0.85) but does not transfer as a training signal at this size.
- Glimmer 30B as the frozen text encoder instead of MiniLM: ties it (0.396 vs 0.416) at ~1,000x the size.
- Switching between encoders online (Hedge) or stacking them with a per-read gate: +0.012 to +0.014 MRR, not
  significant (`subnn_hedge.md`, `subnn_fuse.md`).
- Glimmer's raw hidden state at the read decision as the query: 0.49-0.53 vs 0.73 for MiniLM on the text; reading it
  through Glimmer's own vocabulary (logit lens) is at chance (MRR 0.29-0.32 vs 0.30 random over 91 reads).
- On replay, the trail and the symbol graph added no held-out gain to the v1 model, and the connected-symbol menu was
  never needed in 25 offers. Live, the full arm (coprocessor + graph + menu) is the cheapest per session; the replay
  cannot measure the turns a menu saves.

## Use

```
python -m cortexnet.mcp_server                                # stdio MCP server, no dependencies beyond numpy
claude mcp add cortexnet -- python -m cortexnet.mcp_server    # e.g. register with Claude Code
```

Tools: `cortex_read(path, query?)`, `cortex_range(path, start, end)`, `cortex_outline(path)`, `cortex_approve(ids)`,
`cortex_stats()`. Big symbols are split into sub-blocks at 600 tokens (`CORTEXNET_SPLIT`, 0 turns it off). Opt-in:
`CORTEXNET_COPROC=<head.npz>` (needs torch + transformers for MiniLM), `CORTEXNET_GRAPH=1`, `CORTEXNET_MENU=1`,
`CORTEXNET_POLICY=rule_ab`; the first three together are the full system measured above.
State goes to `$CORTEXNET_STATE` or `./.cortexnet/`. Fade for your own agent loop: `cortexnet.fade.watermark_fade`.

### Local model on a laptop

```
llama-server -m Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -ngl 99 -c 8192 -np 1 --jinja --cache-ram 0 --port 8080
python -I bench/live_ab.py --openai http://127.0.0.1:8080/v1 --model llama-3.1-8b --count-errors --hint \
    --session --ctx 8192 --arms baseline,cortex+fade,full+fade --coproc <head.npz>
```

llama-server reads `LLAMA_API_KEY` from the environment as its own API key; unset it for the server if that variable
holds a key for something else. `--cache-ram 0` turns off the 8 GiB host-RAM prompt cache of recent builds (llama-server
still reuses the slot's prefix). Arms: `baseline`, `cortex`, `coproc`, `coproc_ab`, `full`, each optionally `+fade`.

### Egress proxy (any agent, no tool changes)

```
python -m cortexnet.proxy run --upstream https://api.anthropic.com -- claude     # Claude Code
python -m cortexnet.proxy run --upstream https://api.openai.com -- codex         # OPENAI_BASE_URL agents
python -m cortexnet.proxy serve --upstream <base> --port 8787                    # long-running; point the agent here
python -m cortexnet.proxy stats                                                  # usage and read savings so far
```

A local reverse proxy on the agent's model-API egress (`run` sets `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL` for the
child). When the newest turn holds a whole-file read of a large source file (a read tool with a path, `cat FILE`, or
PowerShell `Get-Content`), the proxy swaps the result for CortexNET's outline-first view before it leaves the machine.
Rewrites are memoized by tool-call id and replayed byte-for-byte, so the provider's prefix cache keeps matching; nothing
older than the newest turn, no assistant message and no response is changed. A second whole read of a file gets the
whole file. `--aware` reads with the agent's own intent (its task and the reasoning around each call) and tells it how
reads work. Anthropic Messages, OpenAI Chat Completions and OpenAI Responses. `--passthrough` logs usage without
rewriting, for A/B baselines. Logs usage numbers only, never request bodies.

## Layout

| Path | Contents |
|---|---|
| `cortexnet/outline.py` | file sketches: Python via `ast`, other languages via line patterns, sections as fallback |
| `cortexnet/learn.py` | v1 include model: online logistic over symbol features, labels from the agent's range requests |
| `cortexnet/reader.py` | outline-first reads, ranges, connected-symbol menu, miss accounting, event log |
| `cortexnet/coproc.py`, `cortexnet/codec.py`, `cortexnet/policy.py` | coprocessor runtime and calibration, 64-byte bit-plane symbol codes, cost-aware include rule |
| `cortexnet/graph.py` | co-need graph over the coprocessor's symbol cells |
| `cortexnet/fade.py` | fade on a watermark for an agent's message history |
| `cortexnet/shellout.py` | outlines of long shell outputs (opt-in in the proxy) |
| `cortexnet/mcp_server.py` | minimal MCP server (JSON-RPC over stdio) |
| `cortexnet/proxy.py` | egress proxy: outline-first rewriting of fresh file reads in agent API traffic, usage log |
| `bench/live_ab.py` | live A/B: same agent and questions, any set of arms, one question at a time or `--session`; local transformers, any OpenAI-compatible server, or the Meta Model API |
| `bench/subnn_data.py`, `bench/subnn_train.py`, `bench/subnn_attn.py` | coprocessor cases, training and cross-validation, attention-label study |
| `bench/swe_agent.py`, `bench/modal_glimmer.py`, `bench/swe_ab_report.py` | Glimmer 30B as a coding agent on SWE-bench Lite (Modal), probes, report |
| `bench/proxy_ab.py` | proxy A/B on a real agent CLI (Claude Code, Codex, Antigravity): rewriting vs passthrough |
| `bench/read_eval.py` | offline read evaluation on SWE-bench Lite read cases (gold-line recall vs tokens) |
| `bench/shell_eval.py` | shell-output outlining on local session logs (aggregates only) |
| `bench/replay_edits.py`, `bench/replay_sessions.py` | offline replay over local session logs (aggregates only) |
| `bench/subnn_hedge.py`, `bench/subnn_fuse.py`, `bench/*_signal.py`, `bench/*_snapshot.py`, `bench/commit_symbols.py` | encoder, feature and label studies |
| `bench/results/` | result tables |
| `tests/` | `python -m pytest -q` (or `python -m unittest discover -s tests`) |

## Data

Benchmarks read your own local agent logs. Those contain file contents and prompts: they are never copied into this
repo, and the scripts only write aggregate numbers. The live A/B and SWE-bench fixtures use only this repo's code, the
Python standard library and public SWE-bench Lite tasks.
