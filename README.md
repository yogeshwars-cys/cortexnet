# CortexNET

A small neural coprocessor that runs next to a coding agent's LLM and decides what code goes into its context.
Instead of pulling whole files into the context window, the agent gets an outline (symbols, line spans, signatures)
plus only the bodies the coprocessor expects the next step to need. When the agent asks for lines it was not shown,
that request is a miss, and the misses are what the coprocessor learns from.

## Status

| Part | State |
|---|---|
| Outline-first reader (outlines, ranges, sub-block splitting, miss accounting) | built, tested, measured on replay |
| Include model v1: online logistic over symbol features, trained on misses | built, measured on replay |
| Delivery to agents over MCP (`cortex_read`, `cortex_range`, `cortex_outline`) | built, in daily use |
| **Cortex-SubNN coprocessor** (below) | built, cross-validated on 854 logged reads |
| 64-byte symbol codes (`cortexnet/codec.py`) | built, no measurable ranking loss |
| Live A/B, Meta Muse Spark 1.3 (Meta Model API) as the agent | run, 14 tasks |
| Live A/B, Meta Muse Glimmer 30B (self-hosted) as the agent, on SWE-bench Lite | run, 39 paired tasks |

## Why

Measured on 144 real agent sessions (20k file accesses):

- About half of all tool-output tokens are file contents (`Read` 19%, `cat`/`sed`/`head` in shell ~29%).
- Smarter whole-file caching barely helps: even the offline optimum (Belady) re-fetches only 3-11% fewer tokens than
  LRU, because 30-60% of the tokens are first reads that no eviction policy avoids.
- What the agent edits after reading a file is small: a median 3.6% of the file (16-line blocks).
- A cheap lexical scanner cannot find that part from the prompt (no better than random blocks), so the locator has to
  read what the model is actually thinking.

Replaying 99 logged edits through the reader: outline + the right symbol costs 19% of the whole-file tokens (median per
edit 12%). Replaying 62 whole sessions in order (small files whole, partial reads, re-reads, one range call per miss):
62% of the read tokens, 56% with big symbols split into sub-blocks. Both are upper bounds: they assume the right
symbol gets picked. Picking it is the coprocessor's job, and the live A/B is the real test.

## Results

**Glimmer 30B as the agent** (`bench/swe_agent.py`, `bench/modal_glimmer.py::ab`, report `bench/results/swe_ab.md`).
SWE-bench Lite, one H100 per arm, the same tasks with whole-file reads, with CortexNET, and with CortexNET plus fading
(a read drops to its outline once the agent moves on). Paired over the 39 tasks every arm finished (95% CIs from a
paired bootstrap over tasks):

| | whole file | CortexNET | CortexNET + fade |
|---|---|---|---|
| prefill tokens (the model re-reads its context every turn) | 195k | 0.58x [0.47, 0.71] | 0.56x [0.45, 0.72] |
| peak context | 20.4k | 0.47x [0.40, 0.56] | 0.42x [0.36, 0.50] |
| peak KV cache (Glimmer's layout: 13 full-attention layers grow, 39 sliding-window layers cap at 2k) | 337 MB | 0.59x [0.53, 0.67] | 0.56x [0.49, 0.63] |
| file tokens put into context | 636k | 33% | 31% |
| turns | 17 | 1.16x [1.02, 1.32] | 1.19x [1.03, 1.38] |
| generation time | 165 s | 0.98x | 1.07x |
| submitted the gold file / gold symbol | 82% / 41% | 85% / 46% | 82% / 41% |

Task success is unchanged within noise (gold symbol, CortexNET vs whole: 4 tasks only CortexNET solved, 2 only whole
solved, exact McNemar p = 0.69; fade vs whole 1 and 1). One whole-file episode hit the 60k-token context limit.

One whole-file episode ran the 80 GB GPU out of memory; the same task fit on a rerun, and both CortexNET arms solved
it, fade at 0.43x the prefill and 0.54x the peak KV (`bench/results/swe_incident_xarray_4094.md`). Whole-file reads
also show what the agent used: from the
attention of a full-attention layer (65 whole-file reads), 80% of the attention on code bodies lands on 56% of the
code shown, and ranking symbols by that attention finds the needed ones at MRR 0.85 (random 0.21, 40 labelled reads).
That gives labels for reads the agent never comes back to.

**Muse Spark 1.3 as the agent** (Meta Model API, `bench/live_ab.py --meta`, 14 code questions): both arms 14/14
correct; CortexNET used 52% of the input tokens and 43% of the cost. The gain is all on large files.

**Coprocessor ranking** (854 read-time decisions from 91 logged sessions, 5-fold cross-validation grouped by session,
`bench/subnn_train.py`): MRR 0.416 for the full model vs 0.348 for frozen MiniLM zero-shot (+0.068, 95% CI
[+0.047, +0.090]), 0.309 for the v1 include model and 0.280 for random order. With symbol vectors stored as 64-byte
codes: 0.414.

**Where it pays:** the cost-aware decision rule saves most for agents with small or expensive contexts, such as a
self-hosted model where every context token costs KV memory and prefill. At the context sizes of a hosted agent like
Claude Code it falls back to whole files about 92% of the time.

## The coprocessor: Cortex-SubNN

The v1 include model only sees features of the file. The coprocessor also sees what the LLM is about to do:

1. **Views.** The task prompt, the model's latest reasoning before the read, and the trail of symbols it looked at
   last, fused by a learned gate (when there is no reasoning, the gate moves weight to the prompt and trail).
2. **Late interaction.** Each symbol is split into up to 4 sub-spans; a symbol's score is its best sub-span match,
   through a low-rank bilinear head on a frozen MiniLM encoder, with an int8 bottleneck.
3. **Training.** Cost-weighted contrastive loss on which symbol the agent needed next (edits, line-range requests,
   `sed -n`, git history), hard negatives from the same file.
4. **Decision.** Calibrated probabilities pick the up-front bodies, or the whole file when that is cheaper in
   expectation (`cortexnet/policy.py`). Otherwise the agent gets the outline alone.

## What did not help

Measured, see `bench/results/`:

- A decayed session trail of identifiers and a cross-file symbol graph add no held-out gain to the v1 include model;
  the graph feature alone is at chance (AUC 0.50). A menu of connected symbols in neighbouring files was offered 25
  times in the replay and never needed, so it is off by default (`Reader(menu=True)` to try it).
- Glimmer 30B as the frozen text encoder instead of MiniLM: ties it (0.396 vs 0.416) at ~1,000x the size.
- Switching between the two encoders online (Hedge) or stacking them with a per-read gate: +0.012 to +0.014 MRR,
  not significant (`subnn_hedge.md`, `subnn_fuse.md`).
- Glimmer's raw hidden state at the read decision as the query, on 55 live cases: 0.49-0.53 vs 0.73 for MiniLM on the
  text (a 6,656-d projection overfits). Reading it through Glimmer's own vocabulary (logit lens, symbol names scored
  by their word pieces) is at chance: MRR 0.29-0.32 vs 0.30 random over 91 reads.

## Use

```
python -m cortexnet.mcp_server                                # stdio MCP server, no dependencies beyond numpy
claude mcp add cortexnet -- python -m cortexnet.mcp_server    # e.g. register with Claude Code
```

Tools: `cortex_read(path, query?)`, `cortex_range(path, start, end)`, `cortex_outline(path)`, `cortex_stats()`.
Big symbols are split into sub-blocks at 600 tokens (`CORTEXNET_SPLIT`, 0 turns it off). State (include model, event
log) goes to `$CORTEXNET_STATE` or `./.cortexnet/`.

## Layout

| Path | Contents |
|---|---|
| `cortexnet/outline.py` | file sketches: Python via `ast`, other languages via line patterns, sections as fallback |
| `cortexnet/learn.py` | v1 include model: online logistic over symbol features, labels from the agent's range requests |
| `cortexnet/reader.py` | outline-first reads, ranges, miss accounting, event log |
| `cortexnet/mcp_server.py` | minimal MCP server (JSON-RPC over stdio) |
| `cortexnet/coproc.py`, `cortexnet/codec.py`, `cortexnet/policy.py` | coprocessor runtime, 64-byte bit-plane symbol codes, cost-aware include rule |
| `bench/subnn_data.py`, `bench/subnn_train.py` | symbol-level cases from session logs and git history; coprocessor training and cross-validation |
| `bench/swe_agent.py`, `bench/modal_glimmer.py`, `bench/swe_ab_report.py` | Glimmer 30B as a coding agent on SWE-bench Lite (Modal), A/B arms, attention and logit-lens probes, report |
| `bench/subnn_hedge.py`, `bench/subnn_fuse.py` | encoder switching and stacking studies |
| `bench/replay_edits.py`, `bench/replay_sessions.py` | offline replay over local session logs (aggregates only) |
| `bench/*_signal.py`, `bench/*_snapshot.py`, `bench/commit_symbols.py` | feature and label studies |
| `bench/live_ab.py` | live A/B: same agent and tasks, whole-file reads vs CortexNET; local transformers or any OpenAI-compatible server |
| `bench/results/` | result tables |
| `tests/` | `python -m unittest discover -s tests` |

## Data

Benchmarks read your own local agent logs. Those contain file contents and prompts: they are never copied into this
repo, and the scripts only write aggregate numbers.
