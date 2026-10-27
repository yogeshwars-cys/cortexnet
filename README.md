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
| **Cortex-SubNN coprocessor** (below) | designed, not built yet |
| Live A/B on a strong open model (Meta Muse Glimmer 30B): tokens and task success | harness built, not run yet |

The numbers below come from the reader and the v1 include model. The coprocessor has no numbers yet.

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

## The coprocessor: Cortex-SubNN

The v1 include model only sees features of the file. The coprocessor also sees what the LLM is about to do:

1. **Snapshot.** Before each tool call, take the model's latest reasoning plus the tool arguments and the last symbol
   it looked at.
2. **Bottleneck.** A two-layer int8 MLP compresses that into a short intent vector (k = 32). The same network embeds
   every symbol in the workspace. Trained contrastively (InfoNCE plus L1 sparsity) on which symbol the agent needed
   next, counting reads for reference as well as edits.
3. **Round-trip scoring.** A symbol is scored by walking from the intent to the symbol and back. Generic helpers that
   connect to everything leak probability and score low, which fixes the failure of the plain symbol graph (AUC 0.50,
   below).
4. **Cost gate.** A body is included only if the expected saved round trip outweighs its token cost and the score clears
   a floor. Otherwise the agent gets the outline alone.

Open question: whether the coprocessor reads the model's reasoning text or its hidden states. Text keeps it model
agnostic and cheap; hidden states need the host model on a GPU but give a richer signal.

## What did not help

Measured, see `bench/results/`: a decayed session trail of identifiers and a cross-file symbol graph (calls,
called-by, body overlap) add no held-out gain to the v1 include model; the graph feature alone is at chance (AUC 0.50).
A menu of connected symbols in neighbouring files was offered 25 times in the replay and never needed, so it is off by
default (`Reader(menu=True)` to try it).

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
| `bench/replay_edits.py`, `bench/replay_sessions.py` | offline replay over local session logs (aggregates only) |
| `bench/*_signal.py`, `bench/*_snapshot.py`, `bench/commit_symbols.py` | feature and label studies |
| `bench/live_ab.py` | live A/B: same agent and tasks, whole-file reads vs CortexNET; local transformers or any OpenAI-compatible server |
| `bench/results/` | result tables |
| `tests/` | `python -m unittest discover -s tests` |

## Data

Benchmarks read your own local agent logs. Those contain file contents and prompts: they are never copied into this
repo, and the scripts only write aggregate numbers.
