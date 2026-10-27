# CortexNET

Outline-first context for coding agents. Instead of pulling whole files into the context window, the agent gets an
outline (symbols, line spans, signatures) and asks for the line ranges it needs. Every range it asks for that was not
shown is a miss, and the misses train a small online model that decides which bodies to show up front next time.

## Why

Measured on 144 real agent sessions (20k file accesses):

- About half of all tool-output tokens are file contents (`Read` 19%, `cat`/`sed`/`head` in shell ~29%).
- Smarter whole-file caching barely helps: even the offline optimum (Belady) re-fetches only 3-11% fewer tokens than
  LRU, because 30-60% of the tokens are first reads that no eviction policy avoids.
- What the agent edits after reading a file is small: a median 3.6% of the file (16-line blocks).
- A cheap lexical scanner cannot find that part from the prompt (no better than random blocks), so the locator has to
  be the model itself.

Replaying 99 logged edits through the reader: outline + the right symbol costs 19% of the whole-file tokens (median per
edit 12%). Replaying 62 whole sessions in order (small files whole, partial reads, re-reads, one range call per miss):
62% of the read tokens, 56% with big symbols split into sub-blocks (`CORTEXNET_SPLIT`, default 600 tokens in the
MCP server). Both are upper bounds: they assume the model picks the right symbol from the outline. The live
benchmark (same tasks, with and without CortexNET, tokens and task success) is the real test.

What did not help (measured, `bench/results/`): a decayed session trail of identifiers and a cross-file symbol graph
(calls, called-by, body overlap) add no held-out gain to the include model; the graph feature alone is at chance
(AUC 0.50). A menu of connected symbols in neighbouring files was offered 25 times in the replay and never needed, so
it is off by default (`Reader(menu=True)` to try it).

## Use

```
python -m cortexnet.mcp_server                   # stdio MCP server, no dependencies beyond numpy
claude mcp add cortexnet -- python -m cortexnet.mcp_server   # register with Claude Code
```

Tools: `cortex_read(path, query?)`, `cortex_range(path, start, end)`, `cortex_outline(path)`, `cortex_stats()`.
State (include model, event log) goes to `$CORTEXNET_STATE` or `./.cortexnet/`.

## Layout

| Path | Contents |
|---|---|
| `cortexnet/outline.py` | file sketches: Python via `ast`, other languages via line patterns, sections as fallback |
| `cortexnet/learn.py` | include model: online logistic over symbol features, labels from the agent's range requests |
| `cortexnet/reader.py` | outline-first reads, ranges, miss accounting, event log |
| `cortexnet/mcp_server.py` | minimal MCP server (JSON-RPC over stdio) |
| `bench/replay_edits.py` | offline replay over local session logs (prints aggregates only) |
| `tests/` | `python -m unittest discover -s tests` |

## Data

Benchmarks read your own local agent logs. Those contain file contents and prompts: they are never copied into this
repo, and the scripts only write aggregate numbers.
