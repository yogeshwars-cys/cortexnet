# Experimental: the Cortex-SubNN coprocessor, symbol graph and connected-symbol menu

Off by default. The coprocessor is a learned ranker that picks which symbol bodies a read shows up front; the symbol
graph and the connected-symbol menu are built on its symbol codes. It raised per-read recall offline, but in no live
run did it beat plain CortexNET reads, so CortexNET's results and defaults do not depend on it. Everything measured
about it is kept here.

## Design

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
   can bring a body up front. Its own top symbol also goes up front when it is at most 400 tokens
   (`Reader(coproc_top=1, coproc_max=400)`).

**Symbol graph and menu.** A co-need graph over the coprocessor's symbol cells (`cortexnet/graph.py`): symbols the
agent needs together, or that are semantically close, are offered from other files as one-line menu items it can pull
with `cortex_approve`.

## Offline

854 read-time decisions from 91 logged sessions, 5-fold cross-validation grouped by session (`bench/subnn_train.py`,
`bench/results/subnn_recal.md`): MRR 0.414 against 0.349 for frozen MiniLM zero-shot (+0.065 [+0.043, +0.087]), 0.309
for the v1 include model, 0.280 random; symbol vectors stored as 64-byte codes.

Gold-line recall per read on the 139 `read_file` calls of the Glimmer 30B SWE-bench Lite episodes (`bench/read_eval.py`,
`bench/results/read_eval.md`; the head never saw SWE-bench data; 95% intervals from a bootstrap over tasks):

| configuration | gold-line recall | vs the v1 include model | tokens per read |
|---|---|---|---|
| v1 include model | 0.312 | | 2,385 |
| coprocessor, top-1 pick | 0.396 | +0.084 [+0.014, +0.170] | 2,498 |
| coprocessor + x-ray | 0.444 | +0.133 [+0.050, +0.232] | 2,806 |
| coprocessor, top-3 picks | **0.497** | **+0.185 [+0.081, +0.307]** | 2,724 |
| coprocessor, top-3 picks + MMR | 0.476 | +0.164 [+0.069, +0.279] | 2,722 |

## Live

| run | plain CortexNET | with the coprocessor |
|---|---|---|
| Llama 3.1 8B laptop session, 20 questions, + fade (`live_ab_llama_3_1_8b_session8k_*.md`) | 13/20, 229k input, 30 calls | 13/20, 230k, 26 calls; with graph and menu 13/20, 192k, 20 calls |
| Muse Spark Open, SWE-bench Lite (`swe_api_baseten_muse_spark_open_wide_forced.md`) | gold symbol 48% | 40% (top-3 picks) |
| Llama 3.3 70B, SWE-bench Lite (`swe_api_openai_llama_3_3_70b_read.md`) | 8% | 10% |
| Muse Spark 1.3, SWE-bench Lite (`swe_api_meta_muse-spark-1.3_medium.md`) | 38% | 32% |
| Muse Spark 1.3 and Muse Spark Open, 20-question sessions (`debut.md` sources) | 20/20, 20/20 | 20/20, 20/20 (graph and menu on); input 1.18x (Meta), 0.95x (Baseten) |

The full system (coprocessor, graph, menu) was the cheapest arm in the laptop session, but on SWE-bench the recall
gain per read did not turn into more solved tasks.

## What did not help

- **Two bugs, fixed.** The first exported head had an inverted calibration (slope -0.71, fit on raw scores pooled
  across files), so its probabilities reversed its own ranking, and the reader replaced the include model's
  probabilities instead of adding to them. Both are fixed and covered by tests.
- **Attention labels** (`bench/subnn_attn.py`, `subnn_attn.md`): training toward Glimmer 30B's layer-28 attention over
  57 whole-file reads gives no held-out gain (sessions MRR -0.010 [-0.024, +0.003], SWE-bench reads +0.008
  [-0.018, +0.034]). Attention ranks the needed symbols well (MRR 0.85) but does not transfer as a training signal.
- Glimmer 30B as the frozen text encoder instead of MiniLM: ties it (0.396 vs 0.416) at ~1,000x the size.
- Switching between encoders online (Hedge) or stacking them with a per-read gate: +0.012 to +0.014 MRR, not
  significant (`subnn_hedge.md`, `subnn_fuse.md`).
- Glimmer's raw hidden state at the read decision as the query: 0.49-0.53 vs 0.73 for MiniLM on the text; read through
  Glimmer's own vocabulary (logit lens) it is at chance (MRR 0.29-0.32 vs 0.30 random over 91 reads).
- On replay, the trail and the symbol graph added no held-out gain, and the connected-symbol menu was never needed in
  25 offers. A winner-take-all menu with a floor cuts menu tokens 79% but costs 1.9% more read tokens.
- MMR over the coprocessor's picks (`Reader(mmr_beta=0.5)`): -0.021 recall; the symbols a fix needs are often related.
- **An earlier next-file predictor, removed.** Its 98% hit@3 on Llama came from 20 questions repeated across 19 runs.
  The spiking file manager (`cortexnet/filesnn.py`) replaced it.

## Use

```
cortex --coproc <head.npz>                                        # CLI: coprocessor + graph + connected menu
CORTEXNET_COPROC=<head.npz> CORTEXNET_GRAPH=1 CORTEXNET_MENU=1 python -m cortexnet.mcp_server
python -I bench/live_ab.py ... --arms baseline,cortex+fade,full+fade --coproc <head.npz>
python -I bench/swe_api.py run --baseten --arms whole,cortex,coproc --coproc <head.npz>
cortex learn                                                      # session reads -> training cases
```

The coprocessor needs torch and transformers (MiniLM). Training: `bench/subnn_data.py`, `bench/subnn_train.py`;
`cortex learn` (`cortexnet/cases.py`) turns the CLI's logged reads and the ranges that followed them into a training
set, which needs a few hundred cases before a retrained head means anything.

| Path | Contents |
|---|---|
| `cortexnet/coproc.py`, `cortexnet/codec.py` | coprocessor runtime and calibration, 64-byte bit-plane symbol codes |
| `cortexnet/graph.py` | co-need graph over the coprocessor's symbol cells |
| `cortexnet/cases.py` | `cortex learn`: session reads to training cases |
| `bench/subnn_data.py`, `bench/subnn_train.py`, `bench/subnn_attn.py` | cases, training and cross-validation, attention-label study |
| `bench/read_eval.py` | offline read evaluation on SWE-bench Lite read cases |
| `bench/subnn_hedge.py`, `bench/subnn_fuse.py`, `bench/*_signal.py`, `bench/*_snapshot.py`, `bench/commit_symbols.py` | encoder, feature and label studies |
