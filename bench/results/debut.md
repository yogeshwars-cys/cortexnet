# CortexNET on Meta's models and Llama 3.3 70B (2026-10-29)

Two harnesses, three models, every arm the same model with only the read tool changed:

- **Session A/B** (`bench/live_ab.py --session`): 20 code questions about this repo and the Python standard library in
  one conversation per arm, so the context accumulates as in a real session. Workspace pinned to commit e16b82e.
- **SWE-bench Lite agent** (`bench/swe_api.py`): the 40 instances of the Glimmer 30B A/B (`swe_ab.md`), the same tools
  and scoring (the submitted file / symbol overlaps the gold patch; no tests are run), driven through a chat API.
  Arms: whole-file reads and CortexNET outline-first reads. A third arm with the experimental coprocessor ran in every
  configuration; its numbers are in the per-run reports and `docs/coprocessor.md`.

Models: Muse Spark Open (`meta-models/muse-spark-open-1026` on Baseten), Muse Spark 1.3 (Meta Model API, reasoning
effort medium), Llama 3.3 70B Instruct FP8 (vLLM 0.32 on one Modal H200, `bench/modal_vllm.py`). One run per
configuration at temperature 0; 95% intervals from paired bootstraps over questions or instances.

## Session A/B: 20 questions in one conversation

| model | arm | correct | input tokens | uncached input | context at the end | cost |
|---|---|---|---|---|---|---|
| Muse Spark 1.3 (Meta API) | whole-file | 19/20 | 3,255,752 | 1,272,593 | 138,964 | $1.912 |
| | **CortexNET** | **20/20** | 380,533 (12%) | 92,470 | 13,305 | **$0.182 (9.5%)** |
| Muse Spark Open (Baseten) | whole-file | 20/20 | 2,727,167 | 158,591 | 155,657 | credit |
| | **CortexNET** | 20/20 | 447,347 (16%) | 18,419 (12%) | 14,326 | credit |

Reports: `live_ab_muse_spark_1_3_medium_session.md`, `live_ab_meta_models_muse_spark_open_1026_session.md`.
A prefix cache served 61% (Meta) and 94% (Baseten) of the whole-file arm's input; the savings above hold on the
uncached tokens too.

## SWE-bench Lite: 40 instances

| model, limits | arm | gold file | gold symbol | file tokens into context | peak context | input tokens | turns |
|---|---|---|---|---|---|---|---|
| Muse Spark Open, 40 calls / 120k, submit enforced | whole-file | 70% | 30% | 65,698 | 75,511 | 542k | 12 |
| | **CortexNET** | **78%** | **48%** (p = 0.02) | **0.37x** | **0.44x** | 0.78x | 1.61x |
| Llama 3.3 70B, 20 calls / 60k, read before submit | whole-file | 40% | 12% | 7,033 | 10,326 | 16.2k | 4 |
| | **CortexNET** | 42% | 8% (p = 0.50) | **0.19x** | **0.32x** | 0.58x | 1.01x |
| Muse Spark 1.3 (Meta API), 20 calls / 60k | whole-file | 68% | 35% | 27,069 | 33,327 | 231k | 11 |
| | **CortexNET** | 55% | 38% (p = 1.00) | **0.47x** | **0.54x** | 0.58x | 1.09x |

p = exact McNemar on the gold symbol against whole-file reads. Cost on the Meta API: **$0.132 per instance with
CortexNET against $0.217 with whole-file reads (0.61x [0.51, 0.73])**. Reports: `swe_api_*.md` (one per run).

- **Muse Spark Open, enforced submit** is the one significant outcome difference: CortexNET solved 7 instances
  whole-file reads did not, and lost none. Whole-file reads hit the 120k context limit on 7 of 40.
- **Llama 3.3 70B** makes few calls (4 per instance) and no limit bound any arm: 81% fewer file tokens at the same
  turns and the same success.
- **Muse Spark 1.3** ran under the Glimmer A/B's limits (20 calls, 60k): CortexNET's extra turns ran into the call
  cap on 16 instances (whole-file: 9), which is where its lower gold-file rate comes from; the gold symbol rate is
  level. A rerun with the Baseten run's limits and an enforced submit is the open measurement.

## Llama 4 Maverick under protocol v2 (2026-10-30)

Llama 4 Maverick (400B MoE, FP8) on four Modal H200, `bench/swe_api.py --protocol v2`: rules fixed before the run
(whole-file vs CortexNET, 40 calls, 120k context, a tool call on every turn, submit only for a file read in an earlier
turn). Stopped at the Modal credit after 67 of 80 episodes; paired over the 28 instances both arms finished
(`swe_api_openai_llama_4_maverick_v2.md`):

| | whole-file | CortexNET | CortexNET / whole |
|---|---|---|---|
| gold file | 61% | 61% | |
| gold symbol | 21% | **32%** (5 won, 2 lost, p = 0.45) | |
| file tokens into context | 28,704 | 9,006 | **0.31x** [0.21, 0.46] |
| peak context | 25,243 | 9,991 | **0.40x** [0.26, 0.60] |
| input tokens | 35,984 | 26,588 | 0.74x [0.48, 1.09] |
| turns | 4 | 6 | 1.43x [1.18, 1.74] |

The same picture as Muse Spark Open: a third of the file text, less than half the peak context, the right file as
often, the right symbol more often (not significant at 28 instances). Two rules joined protocol v2 during its first
minutes on this model, before any data was kept: Maverick answered in prose without opening a file (so every turn must
be a tool call), and it issued its reads and its submit in one turn, before seeing the reads (so a read counts only
from an earlier turn).

## What the runs taught the harness

- **Limits decide outcomes before the model does.** Under the Glimmer limits (20 calls, 60k) Muse Spark Open ran
  whole-file reads into the context limit (21 of 40) and CortexNET into the call limit (22-24 of 40)
  (`swe_api_baseten_muse_spark_open.md`); whenever it submitted, the file was right (93-100%).
- **Submit protocol.** With the limits relaxed (40 calls, 120k) but no enforcement, Muse Spark Open wrote its fix as
  text and stopped instead of calling `submit` in 21-22 of 40 CortexNET episodes (whole-file: 9), so CortexNET scored
  10% gold symbol against 38% (`swe_api_baseten_muse_spark_open_wide.md`). The final messages held correct patches in
  the gold files. Enforcing the tool call after one reminder (`tool_choice` = submit, all arms alike) is the run in
  the table; the enforcement was added after seeing that result, and both runs are kept.
- **Llama 3.3 70B guesses without reading.** Without a rule it submitted on its first turn in 18 of 23 episodes, from
  the issue text alone (5 of 23 the gold file; `traces/swe_api_openai_llama_3_3_70b_blind_submit.jsonl`), so the reads
  under test never happened. Its run accepts a submit only for a file the agent has read (all arms alike).
- **Bugs found and fixed:** the Meta session harness never sent questions 2-20 (it chained only tool outputs); usage
  fields of chat-completions servers were not read (cached and reasoning tokens showed 0); a malformed tool argument
  or a dropped connection ended a whole run; a forced decode could hang a request for 10 minutes; vLLM's FlashInfer
  sampler needs a CUDA compiler the slim image lacks.

## Spend

Meta Model API $22.24 on the ledger (these runs, the stopped first session attempt, and $0.29 from earlier days); Modal $27.68 of the
workspace's $30 monthly credit ($6.54 for the 70B; $21.14 for Maverick: its 417 GB download, two failed starts before
the loading and CUDA-compiler fixes, and the 41-minute run); Baseten on the hackathon credit.
