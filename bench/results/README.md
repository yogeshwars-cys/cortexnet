# Results index

Every report in this folder, what it measures and the script that produced it. Per-run traces (every request, answer
and tool output) go to `traces/`, which is not committed; the reports hold the aggregates. Unless a report says
otherwise, each configuration ran once and 95% intervals come from paired bootstraps over questions or tasks.

## Held-out sessions (machine-written questions, unseen repositories)

| report | what | script |
|---|---|---|
| `heldout_v2.md` | **test set**: Llama 3.1 8B, 320 questions over 8 unseen subtrees, 8k / 16k / 32k, paged vs CortexNET, anchors on and off | `bench/heldout.py run --set v2`, `bench/modal_llamacpp.py` |
| `heldout.md` | development set (v1): the same with the reader before anchored reads; its failures were studied | `bench/heldout.py run` |
| `heldout_v3.md` | Llama 3.2 3B and 1B, sessions up to 60 questions, pre-reads and turn fade | `bench/heldout.py run --set v3` |

Questions: `bench/heldout/questions*.jsonl` (`bench/heldout.py gen` regenerates them from the SWE-bench Lite snapshots).

## Local and hosted sessions (the hand-written 20 questions)

| report | what | script |
|---|---|---|
| `live_ab_llama_3_1_8b_session8k_final.md` | the laptop table: Llama 3.1 8B, 8k, one session, whole vs CortexNET vs + fade | `bench/live_ab.py --session` |
| `live_ab_llama_3_1_8b_session8k*.md` | earlier laptop session runs (fade variants, refractory, rank order) | `bench/live_ab.py --session` |
| `live_ab_llama_3_1_8b_8192.md` | the same questions one at a time | `bench/live_ab.py` |
| `live_ab_muse_spark_1_3_medium_session.md`, `live_ab_meta_models_muse_spark_open_1026_session.md` | Muse Spark 1.3 (Meta Model API) and Muse Spark Open (Baseten), 20-question sessions | `bench/live_ab.py --session --meta` / `--openai` |
| `live_ab_muse_spark_1_3_medium.md` | Muse Spark 1.3, 14 questions one at a time | `bench/live_ab.py --meta` |
| `live_ab_Qwen_Qwen3_1_7B*.md` | first smoke runs on a 1.7B model | `bench/live_ab.py --hf` |
| `debut.md` | summary of the Meta, Llama 3.3 70B and Llama 4 Maverick runs | (summary) |

## SWE-bench Lite agents

| report | what | script |
|---|---|---|
| `swe_ab.md` | Muse Glimmer 30B on H100s: whole vs CortexNET vs per-turn fade, prefill, KV cache, success | `bench/swe_agent.py`, `bench/modal_glimmer.py`, `bench/swe_ab_report.py` |
| `swe_api_baseten_muse_spark_open*.md` | Muse Spark Open: Glimmer limits, relaxed limits, enforced submit | `bench/swe_api.py --baseten` |
| `swe_api_meta_muse-spark-1.3_medium.md` | Muse Spark 1.3 on the Meta Model API | `bench/swe_api.py --meta` |
| `swe_api_openai_llama_3_3_70b_read.md`, `swe_api_openai_llama_4_maverick_v2.md` | Llama 3.3 70B and Llama 4 Maverick on vLLM | `bench/swe_api.py --openai`, `bench/modal_vllm.py` |
| `swe_incident_xarray_4094.md` | the whole-file episode that ran an 80 GB GPU out of memory | (incident record) |

## Engine and proxy

| report | what | script |
|---|---|---|
| `stress_engine.md` | pathological files, 3,000 real files, the MCP server under a 2,000-call burst | `bench/stress_engine.py all` (CI runs `mcp`) |
| `proxy_ab_claude-meta.md`, `proxy_ab_agy.md` | Claude Code and Antigravity through the egress proxy | `bench/proxy_ab.py` |
| `shell_eval.md` | outlining long shell outputs | `bench/shell_eval.py` |
| `files_replay.md`, `working_set.md` | spiking file manager on real file-operation streams; working-set statistics | `bench/files_replay.py`, `bench/working_set.py` |
| `replay_sessions.md`, `replay_edits.md` | neural-dynamics options and edit replay over logged sessions | `bench/replay_sessions.py`, `bench/replay_edits.py` |

## Coprocessor (experimental, off by default; see `docs/coprocessor.md`)

| report | what | script |
|---|---|---|
| `read_eval.md` | gold-line recall of one read on 139 SWE-bench Lite reads | `bench/read_eval.py` |
| `read_eval_*_head.md`, `read_eval_anchors.md` | the same on today's reader: first, control and SWE-trained heads; anchors on and off | `bench/read_eval.py --coproc <head>` |
| `subnn_swe.md`, `subnn_base_remap.md` | coprocessor retrained with SWE-bench issue cases, and its control | `bench/subnn_swe.py`, `bench/subnn_train.py` |
| `subnn*.md` | training, cross-validation, encoders, codec, calibration, attention labels | `bench/subnn_train.py`, `bench/subnn_*.py` |
| `commit_symbols.md`, `*_signal.md`, `*_snapshot.md` | label and feature studies | `bench/commit_symbols.py`, `bench/*_signal.py`, `bench/*_snapshot.py` |
