# Incident: whole-file arm out of GPU memory on `pydata__xarray-4094` (2026-10-27)

Part of the Glimmer-as-agent A/B (`bench/modal_glimmer.py::ab`, report `swe_ab.md`). Public task (SWE-bench Lite);
all numbers below are copied from the saved episode records (`/data/ab/<arm>/pydata__xarray-4094.json` on the
`cortexnet-subnn` Modal volume).

## What happened

1. First launch, all three arms in one Modal app (`ap-kP6vWe6qHSiO1vO4PmClNe`, one H100 80 GB per arm). The
   whole-file arm finished 18 tasks; on task 19, `pydata__xarray-4094`, `model.generate` failed in an MLP layer:

   ```
   File ".../transformers/models/muse_glimmer/modeling_muse_glimmer.py", line 154, in forward
       down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
   torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.15 GiB. GPU 0 has a total capacity of 79.17 GiB
   of which 1.90 GiB is free. Process 1 has 77.26 GiB memory in use. Of the allocated memory 68.39 GiB is allocated
   by PyTorch, and 5.97 GiB is reserved by PyTorch but unallocated.
   ```

   The exception ended that arm's run and stopped the whole app, so the CortexNET and fade arms were cut off at 25
   and 27 tasks. The context length at the moment of the failure was not logged.
2. Fix: `gen` now catches `torch.OutOfMemoryError` and ends the episode as a recorded failure (`oom: true`) instead
   of crashing; arms run as separate apps (`ab_arm`) so one arm cannot stop another.
3. The resumed whole-file arm (`ap-iL8NRqKHxlWdvlqvza2Y5f`) ran the same task again and **did not** run out of
   memory. Generation differs between runs and the earlier container had been through 18 episodes, so the crash
   depended on that run; it does not show that whole-file reads cannot fit an 80 GB GPU.

## The same task in every arm

| | whole file (rerun) | CortexNET | CortexNET + fade |
|---|---|---|---|
| submitted file / symbol in the gold patch | yes / yes | yes / yes | yes / yes |
| prefill tokens (sum over turns) | 557,959 | 379,364 (0.68x) | 240,757 (0.43x) |
| peak context tokens | 31,605 | 29,199 | 14,390 |
| peak KV cache | 479 MB | 449 MB | 261 MB (0.54x) |
| turns / tool calls | 24 / 12 | 22 / 11 | 25 / 14 |
| file tokens from read_file + read_lines | 25,309 | 23,214 | 9,975 |
| wall seconds | 411 | 413 | 273 |

On this task CortexNET's cost rule returned the whole of `xarray/core/dataarray.py` (a ~22k-token file it expected
to need), so the plain CortexNET arm saved compute (0.68x prefill) but not much peak memory. Fading saved both: once
the agent moved on, the file dropped to its outline (0.43x prefill, 0.54x peak KV), and the task was still solved.
