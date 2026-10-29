# Spiking file manager: which file a real model wants next

Each model's real file operations in time order; before each one the manager predicts from the model's words, then learns from the truth. Held out: Llama by question (its runs repeat 20 questions), Claude by session. eta chosen on the tuning halves: 0.0 (mean hit@3 by eta: 0.0: 0.849, 0.02: 0.849, 0.05: 0.849, 0.1: 0.849, 0.2: 0.849).

| model | streams | file ops | predictable | method | hit@1 | hit@3 | new file hits@3 (n) |
|---|---|---|---|---|---|---|---|
| Claude (Claude Code sessions) | 65 | 3,083 | 67% (held out 1,380) | spiking manager (gated) | 54.8% | 70.1% | 4.2% (142) |
| Claude (Claude Code sessions) | 65 | 3,083 | 67% (held out 1,380) | innate population (learning off) | 54.8% | 70.1% | 4.2% (142) |
| Claude (Claude Code sessions) | 65 | 3,083 | 67% (held out 1,380) | plastic population alone | 54.8% | 70.1% | 4.2% (142) |
| Claude (Claude Code sessions) | 65 | 3,083 | 67% (held out 1,380) | mention, then recency | 52.4% | 68.9% | 4.2% (142) |
| Claude (Claude Code sessions) | 65 | 3,083 | 67% (held out 1,380) | recency | 63.3% | 72.4% | 0.0% (142) |
| Llama 3.1 8B (laptop sessions) | 19 | 356 | 100% (held out 166) | spiking manager (gated) | 48.8% | 65.7% | 55.3% (85) |
| Llama 3.1 8B (laptop sessions) | 19 | 356 | 100% (held out 166) | innate population (learning off) | 48.8% | 65.7% | 55.3% (85) |
| Llama 3.1 8B (laptop sessions) | 19 | 356 | 100% (held out 166) | plastic population alone | 48.8% | 65.7% | 55.3% (85) |
| Llama 3.1 8B (laptop sessions) | 19 | 356 | 100% (held out 166) | mention, then recency | 65.7% | 65.7% | 55.3% (85) |
| Llama 3.1 8B (laptop sessions) | 19 | 356 | 100% (held out 166) | recency | 37.3% | 37.3% | 0.0% (85) |

Paired differences (bootstrap over predictions, 95% interval) and the network:

- Claude (Claude Code sessions): manager vs learning off, hit@3 on the held-out half: +0.000 [+0.000, +0.000]
- Claude (Claude Code sessions): manager vs plastic alone, hit@3 on the held-out half: +0.000 [+0.000, +0.000]
- Claude (Claude Code sessions): manager vs mention, then recency, hit@3 on the held-out half: +0.012 [+0.001, +0.022]
- Claude (Claude Code sessions): manager vs recency, hit@3 on the held-out half: -0.023 [-0.034, -0.012]
- Claude (Claude Code sessions): plastic weights at the end {'touch': 1.0, 'path': 1.0, 'symbol': 0.5, 'habit': 0.2} (innate {'touch': 1.0, 'path': 1.0, 'symbol': 0.5, 'habit': 0.2}); driving at the end: innate; 7.6 spikes per prediction, 99% of neurons silent
- Llama 3.1 8B (laptop sessions): manager vs learning off, hit@3 on the held-out half: +0.000 [+0.000, +0.000]
- Llama 3.1 8B (laptop sessions): manager vs plastic alone, hit@3 on the held-out half: +0.000 [+0.000, +0.000]
- Llama 3.1 8B (laptop sessions): manager vs mention, then recency, hit@3 on the held-out half: +0.000 [+0.000, +0.000]
- Llama 3.1 8B (laptop sessions): manager vs recency, hit@3 on the held-out half: +0.283 [+0.217, +0.350]
- Llama 3.1 8B (laptop sessions): plastic weights at the end {'touch': 1.0, 'path': 1.0, 'symbol': 0.5, 'habit': 0.2} (innate {'touch': 1.0, 'path': 1.0, 'symbol': 0.5, 'habit': 0.2}); driving at the end: innate; 17.9 spikes per prediction, 89% of neurons silent

(176s)
