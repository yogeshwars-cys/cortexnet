"""Up-front include policy under a prefix-cached agent.

Costs (input-token equivalents): an extra round trip re-reads the cached history, turn = alpha * L_hist + overhead;
every token put into context costs m = 1 + alpha * H_rem (paid once, then re-read on each of the H_rem later calls).
Scaling both sides by m leaves one number, the effective round-trip cost turn / m, that Rule A/B compares against.
Measured on 91 sessions (bench/subnn_train.py, H_rem median 72): Rule A/B 38% of whole-file tokens vs 65% for a
fixed budget fill.
"""
import numpy as np

ALPHA_CACHE = 0.1       # cache-read price relative to fresh input tokens
TOOL_OVH = 400          # one extra tool step: ~60 output tokens at 5x + ~100 framing tokens
H_REM_PRIOR = 72        # median later API calls that re-read a read's result (91 sessions)


def effective_turn(ctx_tokens, h_rem=H_REM_PRIOR, alpha=ALPHA_CACHE, overhead=TOOL_OVH):
    return (alpha * ctx_tokens + overhead) / (1 + alpha * h_rem)


def rule_ab(f, p, c_turn):
    """Rule A (bypass to the whole file) then Rule B (greedy set that lowers expected cost). Returns set or None.

    f: dict(cost=[per-symbol body tokens], out_tok=outline tokens, file_tok=whole-file tokens); p: need probabilities.
    Adding j changes the expected cost by c_j (1 - p_j) - C_turn * prod_out * p_j / (1 - p_j), where prod_out is the
    probability that nothing outside the set is needed; take the most negative change until none is negative.
    """
    cost = np.asarray(f["cost"], float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    if f["out_tok"] + (1 - p.max()) * c_turn + float((p * cost).sum()) >= f["file_tok"]:
        return None
    inc = np.zeros(len(p), bool)
    while True:
        prod_out = np.prod(1 - p[~inc])
        delta = cost * (1 - p) - c_turn * prod_out * p / (1 - p)
        delta[inc] = np.inf
        j = int(np.argmin(delta))
        if delta[j] >= 0:
            return set(np.where(inc)[0])
        inc[j] = True
