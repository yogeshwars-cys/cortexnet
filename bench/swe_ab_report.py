"""Report for the Glimmer-as-worker A/B (bench/modal_glimmer.py::ab): tokens and KV, outcome, attention usage, lens.

Episodes come from the Modal volume (`modal volume get cortexnet-subnn ab/<arm> <dir>`), one JSON per instance.

1. Token A/B, paired over instances finished in every arm: prefill tokens (Glimmer re-reads the whole context each
   turn here, so this is the compute), peak context, peak KV MB, turns, generation seconds, submission vs gold patch.
2. Attention usage (whole-file arm): for every read_file result, how attention from later generated tokens spreads
   over the file's symbols; the share of the shown symbol tokens that holds 80% of the symbol attention; and where
   the needed symbols rank by usage.
3. Logit lens (whole + cortex arms): rank of the needed symbols among the file's leaf symbols by Glimmer's own
   vocabulary projection at the read decision state, per layer (first-token scores: names sharing a first token tie).

  python bench/swe_ab_report.py [--dir D:/dev-cache/cortexnet/ab] [--out bench/results/swe_ab.md]
"""
import argparse
import glob
import json
import os

import numpy as np

from cortexnet.coproc import leaves

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("whole", "cortex", "fade")


def load(d):
    eps = {}
    for a in ARMS:
        for p in glob.glob(os.path.join(d, a, "*.json")):
            r = json.load(open(p, encoding="utf8"))
            if "metrics" in r:
                eps.setdefault(a, {})[r["instance_id"]] = r
    return eps


def boot_ratio(x, y, n=4000, seed=0):
    """sum(y)/sum(x) with a paired bootstrap over instances."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(x), (n, len(x)))
    r = y[ix].sum(1) / x[ix].sum(1)
    return y.sum() / x.sum(), *np.percentile(r, [2.5, 97.5])


def rr_of(scores, need, rng):
    """Reciprocal rank of the first needed item; ties broken at random."""
    names = list(scores)
    s = np.array([scores[n] for n in names], float) + 1e-9 * rng.random(len(names))
    order = [names[i] for i in np.argsort(-s)]
    hit = [k for k, n in enumerate(order) if n in need]
    return 1 / (1 + hit[0]) if hit else None


def needed(r):
    """call -> (set of needed leaf names, all leaf names) from the episode's labelled cases."""
    out = {}
    for c in r["cases"]:
        text = r["texts"].get(c["text"])
        if text is None:
            continue
        syms = leaves(text, c["path"])
        out[c["step"]] = ({syms[i].name for i in c["y"] if i < len(syms)}, [s.name for s in syms])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/dev-cache/cortexnet/ab")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "swe_ab.md"))
    a = ap.parse_args()
    eps = load(a.dir)
    have = {k: len(v) for k, v in eps.items()}
    common = sorted(set.intersection(*(set(eps.get(k, {})) for k in ARMS)))
    L = ["# Glimmer 30B as the worker: whole-file reads vs CortexNET vs CortexNET + fade", "",
         f"SWE-bench Lite (public), Muse Glimmer 30B on one H100 per arm, HF generate (no KV reuse across turns: "
         f"every turn re-reads the whole context, so prefill tokens are the compute). Episodes per arm: {have}; "
         f"paired over the {len(common)} instances every arm finished. Success = the submitted file is in the gold "
         f"patch / the submitted symbol overlaps its changed lines (no tests are run). KV counts Glimmer's layout: "
         f"13 full-attention layers grow with context, 39 sliding-window layers stop at 2,048 tokens.", ""]

    # ---- 1. tokens and outcome
    M = {k: [eps[k][i]["metrics"] for i in common] for k in ARMS if k in eps}
    keys = [("prefill", "prefill tokens (sum over turns)"), ("peak", "peak context tokens"),
            ("peak_kv_mb", "peak KV cache (MB)"), ("turns", "turns"), ("gen_secs", "generation seconds")]
    L += ["## 1. Cost and outcome (paired)", "", "| metric | whole (mean) | cortex (mean) | fade (mean) | cortex / whole "
          "(95% CI) | fade / whole (95% CI) |", "|---|---|---|---|---|---|"]
    for k, lab in keys:
        w, c, f = ([m[k] for m in M[x]] for x in ARMS)
        rc, rf = boot_ratio(w, c), boot_ratio(w, f)
        L.append(f"| {lab} | {np.mean(w):,.0f} | {np.mean(c):,.0f} | {np.mean(f):,.0f} | "
                 f"{rc[0]:.2f} [{rc[1]:.2f}, {rc[2]:.2f}] | {rf[0]:.2f} [{rf[1]:.2f}, {rf[2]:.2f}] |")
    for k, lab in (("file_ok", "submitted file in gold patch"), ("symbol_ok", "submitted symbol in gold patch"),
                   ("submitted", "submitted at all"), ("hit_limit", "hit the 60k-token context limit"),
                   ("oom", "ran out of GPU memory (80 GB H100)")):
        L.append(f"| {lab} | {np.mean([m.get(k, False) for m in M['whole']]):.0%} | {np.mean([m.get(k, False) for m in M['cortex']]):.0%} "
                 f"| {np.mean([m.get(k, False) for m in M['fade']]):.0%} | | |")
    rt = {x: sum(m["tool_tok"].get("read_file", 0) + m["tool_tok"].get("read_lines", 0) for m in M[x]) for x in ARMS}
    L += ["", f"File tokens put into context (read_file + read_lines, summed): whole {rt['whole']:,}, cortex "
          f"{rt['cortex']:,} ({rt['cortex'] / max(rt['whole'], 1):.0%}), fade {rt['fade']:,}. Reads faded in the fade "
          f"arm: {sum(m['faded'] for m in M['fade'])}.", ""]

    # ---- 2. attention usage (whole arm)
    rng = np.random.default_rng(0)
    share80, body_share, rr_use, rr_rand, n_reads = [], [], [], [], 0
    for r in eps.get("whole", {}).values():
        need = needed(r)
        for u in r["usage"]:
            n_reads += 1
            tot = sum(u["sym"].values())
            if u["block"] > 0:
                body_share.append(tot / u["block"])
            if tot > 0 and u["sym_tok"]:
                dens = sorted(((u["sym"][s] / u["sym_tok"][s], u["sym_tok"][s], u["sym"][s]) for s in u["sym"]),
                              reverse=True)
                cum, tok = 0.0, 0
                for _, t, mss in dens:
                    cum += mss
                    tok += t
                    if cum >= 0.8 * tot:
                        break
                share80.append(tok / sum(u["sym_tok"].values()))
            if u["call"] in need:
                nd, names = need[u["call"]]
                sc = {n: u["sym"].get(n, 0.0) for n in names}
                x = rr_of(sc, nd, rng)
                if x is not None:
                    rr_use.append(x)
                    rr_rand.append(np.mean([rr_of({n: rng.random() for n in names}, nd, rng) for _ in range(20)]))
    L += ["## 2. What Glimmer actually used of a whole file (attention, layer 28)", ""]
    if share80:
        L += [f"- {n_reads} whole-file reads. Share of later attention on the block that lands on symbol bodies (vs "
              f"outline/header): median {np.median(body_share):.0%}.",
              f"- 80% of the attention on symbol bodies falls on **{np.median(share80):.0%}** of the shown symbol "
              f"tokens (median over reads; mean {np.mean(share80):.0%}).",
              f"- Needed symbols ranked by attention usage: MRR {np.mean(rr_use):.3f} vs random {np.mean(rr_rand):.3f} "
              f"({len(rr_use)} labelled reads)." if rr_use else "- No labelled whole-file reads yet.", ""]
    else:
        L += ["- No usage data yet.", ""]

    # ---- 3. logit lens
    pieces_p = os.path.join(a.dir, "lens_pieces.json")
    pieces = json.load(open(pieces_p)) if os.path.exists(pieces_p) else {}
    L += ["## 3. Logit lens at the read decision (Glimmer's own vocabulary, no training)", "",
          "First token: each name scored by its first token only. Word pieces: max log-prob over the pieces of the "
          "identifier (split on _ and case), recomputed from the saved states (`modal_glimmer.py::lens`).", "",
          "| scoring | layer | MRR (needed symbols) | random | reads | names tied |", "|---|---|---|---|---|---|"]
    for src in ("first token", "word pieces"):
        if src == "word pieces" and not pieces:
            continue
        lens_table(src, eps, pieces, rng, L)
    L.append("")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf8").write("\n".join(L))
    print("\n".join(L))


def lens_table(src, eps, pieces, rng, L):
    by_layer = {}
    for arm in ("whole", "cortex"):
        for iid, r in eps.get(arm, {}).items():
            need = needed(r)
            rows = r["lens"] if src == "first token" else pieces.get(arm, {}).get(iid, [])
            for ln in rows:
                if ln["call"] not in need:
                    continue
                nd, names = need[ln["call"]]
                for layer, row in ln["layers"].items():
                    sc = {n: row.get("sym", row).get(n, -1e9) for n in names}
                    x = rr_of(sc, nd, rng)
                    if x is None:
                        continue
                    vals = list(sc.values())
                    tied = 1 - len(set(vals)) / len(vals)
                    rnd = np.mean([rr_of({n: rng.random() for n in names}, nd, rng) for _ in range(20)])
                    by_layer.setdefault(layer, []).append((x, rnd, tied))
    for layer in sorted(by_layer, key=int):
        v = np.array(by_layer[layer])
        L.append(f"| {src} | {layer} | {v[:, 0].mean():.3f} | {v[:, 1].mean():.3f} | {len(v)} | {v[:, 2].mean():.0%} |")


if __name__ == "__main__":
    main()
