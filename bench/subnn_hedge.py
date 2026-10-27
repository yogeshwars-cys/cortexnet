"""Librarian/Professor switch: MiniLM ranks by default, Glimmer L26 is scored silently, and weight shifts to Glimmer
once it has been right more often in this session (discounted Hedge with a click threshold).

Post-pass over out-of-fold per-symbol scores (bench/modal_glimmer.py::dump -> oof_minilm.npz, oof_glimmer_L26.npz).
Cases of one read (same session and step) share alpha and update the scoreboard together, after the read.
Knobs (gamma, eta1, eta2, tau, b0) are picked by nested CV over sessions; arms are compared with a session-clustered
paired bootstrap.

  python bench/subnn_hedge.py [--dir D:/dev-cache/cortexnet/subnn] [--out bench/results/subnn_hedge.md]
"""
import argparse
import itertools
import os

import numpy as np

FULL = "full Cortex-SubNN"
HERE = os.path.dirname(os.path.abspath(__file__))


def load(path, name):
    z = np.load(path)
    names = list(z["names"])
    s, off = z[f"s{names.index(name)}"], z["off"]
    return dict(keys=z["keys"], sid=z["sid"], step=z["step"], y=z["y"], off=off,
                s=[s[off[i]:off[i + 1]] for i in range(len(off) - 1)])


def zs(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def rr(score, y):
    order = np.argsort(-(score + 1e-9 * np.arange(len(score))[::-1]), kind="stable")
    return 1.0 / (1 + int(np.flatnonzero(y[order])[0]))


def margin(z):
    if len(z) < 2:
        return 0.0
    a = np.sort(z)[::-1]
    return a[0] - a[1]


def sessions(D):
    """sid -> list of reads, each a list of case indices, in step order."""
    by = {}
    for i, (s, t) in enumerate(zip(D["sid"], D["step"])):
        by.setdefault(s, {}).setdefault(int(t), []).append(i)
    return {s: [v[t] for t in sorted(v)] for s, v in by.items()}


def run(S, Zl, Zp, ys, knobs, sids):
    """Per-case reciprocal rank of the mixed ranking, for the cases of `sids`."""
    g, e1, e2, tau, b0 = knobs
    out = {}
    for s in sids:
        A = -b0
        for read in S[s]:
            upd = 0.0
            for i in read:
                dm = margin(Zp[i]) - margin(Zl[i])
                al = 1 / (1 + np.exp(-(e1 * A + e2 * dm - tau)))
                out[i] = rr((1 - al) * Zl[i] + al * Zp[i], ys[i])
                upd += rr(Zp[i], ys[i]) - rr(Zl[i], ys[i])
            A = g * A + upd / len(read)
    return out


def boot(diff_by_sid, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    sids = list(diff_by_sid)
    tot = np.array([diff_by_sid[s][0] for s in sids])
    cnt = np.array([diff_by_sid[s][1] for s in sids])
    est = tot.sum() / cnt.sum()
    bs = []
    for _ in range(n):
        ix = rng.integers(0, len(sids), len(sids))
        bs.append(tot[ix].sum() / cnt[ix].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.environ.get("SUBNN_DATA", "D:/dev-cache/cortexnet/subnn"))
    ap.add_argument("--lib", default="oof_minilm.npz")
    ap.add_argument("--prof", default="oof_glimmer_L26.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "subnn_hedge.md"))
    a = ap.parse_args()
    L, P = load(os.path.join(a.dir, a.lib), FULL), load(os.path.join(a.dir, a.prof), FULL)
    assert (L["keys"] == P["keys"]).all() and (L["off"] == P["off"]).all(), "case order differs between dumps"
    n = len(L["s"])
    ys = [L["y"][L["off"][i]:L["off"][i + 1]].astype(bool) for i in range(n)]
    Zl, Zp = [zs(x) for x in L["s"]], [zs(x) for x in P["s"]]
    r_l = np.array([rr(Zl[i], ys[i]) for i in range(n)])
    r_p = np.array([rr(Zp[i], ys[i]) for i in range(n)])
    S = sessions(L)
    sid_of = L["sid"]

    # ---- where does each expert win?
    pos = np.zeros(n, int)                      # read index within the session
    for s, reads in S.items():
        for k, read in enumerate(reads):
            pos[read] = k
    buckets = [(0, 0, "1st read"), (1, 2, "2nd-3rd"), (3, 6, "4th-7th"), (7, 10 ** 9, "8th+")]

    # ---- arms
    static = {al: np.array([rr((1 - al) * Zl[i] + al * Zp[i], ys[i]) for i in range(n)])
              for al in np.round(np.linspace(0, 1, 11), 1)}
    grid = list(itertools.product([0.5, 0.7, 0.8, 0.9, 1.0], [2.0, 4.0, 8.0], [0.0, 0.5, 1.0],
                                  [0.0, 1.0, 2.0, 4.0], [0.0, 0.25, 0.5]))
    sids = sorted(S)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(sids))
    fold = {sids[j]: k % a.folds for k, j in enumerate(perm)}

    def nested(knob_grid):
        res, picks = {}, []
        for k in range(a.folds):
            tr = [s for s in sids if fold[s] != k]
            te = [s for s in sids if fold[s] == k]
            best = max(knob_grid, key=lambda kb: np.mean(list(run(S, Zl, Zp, ys, kb, tr).values())))
            picks.append(best)
            res.update(run(S, Zl, Zp, ys, best, te))
        return np.array([res[i] for i in range(n)]), picks

    def nested_static():
        res, picks = np.zeros(n), []
        for k in range(a.folds):
            tr = np.array([fold[s] != k for s in sid_of])
            best = max(static, key=lambda al: static[al][tr].mean())
            picks.append(best)
            res[~tr] = static[best][~tr]
        return res, picks

    hedge, hp = nested(grid)
    hedge_A, hpA = nested([kb for kb in grid if kb[2] == 0.0])            # scoreboard only, no margin
    gate_m, _ = nested([(0.5, 0.0, e2, t, 0.0) for e2 in [0.5, 1.0, 2.0, 4.0] for t in [0.0, 1.0, 2.0, 4.0]])
    st, sp = nested_static()
    arms = {"Librarian only (MiniLM full SubNN)": r_l,
            "Professor only (Glimmer L26 full SubNN)": r_p,
            "static 50/50 z-score blend": static[0.5],
            "static blend, alpha picked by nested CV": st,
            "margin-only gate (no scoreboard, eta1 = 0)": gate_m,
            "Hedge scoreboard only (eta2 = 0)": hedge_A,
            "Hedge scoreboard + margin (the proposal)": hedge,
            "oracle per case (upper bound)": np.maximum(r_l, r_p)}

    def vs_lib(r):
        d = {}
        for s in sids:
            m = sid_of == s
            d[s] = ((r[m] - r_l[m]).sum(), m.sum())
        return boot(d)

    L_ = ["# Librarian / Professor switch (online Hedge over two rankers)", "",
          f"Out-of-fold per-symbol scores of the full Cortex-SubNN (3-seed ensembles, 5-fold CV grouped by session) "
          f"for MiniLM-L6 (Librarian, default) and Muse Glimmer 30B layer 26 (Professor, scored silently). "
          f"{n} read-time cases in {len(sids)} sessions ({sum(len(v) for v in S.values())} reads). Scores are "
          f"z-normalised within the file. Cases of one read share alpha; the scoreboard updates after the read with "
          f"both experts' reciprocal ranks. Knobs (gamma, eta1, eta2, tau, b0) and static alpha are picked by "
          f"{a.folds}-fold nested CV over sessions (a different session split from the training CV); CIs are a "
          f"session-clustered paired bootstrap vs the Librarian.", "",
          "| arm | MRR | vs Librarian (95% CI) |", "|---|---|---|"]
    for name, r in arms.items():
        est, lo, hi = vs_lib(r)
        L_.append(f"| {name} | {r.mean():.3f} | {est:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    L_ += ["", "## Where each expert wins", "",
           f"Per-case reciprocal ranks: correlation {np.corrcoef(r_l, r_p)[0, 1]:.2f}; Professor strictly better on "
           f"{(r_p > r_l).mean():.0%} of cases, Librarian on {(r_l > r_p).mean():.0%}, tied on {(r_l == r_p).mean():.0%}.",
           "", "| read index in session | cases | Librarian MRR | Professor MRR | Professor - Librarian (95% CI) |",
           "|---|---|---|---|---|"]
    for lo_, hi_, lab in buckets:
        m = (pos >= lo_) & (pos <= hi_)
        if not m.any():
            continue
        d = {}
        for s in set(sid_of[m]):
            mm = m & (sid_of == s)
            d[s] = ((r_p[mm] - r_l[mm]).sum(), mm.sum())
        est, lo, hi = boot(d)
        L_.append(f"| {lab} | {m.sum()} | {r_l[m].mean():.3f} | {r_p[m].mean():.3f} | {est:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    L_ += ["", "Static blend by alpha (in-sample, for shape only): " +
           ", ".join(f"{al:.1f}: {v.mean():.3f}" for al, v in static.items()), "",
           f"Knobs picked per outer fold (gamma, eta1, eta2, tau, b0): {hp}", f"Static alpha per fold: {[float(x) for x in sp]}", ""]
    open(a.out, "w", encoding="utf8").write("\n".join(L_))
    print("\n".join(L_))


if __name__ == "__main__":
    main()
