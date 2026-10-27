"""One combined report instead of separate clerks: stacked fusion of every ranker with a per-read gate.

Per symbol, x = z-scored (within the file) scores of every ranker in the out-of-fold dumps (MiniLM and Glimmer L26:
full Cortex-SubNN and frozen zero-shot; lexical; v1 include model; smallest first). Per read, g = context: log symbol
count, agreement between the two full rankers (Spearman, top-1 match), each one's top-1/top-2 margin, lexical peak.

  linear:  score(s) = w . x_s
  gated:   score(s) = sum_j pi_j(g) x_sj + w . x_s,  pi = softmax(G g + b)     (mixture over clerks, per read)
  bilinear: score(s) = (w + V g) . x_s                                          (any per-read re-weighting)

Trained listwise (log of the softmax mass on the needed symbols), L2 picked by inner CV, evaluated by 5-fold CV over
sessions (the same session folds as subnn_hedge.py); session-clustered paired bootstrap vs MiniLM full.

  python bench/subnn_fuse.py [--dir D:/dev-cache/cortexnet/subnn] [--out bench/results/subnn_fuse.md]
"""
import argparse
import os

import numpy as np
import torch

from subnn_hedge import FULL, boot, load, margin, rr, zs

HERE = os.path.dirname(os.path.abspath(__file__))
ZERO = "frozen zero-shot (max sub-span cosine)"
OTHERS = ["lexical (name + identifier overlap with prompt/snapshot)", "v1 include features, logistic", "smallest first"]


def spearman(a, b):
    if len(a) < 3:
        return 0.0
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    c = np.corrcoef(ra, rb)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def build(dir_, lib, prof):
    L = {n: load(os.path.join(dir_, lib), n) for n in (FULL, ZERO, *OTHERS)}
    P = {n: load(os.path.join(dir_, prof), n) for n in (FULL, ZERO)}
    base = L[FULL]
    assert (base["keys"] == P[FULL]["keys"]).all() and (base["off"] == P[FULL]["off"]).all()
    off = base["off"]
    n = len(off) - 1
    cols = [("MiniLM full", L[FULL]), ("Glimmer L26 full", P[FULL]), ("MiniLM zero-shot", L[ZERO]),
            ("Glimmer L26 zero-shot", P[ZERO]), ("lexical", L[OTHERS[0]]), ("v1 include", L[OTHERS[1]]),
            ("smallest first", L[OTHERS[2]])]
    X, Y, G = [], [], []
    for i in range(n):
        x = np.stack([zs(np.asarray(d["s"][i], float)) for _, d in cols], 1)
        lf, pf = x[:, 0], x[:, 1]
        g = [np.log(len(lf)), spearman(lf, pf), float(np.argmax(lf) == np.argmax(pf)), margin(lf), margin(pf),
             float(np.max(np.asarray(L[OTHERS[0]]["s"][i], float)))]
        X.append(x)
        Y.append(base["y"][off[i]:off[i + 1]].astype(bool))
        G.append(g)
    G = np.array(G)
    G = (G - G.mean(0)) / (G.std(0) + 1e-9)
    return [c for c, _ in cols], X, Y, G, base["sid"]


class Fuse(torch.nn.Module):
    def __init__(self, d, dg, kind):
        super().__init__()
        self.kind = kind
        self.w = torch.nn.Parameter(torch.zeros(d))
        with torch.no_grad():
            self.w[0] = 1.0                                   # start as the Librarian alone
        if kind == "gated":
            self.G = torch.nn.Linear(dg, d)
            torch.nn.init.zeros_(self.G.weight)
            torch.nn.init.zeros_(self.G.bias)
            self.scale = torch.nn.Parameter(torch.zeros(()))  # mixture starts switched off
        if kind == "bilinear":
            self.V = torch.nn.Parameter(torch.zeros(d, dg))

    def weights(self, g):
        if self.kind == "linear":
            return self.w.expand(g.shape[0], -1)
        if self.kind == "gated":
            return self.w + torch.softmax(self.G(g), -1) * self.scale
        return self.w + g @ self.V.T

    def reg(self):
        r = (self.w[1:] ** 2).sum() + (self.w[0] - 1) ** 2
        if self.kind == "gated":
            r = r + (self.G.weight ** 2).sum()
        if self.kind == "bilinear":
            r = r + (self.V ** 2).sum()
        return r


def pack(X, Y, ix):
    m = max(len(Y[i]) for i in ix)
    d = X[ix[0]].shape[1]
    xs = torch.zeros(len(ix), m, d)
    ys = torch.zeros(len(ix), m, dtype=torch.bool)
    mask = torch.zeros(len(ix), m, dtype=torch.bool)
    for k, i in enumerate(ix):
        n = len(Y[i])
        xs[k, :n] = torch.as_tensor(X[i], dtype=torch.float32)
        ys[k, :n] = torch.as_tensor(Y[i])
        mask[k, :n] = True
    return xs, ys, mask


def fit(X, Y, G, ix, kind, lam, steps=400, seed=0):
    torch.manual_seed(seed)
    xs, ys, mask = pack(X, Y, ix)
    g = torch.as_tensor(G[ix], dtype=torch.float32)
    m = Fuse(xs.shape[2], g.shape[1], kind)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(steps):
        sc = (xs * m.weights(g)[:, None, :]).sum(-1).masked_fill(~mask, -1e9)
        loss = -(torch.logsumexp(sc.masked_fill(~ys, -1e9), 1) - torch.logsumexp(sc, 1)).mean() + lam * m.reg()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


def predict(m, X, Y, G, ix):
    with torch.no_grad():
        w = m.weights(torch.as_tensor(G[ix], dtype=torch.float32)).numpy()
    return {i: rr(X[i] @ w[k], Y[i]) for k, i in enumerate(ix)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.environ.get("SUBNN_DATA", "D:/dev-cache/cortexnet/subnn"))
    ap.add_argument("--lib", default="oof_minilm.npz")
    ap.add_argument("--prof", default="oof_glimmer_L26.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "subnn_fuse.md"))
    a = ap.parse_args()
    names, X, Y, G, sid = build(a.dir, a.lib, a.prof)
    n = len(Y)
    sids = sorted(set(sid))
    perm = np.random.default_rng(0).permutation(len(sids))
    fold = {sids[j]: k % a.folds for k, j in enumerate(perm)}
    fcase = np.array([fold[s] for s in sid])
    r_lib = np.array([rr(X[i][:, 0], Y[i]) for i in range(n)])
    r_prof = np.array([rr(X[i][:, 1], Y[i]) for i in range(n)])
    lams = [1e-3, 1e-2, 1e-1]

    def nested(kind):
        out, picks, ws = {}, [], []
        for k in range(a.folds):
            tr = np.flatnonzero(fcase != k)
            te = np.flatnonzero(fcase == k)
            inner = fcase[tr]

            def inner_score(lam):
                got = []
                for k2 in sorted(set(inner)):
                    if k2 == k:
                        continue
                    t2, v2 = tr[inner != k2], tr[inner == k2]
                    got += list(predict(fit(X, Y, G, t2, kind, lam), X, Y, G, v2).values())
                return np.mean(got)
            lam = max(lams, key=inner_score)
            picks.append(lam)
            m = fit(X, Y, G, tr, kind, lam)
            ws.append(m.w.detach().numpy().copy())
            out.update(predict(m, X, Y, G, te))
        return np.array([out[i] for i in range(n)]), picks, np.mean(ws, 0)

    arms = {"Librarian only (MiniLM full)": r_lib, "Glimmer L26 full": r_prof,
            "static 50/50 blend (MiniLM full + Glimmer full)": np.array(
                [rr(X[i][:, 0] + X[i][:, 1], Y[i]) for i in range(n)])}
    info = {}
    for kind in ("linear", "gated", "bilinear"):
        r, picks, w = nested(kind)
        arms[f"combined report, {kind}"] = r
        info[kind] = (picks, w)
        print(f"{kind}: MRR {r.mean():.3f}, lambda per fold {picks}", flush=True)
    arms["oracle per case, MiniLM vs Glimmer (upper bound)"] = np.maximum(r_lib, r_prof)
    arms["oracle per case, any of the 7 clerks"] = np.max(
        [[rr(X[i][:, j], Y[i]) for j in range(len(names))] for i in range(n)], 1)

    def vs_lib(r):
        return boot({s: ((r[sid == s] - r_lib[sid == s]).sum(), (sid == s).sum()) for s in sids})

    lines = ["# Combined report: stacked fusion of every ranker with a per-read gate", "",
             f"Out-of-fold per-symbol scores ({n} read-time cases, {len(sids)} sessions). Clerks: "
             f"{', '.join(names)}, each z-scored within the file. Read context: log symbol count, MiniLM/Glimmer "
             f"Spearman and top-1 agreement, both margins, lexical peak. Listwise loss; L2 by inner CV; 5-fold CV over "
             f"sessions; session-clustered paired bootstrap vs MiniLM full.", "",
             "| arm | MRR | vs Librarian (95% CI) |", "|---|---|---|"]
    for name, r in arms.items():
        est, lo, hi = vs_lib(r)
        lines.append(f"| {name} | {r.mean():.3f} | {est:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "Mean learned base weights per clerk (linear): " +
              ", ".join(f"{c} {w:+.2f}" for c, w in zip(names, info["linear"][1])),
              "", "Single clerks: " + ", ".join(
                  f"{c} {np.mean([rr(X[i][:, j], Y[i]) for i in range(n)]):.3f}" for j, c in enumerate(names)),
              "", f"L2 picked per fold: " + "; ".join(f"{k} {v[0]}" for k, v in info.items()), ""]
    open(a.out, "w", encoding="utf8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
