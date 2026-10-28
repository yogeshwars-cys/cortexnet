"""Attention-distilled labels for Cortex-SubNN: does Glimmer's own reading help the coprocessor as a soft target?

The session replay labels (edits and ranged re-reads) saturate the coprocessor at MRR ~0.42. Glimmer 30B's layer-28
attention over 65 whole-file reads in the SWE-bench A/B ranks the needed symbols at MRR 0.85, so it is a much
stronger label, but only where the model saw the whole file. This trains the same head (bench/subnn_train.py, full
config) three ways on the same folds and compares held-out ranking:

  C0  sessions + git cases (the current training data)
  C1  C0 + the SWE-bench read cases with their hard labels (gold-patch lines, later ranges, submitted symbol)
  C2  C1 + KL(a || softmax(S / tau)) on the reads that have attention, a_s = attention mass on symbol s / total,
      added as lam * sum(KL) / batch size to the margin InfoNCE loss

Folds: sessions as in subnn_train.py (session-grouped, so C0 reproduces its numbers), SWE instances in their own
balanced folds; no instance or session is in both train and test. Early stopping and validation use session cases
only, so SWE labels act through training alone. Test sets: (A) session read-time cases, (B) SWE read cases.
Data: D:/dev-cache/cortexnet/subnn (private sessions) and D:/dev-cache/cortexnet/ab (public SWE-bench episodes).
Only aggregates are written. Usage: python -I bench/subnn_attn.py [--folds 5] [--ens 2] [--lam 1.0] [--quick]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import subnn_train as S  # noqa: E402
from cortexnet.coproc import leaves  # noqa: E402

AB = "D:/dev-cache/cortexnet/ab"


def swe_cases():
    """SWE read cases from every arm's episodes; the whole arm's attention usage attached as `attn` (per leaf)."""
    cases, texts, n_attn = [], {}, 0
    for arm in ("whole", "cortex", "fade"):
        for f in sorted(glob.glob(os.path.join(AB, arm, "*.json"))):
            ep = json.load(open(f, encoding="utf8"))
            use = {u["call"]: u for u in ep.get("usage") or []}
            texts.update(ep.get("texts") or {})
            for c in ep.get("cases") or []:
                c = dict(c, sid=f"swe:{c['sid']}", arm=arm)
                u = use.get(c["step"])
                if u and u.get("sym") and c["text"] in texts:
                    syms = leaves(texts[c["text"]], c["path"])
                    cnt = {}
                    for s in syms:
                        cnt[s.name] = cnt.get(s.name, 0) + 1
                    a = np.array([u["sym"].get(s.name, 0.0) / cnt[s.name] for s in syms], float)
                    if a.sum() > 0:
                        c["attn"] = (a / a.sum()).tolist()
                        n_attn += 1
                cases.append(c)
    return cases, texts, n_attn


def kl_loss(orig):
    def loss(model, pack, ix, cfg, gen):
        base = orig(model, pack, ix, cfg, gen)
        lam = cfg.get("lam", 0.0)
        m = pack.has_a[ix]
        if not lam or not bool(m.any()):
            return base
        jx = ix[m]
        R, valid = pack.infile(jx)
        P = (pack.P[jx, :R.shape[1]] - model.phi_mu) / model.phi_sd * valid[..., None]
        sc, _ = model(pack.views(jx, cfg["views"] == "single"), pack.E(R), (R >= 0).float(), P)
        logq = torch.log_softmax((sc / cfg["tau"]).masked_fill(~valid, -1e9), 1)
        a = pack.A[jx, :R.shape[1]]
        kl = (a * (torch.log(a.clamp(min=1e-12)) - logq)).masked_fill(~valid | (a <= 0), 0.0).sum(1)
        return base + lam * kl.sum() / len(ix)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ens", type=int, default=2, help="seeds averaged per condition and fold")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the attention KL term in C2")
    ap.add_argument("--quick", action="store_true", help="2 folds, 1 seed, 10 epochs (smoke test)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "subnn_attn.md"))
    a = ap.parse_args()
    if a.quick:
        a.folds, a.ens, a.epochs = 2, 1, 10
    t0 = time.time()
    cases, texts = S.load()
    sw, swt, n_attn = swe_cases()
    texts = dict(texts, **swt)
    S.PREP = S.Prep(cases + sw, texts)
    C = S.PREP.case
    for c in C:
        c["swe"] = c["c"]["sid"].startswith("swe:")
    emb, cross, enc_name, cached = S.frozen_inputs(C, "minilm")
    pack = S.Pack(S.PREP, C, emb, cross)
    A = torch.zeros(pack.y.shape)
    has = torch.zeros(len(C), dtype=torch.bool)
    for i, c in enumerate(C):
        if "attn" in c["c"]:
            v = torch.tensor(c["c"]["attn"], dtype=torch.float32)
            A[i, :len(v)], has[i] = v, True
    pack.A, pack.has_a = A, has
    S.batch_loss = kl_loss(S.batch_loss)                     # train() looks it up in the module at call time
    print(f"{len(C)} cases ({sum(c['swe'] for c in C)} SWE, {int(has.sum())} with attention of {n_attn}), "
          f"{emb.shape[0]} texts, {enc_name} {'cached' if cached else 'computed'} ({time.time() - t0:.0f}s)",
          flush=True)

    base = dict(head="bilinear", rank=16, diag=True, lowrank=True, views="multi", gate=True, late=True, struct=True,
                quant=True, cos_skip=True, cross=False, alpha=0.05, beta=0.2, c0=100.0, tau=0.1, lr=2e-3, bs=32,
                k_neg=12, epochs=a.epochs, patience=6)
    conds = {"C0 sessions + git": dict(swe=False, lam=0.0), "C1 + SWE hard labels": dict(swe=True, lam=0.0),
             f"C2 + SWE hard labels + attention KL (lam {a.lam:g})": dict(swe=True, lam=a.lam)}
    real = lambda c: not c["swe"] and c["c"]["source"] == "sessions" and c["c"].get("views_at") != "edit"  # noqa
    fold = S.folds_by_session([c for c in C if not c["swe"]], a.folds)
    fold.update(S.folds_by_session([c for c in C if c["swe"]], a.folds))
    test = {"A sessions": [], "B SWE": []}
    scores = {n: {} for n in list(conds) + ["frozen zero-shot"]}
    for k in range(a.folds):
        te_s = [c for c in C if real(c) and fold.get(c["c"]["sid"]) == k]
        te_w = [c for c in C if c["swe"] and fold.get(c["c"]["sid"]) == k]
        train_sids = sorted({c["c"]["sid"] for c in C if real(c) and fold.get(c["c"]["sid"]) != k})
        rng = np.random.default_rng(100 + k)
        val_sids = set(rng.choice(train_sids, max(1, int(0.15 * len(train_sids))), replace=False))
        va = [c for c in C if real(c) and c["c"]["sid"] in val_sids]
        tr0 = [c for c in C if not c["swe"] and (c["c"]["source"] != "sessions" or
                                                 (fold.get(c["c"]["sid"], -1) != k and c["c"]["sid"] not in val_sids))]
        trw = [c for c in C if c["swe"] and fold.get(c["c"]["sid"]) != k]
        test["A sessions"] += te_s
        test["B SWE"] += te_w
        zs = S.make_rankers(tr0, emb)["frozen zero-shot (max sub-span cosine)"]
        for c in te_s + te_w:
            scores["frozen zero-shot"][id(c)] = zs(c)
        for name, ov in conds.items():
            cfg = dict(base, lam=ov["lam"])
            tr = tr0 + (trw if ov["swe"] else [])
            acc = {}
            for s_ in range(a.ens):
                model, vb = S.train(pack, tr, va, cfg, seed=1000 * k + s_, log=print)
                sc = S.scorer(model, pack, te_s + te_w, cfg)
                for c in te_s + te_w:
                    acc[id(c)] = acc.get(id(c), 0) + sc(c) / a.ens
            scores[name].update(acc)
            print(f"fold {k + 1}/{a.folds} {name}: train {len(tr)}, test sessions {len(te_s)}, SWE {len(te_w)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    L = ["# Cortex-SubNN with attention-distilled labels", "",
         f"{a.folds}-fold CV (sessions grouped as in subnn_train.py, SWE instances grouped separately), {a.ens} seeds "
         f"averaged per condition and fold, {a.epochs} epochs max, early stopping on session cases only. "
         f"{int(has.sum())} SWE reads carry Glimmer layer-28 attention (whole-file arm). Encoder {enc_name}.", ""]
    for tname, cs in test.items():
        L += [f"## Test set {tname} ({len(cs)} read cases)", "",
              "| ranker | MRR | top-1 | recall@budget | any hit | MRR diff vs C0 (95% CI) |", "|---|---|---|---|---|---|"]
        rr0 = S.per_case_rr(lambda c: scores["C0 sessions + git"][id(c)], cs)
        for name, sc in scores.items():
            f = lambda c, sc=sc: sc[id(c)]  # noqa: E731
            r = S.evaluate_rank(f, cs)
            d = S.boot_diff(S.per_case_rr(f, cs), rr0)
            L.append(f"| {name} | {r['mrr']:.3f} | {r['top1']:.3f} | {r['rec']:.3f} | {r['anyhit']:.3f} | "
                     + ("" if name.startswith("C0") else f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]") + " |")
        L.append("")
    L += [f"Run time {time.time() - t0:.0f}s on CPU."]
    if not a.quick:
        open(a.out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
