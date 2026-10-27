"""Do the trail and symbol-graph features carry signal? Hindsight labels + held-out fits on the session replay.

For every outline-mode read in the replay, each symbol of the file is labelled by hindsight: positive if the agent
later edits lines inside it before the file is read again. Features are computed exactly as the Reader computes them
at read time (all features on, same session state). Then:
  1. AUC of each feature alone, pooled over sessions;
  2. logistic fits on one half of the sessions (by time), evaluated on the other half, with and without trail/conn;
  3. include quality at the Reader's budget on the held-out half: share of later-edited symbols already shown.
Usage: python -I bench/feature_signal.py [projects_dir]
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from replay_sessions import enclosing, load_sessions, write, ws_path  # noqa: E402
from cortexnet import Reader, tokens  # noqa: E402
from cortexnet.learn import FEATURES  # noqa: E402

V1 = [i for i, f in enumerate(FEATURES) if f not in ("trail", "conn")]


class Probe(Reader):
    def _features(self, p, syms, lines, query, changed):
        X, b = super()._features(p, syms, lines, query, changed)
        self.last_X = (p, X, [(s.start, s.end, s.children) for s in syms],
                       [tokens("\n".join(lines[s.start - 1:s.end])) for s in syms])
        return X, b


def collect(sessions, root):
    rows = []                                   # per read: session k, X, spans, costs, labels
    for k, (_, _, evs) in enumerate(sessions):
        ws = os.path.join(root, f"s{k}")
        r = Probe(os.path.join(root, "state"), menu=False)
        r.model.lr = 0.0                         # frozen prior: features only, no learning in the probe
        open_read = {}
        for e in evs:
            wp = ws_path(ws, e["path"])
            p = os.path.abspath(wp)
            if e["op"] == "write":
                write(wp, e["text"])
            elif e["op"] == "read" and not (e["off"] or e["lim"]) and e["rows"][0][0] == 1:
                write(wp, "\n".join(t for _, t in e["rows"]))
                r.last_X = None
                out = r.read(wp, e["query"])
                if "Outline" in out and r.last_X and r.last_X[0] == p:
                    _, X, spans, cost = r.last_X
                    rec = dict(k=k, X=X, spans=spans, cost=cost, y=np.zeros(len(spans), bool))
                    rows.append(rec)
                    open_read[p] = rec
                else:
                    open_read.pop(p, None)
            elif e["op"] == "edit" and os.path.exists(wp):
                with open(wp, encoding="utf8") as f:
                    text = f.read()
                i = text.find(e["old"]) if e["old"] else -1
                if i < 0:
                    continue
                lo = text[:i].count("\n") + 1
                hi = lo + e["old"].count("\n")
                if p in open_read:
                    rec = open_read[p]
                    for j, (a, b, ch) in enumerate(rec["spans"]):
                        if not ch and a <= hi and b >= lo:
                            rec["y"][j] = True
                a, b = enclosing(text, p, lo, hi)
                r.range(p, a, b)
                text = text.replace(e["old"], e["new"]) if e["all"] else text.replace(e["old"], e["new"], 1)
                write(wp, text)
                time.sleep(0.002)
        shutil.rmtree(ws, ignore_errors=True)
    return [x for x in rows if x["y"].any()]


def budget_recall(rows, w, b, cols):
    got = tot = 0
    for x in rows:
        leaf = np.array([not ch for _, _, ch in x["spans"]])
        s = x["X"][:, cols] @ w + b
        budget = max(400, 0.25 * sum(c for c, l in zip(x["cost"], leaf) if l))
        used, shown = 0, set()
        for j in np.argsort(-s):
            if leaf[j] and used + x["cost"][j] <= budget:
                shown.add(j)
                used += x["cost"][j]
        tot += x["y"].sum()
        got += sum(1 for j in np.where(x["y"])[0] if j in shown)
    return got / max(tot, 1)


def main():
    sessions = load_sessions()
    root = tempfile.mkdtemp(dir=os.environ.get("TMP"))
    rows = collect(sessions, root)
    shutil.rmtree(root, ignore_errors=True)
    X = np.vstack([x["X"] for x in rows])
    y = np.concatenate([x["y"] for x in rows])
    leaf = np.concatenate([[not ch for _, _, ch in x["spans"]] for x in rows])
    X, y = X[leaf], y[leaf]
    L = ["# Feature signal: do the trail and the symbol graph predict what gets edited?", "",
         f"{len(rows)} outline-mode reads followed by an edit in the same file; {len(y)} leaf symbols, {int(y.sum())} later "
         "edited before the file was read again. Features as computed at read time.", "",
         "## Single-feature AUC (pooled)", "", "| feature | AUC | share of rows non-zero |", "|---|---|---|"]
    for i, f in enumerate(FEATURES):
        if X[:, i].std() > 0:
            L.append(f"| {f} | {roc_auc_score(y, X[:, i]):.3f} | {(X[:, i] != 0).mean():.0%} |")
    ks = np.array([x["k"] for x in rows])
    cut = np.median(ks)
    halves = [[x for x in rows if x["k"] <= cut], [x for x in rows if x["k"] > cut]]
    L += ["", "## Fitted on one half of the sessions (by time), evaluated on the other", "",
          "| features | held-out AUC (A->B, B->A) | edited symbols shown at the Reader's budget |", "|---|---|---|"]
    for name, cols in [("v1 (no trail, no graph)", V1), ("all (with trail + graph)", list(range(len(FEATURES)))),
                       ("prior weights (shipped)", None)]:
        aucs, recs = [], []
        for tr, te in [(halves[0], halves[1]), (halves[1], halves[0])]:
            Xte = np.vstack([x["X"] for x in te])
            yte = np.concatenate([x["y"] for x in te])
            lte = np.concatenate([[not ch for _, _, ch in x["spans"]] for x in te])
            if cols is None:
                m = Reader.__new__(Reader)
                from cortexnet.learn import IncludeModel
                im = IncludeModel()
                w, b, c = im.w, im.b, list(range(len(FEATURES)))
            else:
                Xtr = np.vstack([x["X"] for x in tr])[:, cols]
                ytr = np.concatenate([x["y"] for x in tr])
                ltr = np.concatenate([[not ch for _, _, ch in x["spans"]] for x in tr])
                mu, sd = Xtr[ltr].mean(0), Xtr[ltr].std(0) + 1e-6
                lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit((Xtr[ltr] - mu) / sd, ytr[ltr])
                w, b, c = lr.coef_[0] / sd, lr.intercept_[0] - (lr.coef_[0] * mu / sd).sum(), cols
            aucs.append(roc_auc_score(yte[lte], Xte[lte][:, c] @ w + b))
            recs.append(budget_recall(te, w, b, c))
        L.append(f"| {name} | {aucs[0]:.3f}, {aucs[1]:.3f} | {np.mean(recs):.0%} |")
    ti = FEATURES.index("log_tokens")
    small = np.zeros(len(FEATURES))
    small[ti] = -1.0
    rnd = np.mean([budget_recall([dict(x, X=np.random.default_rng(s).random(x["X"].shape)) for x in rows],
                                 np.eye(len(FEATURES))[0], 0.0, list(range(len(FEATURES)))) for s in range(20)])
    L += [f"| smallest symbols first (size only) | - | {budget_recall(rows, small, 0.0, list(range(len(FEATURES)))):.0%} |",
          f"| random symbols (mean of 20) | - | {rnd:.0%} |"]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "feature_signal.md")
    open(out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
