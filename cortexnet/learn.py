"""Which symbol bodies to show up front: an online logistic model taught by the model's own range requests.

After an outline-first read, every range the model asks for is a miss signal: those symbols were needed and were
not shown. Symbols of that read that are never asked for within the next `horizon` calls on the file count as
not needed. No external teacher: the labels come from the agent's own behaviour.

Features per symbol (all available at read time):
  query_name     the symbol's name appears in the query
  query_overlap  share of the symbol's identifiers that appear in the query
  changed        the symbol's lines changed since this file was last read (someone just edited it)
  asked_before   log1p(times its lines were requested earlier in this session)
  centrality     log1p(how many other symbols in the file reference its name)        (structural baseline)
  rarity         share of its identifiers that occur nowhere else in the file        (entropic baseline)
  log_tokens     size of the body
  kind_*         def / class / const / imports / section-or-heading
  depth          nesting level
  trail          decayed session memory of identifiers (queries, ranges, shown bodies): best name match + mean body match
  conn           symbol-graph activation from recently touched symbols: calls, called-by, body overlap, decayed by age
"""
import json
import math
import os
import re

import numpy as np

FEATURES = ["query_name", "query_overlap", "changed", "asked_before", "centrality", "rarity", "log_tokens",
            "kind_def", "kind_class", "kind_const", "kind_imports", "kind_section", "depth", "trail", "conn"]
PRIOR = {"query_name": 5.0, "query_overlap": 2.0, "changed": 4.0, "asked_before": 1.0, "log_tokens": -0.2,
         "trail": 2.0, "conn": 1.5}
NL = "\n"
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def idents(text):
    out = set()
    for w in IDENT.findall(text or ""):
        out.add(w.lower())
        out.update(p.lower() for p in re.split(r"_|(?<=[a-z])(?=[A-Z])", w) if len(p) > 2)
    return out


def link(name_a, body_a, name_b, body_b):
    """Symbol-graph edge weight: a calls b, b calls a, plus body identifier cosine."""
    cos = len(body_a & body_b) / math.sqrt(max(len(body_a), 1) * max(len(body_b), 1))
    return float(name_b in body_a) + float(name_a in body_b) + cos


def featurize(syms, lines, query, changed, asked, trail=None, active=(), key=None, mask=None):
    """active: iterable of (weight, key, name, body identifiers) for recently touched symbols, any file."""
    q = idents(query)
    trail = trail or {}
    bodies = [idents(NL.join(lines[s.start - 1:s.end])) for s in syms]
    count = {}
    for b in bodies:
        for w in b:
            count[w] = count.get(w, 0) + 1
    X = np.zeros((len(syms), len(FEATURES)), np.float32)
    for r, (s, b) in enumerate(zip(syms, bodies)):
        nm, parts = s.name.lower(), idents(s.name)
        X[r, 0] = float(bool(q) and (nm in q or (bool(parts) and parts <= q)))
        X[r, 1] = len(b & q) / max(len(b), 1) if q else 0.0
        X[r, 2] = float(any(s.start <= ln <= s.end for ln in changed))
        X[r, 3] = math.log1p(sum(1 for (a, z) in asked if a <= s.end and z >= s.start))
        X[r, 4] = math.log1p(sum(1 for o, ob in zip(syms, bodies) if o is not s and nm in ob))
        X[r, 5] = sum(1 for w in b if count.get(w, 0) == 1) / max(len(b), 1)
        X[r, 6] = math.log1p(sum(len(l) for l in lines[s.start - 1:s.end]) / 4)
        k = {"def": 7, "class": 8, "const": 9, "imports": 10}.get(s.kind, 11)
        X[r, k] = 1.0
        X[r, 12] = s.depth
        if trail:
            X[r, 13] = max([trail.get(w, 0.0) for w in parts | {nm}] or [0.0]) + \
                sum(trail.get(w, 0.0) for w in b) / max(len(b), 1)
        me = key(s) if key else None
        X[r, 14] = sum(w * link(an, ab, nm, b) for w, ak, an, ab in active if ak != me)
    if mask is not None:
        X[:, ~mask] = 0.0
    return X, bodies


class IncludeModel:
    def __init__(self, path=None, lr=0.1, horizon=10):
        self.path, self.lr, self.horizon = path, lr, horizon
        self.w = np.array([PRIOR.get(f, 0.0) for f in FEATURES], np.float64)
        self.b, self.updates = -3.0, 0
        if path and os.path.exists(path):
            st = json.load(open(path, encoding="utf8"))
            if st.get("features") == FEATURES:
                self.w, self.b, self.updates = np.array(st["w"]), st["b"], st["updates"]

    def prob(self, X):
        return 1 / (1 + np.exp(-(X @ self.w + self.b)))

    def update(self, X, y):
        g = np.asarray(y, float) - self.prob(X)
        self.w += self.lr * (g @ X) / len(y)
        self.b += self.lr * g.mean()
        self.updates += 1

    def save(self):
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            json.dump({"features": FEATURES, "w": self.w.tolist(), "b": self.b, "updates": self.updates},
                      open(self.path, "w", encoding="utf8"), indent=1)
