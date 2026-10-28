"""Cortex-SubNN: a small coprocessor that picks which symbol bodies to show up front, trained on the agent's own use.

Data: bench/subnn_data.py (private, outside the repo). Only aggregate numbers are written to bench/results/.

Model (one forward per read), on frozen MiniLM embeddings computed once and cached in DATA:
  views      e_q (user prompt), e_h (reasoning snapshot, presence flag delta, length/entropy H), e_prev (decayed trail of
             symbols used earlier in the session), e_qh (prompt + snapshot embedded together, the zero-shot query)
  gate       pi = softmax(W_g [e_q; delta e_h; e_prev; delta; delta_p; H] + log-mask)
  query      u = e_qh + sum_i pi_i (g_i * e_i), g_i = 0 at init (per-dimension blend of each view into the query)
  symbols    M = 4 sub-span prototypes per symbol (signature + 3 body windows), frozen
  score      S = max_r [u . diag(1 + d) e_r + (A u) . (B e_r)] + w_struct . phi(s), with d = 0, B = 0, w_struct = 0 at
             init (A small random, else neither A nor B gets a gradient). At step 0 S is exactly the frozen MiniLM
             zero-shot ranker; training and early stopping (which also scores step 0) can only move it from there.
             The previous bottleneck head (k = 32, int8, cosine skip) is kept as a comparison row.
  loss       cost-weighted margin InfoNCE: in-file siblings + nearest cross-file symbols of the same repo as negatives,
             margin alpha log(1 + c(s-)/c0) on negatives, weight 1 + beta log(1 + c(s+)) on positives
  decision   Platt calibration on the validation window, then Rule A (whole-file bypass) and Rule B (greedy up-front set).
             Cost model with a prefix cache: an extra round trip costs T = alpha_cache L_hist + overhead, and every token
             put into context costs (1 + alpha_cache H_rem), H_rem = later API calls that re-read it before the session
             ends or is compacted (from the session logs). Fixed-C_turn regimes are kept for comparison.

Evaluation: k-fold cross-validation grouped by session (each read-time case tested once by a model that never saw its
session; early stopping and calibration on 15% of the training sessions), plus a time split as a secondary check.
Training data = the other sessions' cases + git cases (down-weighted). Edit-time cases (no read before the edit) are
trained on but reported separately, never mixed into the main test numbers.
Usage: python -I bench/subnn_train.py [--folds 5] [--seeds 3] [--epochs 40] [--quick] [--device cpu|cuda]
(CPU by default: GPU allocations count against Windows commit memory on this machine, and the head is tiny.)
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortexnet import outline, symbols, tokens  # noqa: E402
from cortexnet.learn import FEATURES, featurize, idents  # noqa: E402
from cortexnet.coproc import calib_features, leaves, subspans  # noqa: E402  (shared with the runtime)
from cortexnet.policy import ALPHA_CACHE, TOOL_OVH, rule_ab  # noqa: E402

DATA = os.environ.get("SUBNN_DATA", "D:/dev-cache/cortexnet/subnn")
HERE = os.path.dirname(os.path.abspath(__file__))
ENC = "sentence-transformers/all-MiniLM-L6-v2"
SPLIT, M, GAMMA, MAXLEN = 600, 4, 0.6, 128
DEV = "cpu"
LOGS = os.path.join(os.path.expanduser("~"), ".claude", "projects")


# ---------------------------------------------------------------- data

class Table:
    """Unique texts -> row ids; embeddings computed later."""

    def __init__(self):
        self.ids, self.texts = {}, []

    def row(self, t):
        t = (t or "").strip()
        if not t:
            return -1
        if t not in self.ids:
            self.ids[t] = len(self.texts)
            self.texts.append(t)
        return self.ids[t]


def load():
    cases = [json.loads(l) for l in open(os.path.join(DATA, "cases.jsonl"), encoding="utf8")]
    texts = json.load(open(os.path.join(DATA, "texts.json"), encoding="utf8"))
    return cases, texts


def split(cases):
    ses = sorted([c for c in cases if c["source"] == "sessions"], key=lambda c: c["ts"])
    real = [c for c in ses if c.get("views_at") != "edit"]
    n = len(real)
    t_test, t_val = real[int(0.8 * n)]["ts"], real[int(0.65 * n)]["ts"]
    test = [c for c in real if c["ts"] >= t_test]
    val = [c for c in real if t_val <= c["ts"] < t_test]
    train = [c for c in ses if c["ts"] < t_val] + [c for c in cases if c["source"] == "git"]
    late_edit = [c for c in ses if c.get("views_at") == "edit" and c["ts"] >= t_test]
    return train, val, test, late_edit


class Prep:
    """Everything the model and the baselines need per case, as row ids into one text table."""

    def __init__(self, cases, texts):
        self.tab, self.files, self.case = Table(), {}, []
        self.texts = texts
        for c in cases:
            self.case.append(self.one(c))

    def file(self, h, path):
        key = (h, os.path.splitext(path)[1].lower())
        if key not in self.files:
            text = self.texts[h]
            lines = text.split("\n")
            syms = leaves(text, path)
            rows = np.array([[self.tab.row(t) for t in subspans(s, lines, path)] for s in syms], np.int64)
            cost = np.array([tokens("\n".join(lines[s.start - 1:s.end])) for s in syms], np.float32)
            self.files[key] = dict(syms=syms, lines=lines, rows=rows, cost=cost, path=path, file_tok=tokens(text),
                                   out_tok=tokens(outline(text, path, symbols(text, path, split_tokens=SPLIT))))
        return self.files[key]

    def one(self, c):
        f = self.file(c["text"], c["path"])
        q, h = c["q"] or "", c["h"] or ""
        trail = {}
        for j, it in enumerate(reversed(c["prev"])):
            for w in idents(" ".join(it["names"]) + " " + it["body"]):
                trail[w] = max(trail.get(w, 0.0), GAMMA ** j)
        query = q + "\n" + h
        phi, _ = featurize(f["syms"], f["lines"], query, changed=(), asked=(), trail=trail)
        y = np.zeros(len(f["syms"]), bool)
        y[c["y"]] = True
        hid = idents(h)
        H = np.array([math.log1p(tokens(h)) / 8 if h else 0.0, len(hid) / max(1, len(h.split())) if h else 0.0],
                     np.float32)
        prev = [self.tab.row(" ".join(it["names"]) + "\n" + it["body"][:800]) for it in reversed(c["prev"][-6:])]
        return dict(c=c, key=(c["text"], os.path.splitext(c["path"])[1].lower()), y=y, phi=phi, query=query,
                    trail=trail, q=self.tab.row(q[:1200]), h=self.tab.row(h[-1200:]),
                    qh=self.tab.row((q[:600] + "\n" + h[-600:]).strip()), prev=prev, H=H,
                    w=0.3 if c["source"] == "git" else 1.0)


class Encoder(nn.Module):
    """MiniLM mean-pool + L2 norm (same as sentence-transformers). Frozen: LoRA on the top 2 layers gave +0.004 MRR
    (not significant) in the 2-fold run, so it was dropped and the embeddings are computed once and cached."""

    def __init__(self):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(ENC)
        self.m = AutoModel.from_pretrained(ENC)
        for p in self.m.parameters():
            p.requires_grad_(False)

    def forward(self, texts, bs=256):
        outs = []
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                hs = self.m(**b).last_hidden_state
            msk = b["attention_mask"].unsqueeze(-1).float()
            e = (hs.float() * msk).sum(1) / msk.sum(1).clamp(min=1)
            outs.append(Fn.normalize(e, dim=-1))
        return torch.cat(outs) if outs else torch.zeros(0, 384, device=DEV)


def texts_sha():
    return hashlib.sha1("\0".join(PREP.tab.texts).encode("utf8", "replace")).hexdigest()


def frozen_inputs(C, source="minilm"):
    """Frozen embedding of every text plus the cross-file negatives mined with it, computed once and cached in DATA.

    source = "minilm" (computed here) or the path of a precomputed file {emb [n_texts, d], texts_sha, name} whose rows
    follow PREP.tab.texts (e.g. one Glimmer layer, from bench/modal_glimmer.py). Returns emb, cross, name, cached."""
    h = hashlib.sha1()
    h.update(texts_sha().encode())
    h.update(open(os.path.join(DATA, "cases.jsonl"), "rb").read())
    h.update(f"{source} {ENC} {MAXLEN} {SPLIT} {M}".encode())
    path = os.path.join(DATA, f"frozen_{h.hexdigest()[:12]}.pt")
    if os.path.exists(path):
        d = torch.load(path, weights_only=False)
        return d["emb"].to(DEV), d["cross"], d["name"], True
    if source == "minilm":
        enc = Encoder().to(DEV).eval()
        with torch.no_grad():
            emb = enc(PREP.tab.texts)
        del enc
        name = "MiniLM-L6"
    else:
        d = torch.load(source, weights_only=False)
        if d["texts_sha"] != texts_sha():
            raise SystemExit(f"{source} was embedded from a different text table")
        emb, name = Fn.normalize(d["emb"].float(), dim=-1).to(DEV), d["name"]
    cross = mine_cross(PREP, C, emb)
    torch.save(dict(emb=emb.cpu(), cross=cross, name=name), path)
    return emb, cross, name, False


# ---------------------------------------------------------------- model

def ste8(x):
    return x + (torch.round(x.clamp(-1, 1) * 127) / 127 - x).detach()


class SubNN(nn.Module):
    def __init__(self, nF, d=384, m=256, k=32, views="multi", gate=True, late=True, struct=True, quant=True,
                 cos_skip=True):
        super().__init__()
        self.views, self.gate, self.late, self.struct, self.quant = views, gate, late, struct, quant
        self.cos_skip = cos_skip
        self.wc = nn.Parameter(torch.tensor([1.0, 1.0, 0.5]))     # skip: frozen max-sub-span cosine per view
        self.Wg = nn.Linear(3 * d + 4, 3)
        self.Wq, self.Wh, self.Wp = nn.Linear(d, m), nn.Linear(d, m), nn.Linear(d, m)
        self.ln, self.down, self.skip = nn.LayerNorm(m), nn.Linear(m, k), nn.Linear(m, k, bias=False)
        self.s_ln, self.s1, self.s_down = nn.LayerNorm(d), nn.Linear(d, m), nn.Linear(m, k)
        self.s_skip = nn.Linear(d, k, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.1)))
        self.ws = nn.Linear(nF, 1)
        nn.init.zeros_(self.ws.weight)

    def query(self, eq, eh, ep, dh, dp, H):
        """eq, eh, ep: [B, d]; dh, dp: [B] presence flags; H: [B, 2]."""
        if self.views == "single":            # one stream: prompt + snapshot embedded together (eq holds e_qh)
            u = self.Wq(eq)
            pi = torch.zeros(eq.shape[0], 3, device=eq.device)
            pi[:, 0] = 1
        else:
            mask = torch.stack([torch.ones_like(dh), dh, dp], 1)
            if self.views == "q":
                mask = torch.stack([torch.ones_like(dh), 0 * dh, 0 * dp], 1)
            if self.gate:
                g = self.Wg(torch.cat([eq, eh * dh[:, None], ep, dh[:, None], dp[:, None], H], 1))
            else:
                g = torch.zeros(eq.shape[0], 3, device=eq.device)
            pi = torch.softmax(g + torch.log(mask + 1e-6), 1)
            u = pi[:, :1] * self.Wq(eq) + pi[:, 1:2] * self.Wh(eh) + pi[:, 2:] * self.Wp(ep)
        u = Fn.dropout(u, 0.1, self.training)
        z = Fn.normalize(self.down(Fn.gelu(self.ln(u))) + self.skip(u), dim=-1)
        return (ste8(z) if self.quant else z), pi

    def sym(self, E, msk):
        """E: [..., M, d] sub-span embeddings, msk: [..., M]. Returns [..., M, k] (or [..., 1, k] if not late)."""
        if not self.late:
            E = (E * msk[..., None]).sum(-2, keepdim=True) / msk.sum(-1, keepdim=True)[..., None].clamp(min=1)
            msk = msk[..., :1].clamp(max=1)
        v = Fn.normalize(self.s_down(Fn.gelu(self.s1(self.s_ln(E)))) + self.s_skip(E), dim=-1)
        return (ste8(v) if self.quant else v), msk

    def score(self, z, V, vm, phi, raw=None):
        """z: [B, k], V: [B, C, M', k], vm: [B, C, M'], phi: [B, C, F] -> [B, C].
        raw = (E [B, C, M, d], msk [B, C, M], views (eq, eh, ep, dh, dp)) for the pretrained-cosine skip."""
        dot = torch.einsum("bk,bcmk->bcm", z, V)
        tau = self.log_tau.exp().clamp(0.02, 1.0)
        dot = dot.masked_fill(vm <= 0, -1e4)
        s = tau * torch.logsumexp(dot / tau, -1)
        if self.struct:
            s = s + self.ws(phi).squeeze(-1)
        if self.cos_skip and raw is not None:
            E, msk, (eq, eh, ep, dh, dp) = raw
            cs = []
            for e, w in ((eq, None), (eh, dh), (ep, dp)):
                c = torch.einsum("bd,bcmd->bcm", e, E).masked_fill(msk <= 0, -1.0).max(-1).values
                cs.append(c if w is None else c * w[:, None])
            s = s + (torch.stack(cs, -1) * self.wc).sum(-1)
        return s

    def forward(self, v, E, msk, phi):
        z, pi = self.query(v["eq"], v["eh"], v["ep"], v["dh"], v["dp"], v["H"])
        V, vm = self.sym(E, msk)
        return self.score(z, V, vm, phi, (E, msk, (v["eq"], v["eh"], v["ep"], v["dh"], v["dp"]))), pi

    def gate_weights(self, v):
        return self.query(v["eq"], v["eh"], v["ep"], v["dh"], v["dp"], v["H"])[1]


class Bilinear(nn.Module):
    """Identity-initialised head (see the module docstring): exactly the frozen zero-shot ranker at step 0."""

    def __init__(self, nF, d=384, r=16, views="multi", gate=True, late=True, struct=True, diag=True, lowrank=True):
        super().__init__()
        self.views, self.gate, self.late, self.struct, self.diag, self.lowrank = views, gate, late, struct, diag, lowrank
        self.Wg = nn.Linear(3 * d + 4, 3)
        self.g = nn.Parameter(torch.zeros(3, d))          # per-view blend into the query
        self.dg = nn.Parameter(torch.zeros(d), requires_grad=diag)
        self.A = nn.Parameter(torch.randn(r, d) / math.sqrt(d), requires_grad=lowrank)
        self.B = nn.Parameter(torch.zeros(r, d), requires_grad=lowrank)
        self.ws = nn.Linear(nF, 1)
        nn.init.zeros_(self.ws.weight)
        nn.init.zeros_(self.ws.bias)

    def gate_weights(self, v):
        dh = v["dh"]
        if self.views == "single":
            pi = torch.zeros(dh.shape[0], 3, device=dh.device)
            pi[:, 0] = 1
            return pi
        mask = torch.stack([torch.ones_like(dh), dh, v["dp"]], 1)
        if self.views == "q":
            mask = torch.stack([torch.ones_like(dh), 0 * dh, 0 * dh], 1)
        if self.gate:
            g = self.Wg(torch.cat([v["eq"], v["eh"] * dh[:, None], v["ep"], dh[:, None], v["dp"][:, None], v["H"]], 1))
        else:
            g = torch.zeros(dh.shape[0], 3, device=dh.device)
        return torch.softmax(g + torch.log(mask + 1e-6), 1)

    def forward(self, v, E, msk, phi):
        pi = self.gate_weights(v)
        base = v["eq"] if self.views == "q" else v["eqh"]
        if self.views == "single":
            u = base + self.g[0] * base
        else:
            u = base + pi[:, :1] * self.g[0] * v["eq"] + pi[:, 1:2] * self.g[1] * v["eh"] + \
                pi[:, 2:] * self.g[2] * v["ep"]
        if not self.late:
            E = (E * msk[..., None]).sum(-2, keepdim=True) / msk.sum(-1, keepdim=True)[..., None].clamp(min=1)
            E = Fn.normalize(E, dim=-1)
        s = torch.einsum("bd,bcmd->bcm", u * (1 + self.dg), E)
        if self.lowrank:
            s = s + torch.einsum("br,bcmr->bcm", u @ self.A.t(), E @ self.B.t())
        s = s.max(-1).values        # missing sub-spans are zero vectors scoring 0, exactly as in the zero-shot ranker
        if self.struct:
            s = s + self.ws(phi).squeeze(-1)
        return s, pi


def make_head(cfg, nF, d):
    if cfg["head"] == "bottleneck":
        return SubNN(nF, d=d, views=cfg["views"], gate=cfg["gate"], late=cfg["late"], struct=cfg["struct"],
                     quant=cfg["quant"], cos_skip=cfg["cos_skip"])
    return Bilinear(nF, d=d, r=cfg["rank"], views=cfg["views"], gate=cfg["gate"], late=cfg["late"],
                    struct=cfg["struct"], diag=cfg["diag"], lowrank=cfg["lowrank"])


# ---------------------------------------------------------------- batching

class Pack:
    """Every case as fixed tensors (views, in-file candidates, cross-file negatives), built once per run, so a batch
    is an index into these instead of per-case Python loops."""

    def __init__(self, prep, C, emb, cross, emb_sym=None):
        self.emb, self.ix = emb, {id(c): i for i, c in enumerate(C)}
        self.emb_sym = emb if emb_sym is None else emb_sym       # symbol side (decoded codes with --codec)
        N, d, F = len(C), emb.shape[1], len(FEATURES)
        T = lambda a: torch.as_tensor(np.asarray(a), device=DEV)  # noqa: E731
        self.q, self.h, self.qh = T([c["q"] for c in C]), T([c["h"] for c in C]), T([c["qh"] for c in C])
        self.dh = (self.h >= 0).float()
        self.qv = T([c["c"].get("qv", -1) for c in C])
        self.qvec = None            # live decision states (--qvec): replace e_qh where a case has one
        self.ep, dp = torch.zeros(N, d, device=DEV), []
        for i, c in enumerate(C):
            pr = [r for r in c["prev"] if r >= 0]
            if pr:
                w = torch.tensor([GAMMA ** j for j in range(len(pr))], device=DEV)
                self.ep[i] = Fn.normalize((emb[T(pr)] * w[:, None]).sum(0) / w.sum(), dim=-1)
            dp.append(float(bool(pr)))
        self.dp = T(np.array(dp, np.float32))
        self.H = T(np.stack([c["H"] for c in C]))
        self.w = T(np.array([c["w"] for c in C], np.float32))
        n = [len(c["y"]) for c in C]
        X = max(1, max(len(x) for x in cross))
        R, P = np.full((N, max(n), M), -1, np.int64), np.zeros((N, max(n), F), np.float32)
        cost, y = np.zeros((N, max(n)), np.float32), np.zeros((N, max(n)), bool)
        RX, PX, cx = np.full((N, X, M), -1, np.int64), np.zeros((N, X, F), np.float32), np.zeros((N, X), np.float32)
        for i, c in enumerate(C):
            f = prep.files[c["key"]]
            R[i, :n[i]], P[i, :n[i]], cost[i, :n[i]], y[i, :n[i]] = f["rows"], c["phi"], f["cost"], c["y"]
            for j, (kk, si, ph) in enumerate(cross[i]):
                RX[i, j], PX[i, j], cx[i, j] = prep.files[kk]["rows"][si], ph, prep.files[kk]["cost"][si]
        self.n, self.nx = T(n), T([len(x) for x in cross])
        self.R, self.P, self.cost, self.y = T(R), T(P), T(cost), T(y)
        self.RX, self.PX, self.cx = T(RX), T(PX), T(cx)

    def E(self, R):
        return self.emb_sym[R.clamp(min=0)] * (R >= 0).unsqueeze(-1).float()

    def views(self, ix, single=False):
        e = lambda r: self.emb[r.clamp(min=0)] * (r >= 0).unsqueeze(-1).float()  # noqa: E731
        eqh = e(self.qh[ix])
        if self.qvec is not None:
            qv = self.qv[ix]
            live = qv >= 0
            eqh = torch.where(live[:, None], self.qvec[qv.clamp(min=0)], eqh)
        return dict(eq=e(self.qh[ix] if single else self.q[ix]), eh=e(self.h[ix]), ep=self.ep[ix], dh=self.dh[ix],
                    dp=self.dp[ix], H=self.H[ix], eqh=eqh)

    def infile(self, ix):
        """All in-file symbols of the cases, padded to the batch's largest file: R, valid."""
        n = self.n[ix]
        cm = int(n.max())
        return self.R[ix, :cm], torch.arange(cm, device=DEV)[None] < n[:, None]

    def train_batch(self, ix, k_neg, use_cross, gen):
        """Positives + k_neg random in-file negatives (+ cross-file negatives), as in the per-case version."""
        R, valid = self.infile(ix)
        cm = R.shape[1]
        y = self.y[ix, :cm]
        npos = y.sum(1)
        k_in = (valid.sum(1) - npos).clamp(max=k_neg)
        key = torch.rand(y.shape, generator=gen, device=DEV) + 2.0 * y.float()
        width = int((npos + k_in).max())
        cols = key.masked_fill(~valid, -1.0).topk(width, 1).indices          # positives first, then random negatives
        keep = torch.arange(width, device=DEV)[None] < (npos + k_in)[:, None]
        g3 = cols[..., None].expand(-1, -1, M)
        R = torch.gather(R, 1, g3)
        P = torch.gather(self.P[ix, :cm], 1, cols[..., None].expand(-1, -1, self.P.shape[-1]))
        cost = torch.gather(self.cost[ix, :cm], 1, cols)
        pos = torch.gather(y, 1, cols) & keep
        X = self.RX.shape[1]
        xkeep = torch.arange(X, device=DEV)[None] < torch.minimum(self.nx[ix], k_in.clamp(min=1))[:, None]
        xkeep = xkeep & use_cross
        return dict(R=torch.cat([R, self.RX[ix]], 1), P=torch.cat([P, self.PX[ix]], 1),
                    cost=torch.cat([cost, self.cx[ix]], 1), pos=torch.cat([pos, torch.zeros_like(xkeep)], 1),
                    sel=torch.cat([keep, xkeep], 1))


# ---------------------------------------------------------------- training

def mine_cross(prep, cases, emb, k=8):
    """Per case: the k symbols of other files in the same repo closest to its positives (frozen embeddings)."""
    sym_mean = {}
    for key, f in prep.files.items():
        r = torch.as_tensor(f["rows"], device=DEV)
        e = emb[r.clamp(min=0)] * (r >= 0).unsqueeze(-1).float()
        sym_mean[key] = Fn.normalize(e.sum(1), dim=-1)
    repo_files = {}
    for c in cases:
        repo_files.setdefault(c["c"]["repo"], set()).add(c["key"])
    out = []
    for c in cases:
        others = [kk for kk in repo_files[c["c"]["repo"]] if kk != c["key"]]
        if not others:
            others = [kk for kk in list(prep.files)[:200] if kk != c["key"]]
        pool_keys, pool_vecs = [], []
        for kk in others[:60]:
            pool_keys += [(kk, i) for i in range(len(prep.files[kk]["syms"]))]
            pool_vecs.append(sym_mean[kk])
        if not pool_keys:
            out.append([])
            continue
        P = torch.cat(pool_vecs)
        pos = sym_mean[c["key"]][torch.as_tensor(np.where(c["y"])[0], device=DEV)]
        sim = (pos @ P.t()).max(0).values
        top = torch.topk(sim, min(k, len(pool_keys))).indices.tolist()
        picks = []
        for t in top:
            kk, i = pool_keys[t]
            f = prep.files[kk]
            ph, _ = featurize(f["syms"], f["lines"], c["query"], changed=(), asked=(), trail=c["trail"])
            picks.append((kk, i, ph[i]))
        out.append(picks)
    return out


def train(pack, tr, va, cfg, seed, log=print):
    torch.manual_seed(seed), np.random.seed(seed), random.seed(seed)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    allphi = np.concatenate([c["phi"] for c in tr])
    model = make_head(cfg, len(FEATURES), pack.emb.shape[1]).to(DEV)
    model.phi_mu = torch.as_tensor(allphi.mean(0), device=DEV)
    model.phi_sd = torch.as_tensor(allphi.std(0) + 1e-6, device=DEV)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best = evaluate_rank(scorer(model, pack, va, cfg), va)["mrr"]     # step 0 competes too (zero-shot for Bilinear)
    best_state, bad = {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
    idx = np.array([pack.ix[id(c)] for c in tr])
    for ep in range(cfg["epochs"]):
        model.train()
        rng.shuffle(idx)
        for i in range(0, len(idx), cfg["bs"]):
            loss = batch_loss(model, pack, torch.as_tensor(idx[i:i + cfg["bs"]], device=DEV), cfg, gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = evaluate_rank(scorer(model, pack, va, cfg), va)["mrr"]
        if cfg.get("verbose"):
            log(f"  epoch {ep + 1}: val MRR {m:.3f}")
        if m > best + 1e-4:
            best, bad = m, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best


def batch_loss(model, pack, ix, cfg, gen):
    """Cost-weighted margin InfoNCE, vectorised: each positive against the case's selected negatives."""
    b = pack.train_batch(ix, cfg["k_neg"], cfg["cross"], gen)
    sel, pos = b["sel"], b["pos"]
    P = (b["P"] - model.phi_mu) / model.phi_sd * sel[..., None]
    S, _ = model(pack.views(ix, cfg["views"] == "single"), pack.E(b["R"]), (b["R"] >= 0).float(), P)
    tau = cfg["tau"]
    neg = (S + cfg["alpha"] * torch.log1p(b["cost"] / cfg["c0"])) / tau
    lse = torch.logsumexp(neg.masked_fill(~(sel & ~pos), -1e9), 1, keepdim=True)
    lp = S / tau
    per = (torch.logaddexp(lp, lse) - lp) * (1 + cfg["beta"] * torch.log1p(b["cost"]))
    per = per.masked_fill(~pos, 0).sum(1) / pos.sum(1).clamp(min=1)
    w = pack.w[ix]
    return (w * per).sum() / w.sum().clamp(min=1e-6)


@torch.no_grad()
def scorer(model, pack, cases, cfg, bs=16):
    """Score many cases in batches; returns a per-case lookup usable as a ranker."""
    model.eval()
    out = {}
    for i in range(0, len(cases), bs):
        cs = cases[i:i + bs]
        ix = torch.as_tensor([pack.ix[id(c)] for c in cs], device=DEV)
        R, valid = pack.infile(ix)
        P = (pack.P[ix, :R.shape[1]] - model.phi_mu) / model.phi_sd * valid[..., None]
        S, _ = model(pack.views(ix, cfg["views"] == "single"), pack.E(R), (R >= 0).float(), P)
        for c, s in zip(cs, S):
            out[id(c)] = s[:len(c["y"])].float().cpu().numpy()
    model.train()
    return lambda c: out[id(c)]


# ---------------------------------------------------------------- evaluation

def evaluate_rank(score, cases):
    mrr, top1, rec, anyhit = [], [], [], []
    for c in cases:
        s = np.asarray(score(c), float) + 1e-9 * np.random.default_rng(len(c["y"])).random(len(c["y"]))
        order = np.argsort(-s)
        y = c["y"]
        f = None
        mrr.append(1 / (1 + int(np.where(y[order])[0][0])))
        top1.append(bool(y[order[0]]))
        key = c["key"]
        f = PREP.files[key]
        budget = max(400, 0.25 * f["file_tok"])
        used, shown = 0, set()
        for j in order:
            if used + f["cost"][j] <= budget:
                shown.add(j)
                used += f["cost"][j]
        need = set(np.where(y)[0])
        rec.append(len(need & shown) / len(need))
        anyhit.append(bool(need & shown))
    return dict(mrr=float(np.mean(mrr)), top1=float(np.mean(top1)), rec=float(np.mean(rec)),
                anyhit=float(np.mean(anyhit)), n=len(cases))


def platt(scores, cases):
    """Fit p = sigmoid(w . calib_features(s) + b) on validation symbols (see cortexnet/coproc.py). The ranking inside
    a read depends only on w[0], the weight of s - max s, so w[0] must come out positive. A fit on raw scores pooled
    across files came out negative on 2026-10-27 (a = -0.71) and reversed every read's order."""
    from sklearn.linear_model import LogisticRegression
    x = np.concatenate([calib_features(scores(c)) for c in cases])
    y = np.concatenate([c["y"] for c in cases])
    lr = LogisticRegression(C=10.0).fit(x, y)
    f = lambda s: lr.predict_proba(calib_features(s))[:, 1]  # noqa: E731
    f.ab = tuple(float(v) for v in lr.coef_[0]) + (float(lr.intercept_[0]),)
    if f.ab[0] <= 0:
        print(f"WARNING: calibration weight on s - max s is {f.ab[0]:.3f} <= 0: it would reverse the ranking",
              flush=True)
    return f


def case_cost(f, y, inc, turn, m=1.0):
    """Tokens put into context cost m each (m = 1 + alpha_cache H_rem: paid once, then re-read from the prefix cache
    on every later call); a needed symbol that was not shown costs one extra round trip (turn) plus its body."""
    need = np.where(y)[0]
    miss = [j for j in need if j not in inc]
    ctx = f["out_tok"] + sum(f["cost"][j] for j in inc) + sum(f["cost"][j] for j in miss)
    return m * ctx + (turn if miss else 0)


def evaluate_cost(score, prob, cases, regime):
    """Realised tokens under several policies, summed over cases. regime(c) -> (extra round trip cost, m)."""
    T = dict(whole=0.0, outline_only=0.0, budget_fill=0.0, rule_ab=0.0, oracle=0.0)
    bypass = 0
    for c in cases:
        f, y = PREP.files[c["key"]], c["y"]
        turn, m = regime(c)
        s = np.asarray(score(c), float)
        T["whole"] += m * f["file_tok"]
        T["outline_only"] += case_cost(f, y, set(), turn, m)
        order, budget, used, inc = np.argsort(-s), max(400, 0.25 * f["file_tok"]), 0, set()
        for j in order:
            if used + f["cost"][j] <= budget:
                inc.add(j)
                used += f["cost"][j]
        T["budget_fill"] += case_cost(f, y, inc, turn, m)
        I = rule_ab(f, prob(s), turn / m)     # every context token scales by m, so only turn / m matters
        if I is None:
            bypass += 1
            T["rule_ab"] += m * f["file_tok"]
        else:
            T["rule_ab"] += case_cost(f, y, I, turn, m)
        T["oracle"] += min(m * f["file_tok"], case_cost(f, y, set(np.where(y)[0]), turn, m))
    T["bypass_share"] = bypass / max(1, len(cases))
    return T


PREP = None


def per_case_rr(score, cases):
    out = []
    for c in cases:
        sc = np.asarray(score(c), float) + 1e-9 * np.random.default_rng(len(c["y"])).random(len(c["y"]))
        out.append(1 / (1 + int(np.where(c["y"][np.argsort(-sc)])[0][0])))
    return np.array(out)


def boot_diff(a, b, n=4000, seed=0):
    """Paired bootstrap over cases: mean(a - b) and a 95% interval."""
    rng = np.random.default_rng(seed)
    d = a - b
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def folds_by_session(C, k):
    """Sessions -> k folds, balancing read-time case counts (largest sessions first)."""
    cnt = {}
    for c in C:
        if c["c"]["source"] == "sessions" and c["c"].get("views_at") != "edit":
            cnt[c["c"]["sid"]] = cnt.get(c["c"]["sid"], 0) + 1
    load, fold = [0] * k, {}
    for sid, n in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
        j = int(np.argmin(load))
        fold[sid] = j
        load[j] += n
    return fold


def make_rankers(tr, emb, emb_sym=None):
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([c["phi"] for c in tr])
    Y = np.concatenate([c["y"] for c in tr])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    v1 = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, Y)

    def dense(c):
        f = PREP.files[c["key"]]
        r = torch.as_tensor(f["rows"], device=DEV)
        e = (emb if emb_sym is None else emb_sym)[r.clamp(min=0)] * (r >= 0).unsqueeze(-1).float()
        qv = emb[c["qh"]] if c["qh"] >= 0 else torch.zeros(emb.shape[1], device=DEV)
        return (e @ qv).max(1).values.cpu().numpy()
    return {"random": lambda c: np.random.default_rng(len(c["y"]) * 7 + int(c["y"].sum())).random(len(c["y"])),
            "smallest first": lambda c: -PREP.files[c["key"]]["cost"],
            "lexical (name + identifier overlap with prompt/snapshot)": lambda c: c["phi"][:, 0] * 2 + c["phi"][:, 1],
            "v1 include features, logistic": lambda c: v1.decision_function((c["phi"] - mu) / sd),
            "frozen zero-shot (max sub-span cosine)": dense}


def remaining_calls(sids):
    """sid -> {tool step: H_rem}, the API calls after the read that re-read its result from the prefix cache, up to
    the session's end or next compaction. Steps are counted exactly as in subnn_data.py (one per tool_use block);
    API calls by distinct message id. Cached in DATA; sessions whose log is gone are left out."""
    path = os.path.join(DATA, "hrem.json")
    if os.path.exists(path):
        return {s: {int(k): v for k, v in d.items()} for s, d in json.load(open(path)).items()}
    out = {}
    files = glob.glob(os.path.join(LOGS, "*", "*.jsonl")) + glob.glob(os.path.join(LOGS, "*", "*", "subagents", "*.jsonl"))
    for f in files:
        sid = os.path.splitext(os.path.basename(f))[0]
        if sid not in sids:
            continue
        step, calls, ids, at, cuts = 0, 0, set(), {}, []
        for line in open(f, encoding="utf8", errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("isCompactSummary") or r.get("subtype") == "compact_boundary":
                cuts.append(calls)
            msg = r.get("message") or {}
            c = msg.get("content")
            if r.get("type") != "assistant" or not isinstance(c, list):
                continue
            mid = msg.get("id") or r.get("uuid")
            if mid not in ids:
                ids.add(mid)
                calls += 1
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    step += 1
                    at[step] = calls
        # the result enters context on call k + 1 (paid in full) and is re-read on every later call before the cut
        out[sid] = {s: max(0, min([x for x in cuts if x > k] or [calls]) - k - 1) for s, k in at.items()}
    json.dump(out, open(path, "w"))
    return out


def prefix_cache(compound):
    """Extra round trip = alpha L_hist (cached prefix) + TOOL_OVH; context tokens cost 1 + alpha H_rem if compound."""
    def f(c):
        m = 1 + ALPHA_CACHE * c["hrem"] if compound else 1.0
        return ALPHA_CACHE * c["c"]["c_turn"] + TOOL_OVH, m
    return f


REGIMES = {"C_turn = 500 (no cache accounting; ~ local KV cache)": lambda c: (500, 1.0),
           "C_turn = 2,000": lambda c: (2000, 1.0),
           "previous: C_turn = logged context x 0.12, no compounding": lambda c: (0.12 * c["c"]["c_turn"], 1.0),
           f"prefix cache, no compounding: T = {ALPHA_CACHE} L_hist + {TOOL_OVH}": prefix_cache(False),
           f"prefix cache + compounding: T = {ALPHA_CACHE} L_hist + {TOOL_OVH}, context tokens x (1 + {ALPHA_CACHE} "
           f"H_rem)": prefix_cache(True)}
COST_RANKERS = ["full Cortex-SubNN", "v1 include features, logistic", "frozen zero-shot (max sub-span cosine)",
                "smallest first"]


def main():
    global PREP, DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3, help="seeds for the secondary time-split run")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ens", type=int, default=3, help="seeds averaged per config and fold")
    ap.add_argument("--quick", action="store_true", help="full model only, 2 folds")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--emb", default="minilm", help="'minilm' or a precomputed embedding file (see frozen_inputs)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "subnn.md"))
    ap.add_argument("--codec", type=int, default=0, help="store symbol vectors as BYTES-byte bit-plane codes")
    ap.add_argument("--configs", default="all", choices=("all", "full"), help="full: the full model only")
    ap.add_argument("--export", default="", help="also train one model on all cases and save it for coproc.py")
    ap.add_argument("--qvec", default="", help="npy of live decision states, rows indexed by a case's qv")
    ap.add_argument("--dump", default="", help="npz path for the out-of-fold per-symbol scores of every ranker")
    a = ap.parse_args()
    DEV = a.device
    t0 = time.time()
    cases, texts = load()
    PREP = Prep(cases, texts)
    C = PREP.case
    emb, cross_all, enc_name, cached = frozen_inputs(C, a.emb)
    print(f"{len(C)} cases, {emb.shape[0]} texts x {emb.shape[1]} ({enc_name}, {'cached' if cached else 'computed'}, "
          f"{time.time() - t0:.0f}s)", flush=True)
    emb_sym, codec = None, None
    if a.codec:                 # symbols as codes, decoded; queries stay unquantised (asymmetric, as in wormhole)
        from cortexnet.codec import BitplaneCodec
        rows = np.unique(np.concatenate([f["rows"].ravel() for f in PREP.files.values()]))
        rows = torch.as_tensor(rows[rows >= 0])
        X = emb[rows].cpu().numpy()
        codec = BitplaneCodec.fit(X, code_bytes=a.codec)
        emb_sym = emb.clone()
        emb_sym[rows] = torch.as_tensor(codec.decode(codec.encode(X)), device=emb.device)
        cos = float((emb_sym[rows] * emb[rows]).sum(1).mean())
        enc_name += f", symbols as {codec.code_bytes}-byte codes ({codec.k} coords, planes {codec.width}, " \
                    f"mean decoded cosine {cos:.3f})"
        print(enc_name, flush=True)
    hrem = remaining_calls({c["c"]["sid"] for c in C if c["c"]["source"] == "sessions"})
    known = [hrem[c["c"]["sid"]][c["c"]["step"]] for c in C
             if c["c"]["sid"] in hrem and c["c"]["step"] in hrem[c["c"]["sid"]]]
    h_med = float(np.median(known)) if known else 0.0
    for c in C:
        c["hrem"] = hrem.get(c["c"]["sid"], {}).get(c["c"]["step"], h_med)
    pack = Pack(PREP, C, emb, cross_all, emb_sym)
    if a.qvec:                  # centre and normalise, as the symbol-side Glimmer vectors were
        qv = torch.as_tensor(np.load(a.qvec), dtype=torch.float32)
        if qv.shape[1] != emb.shape[1]:
            raise SystemExit(f"--qvec has {qv.shape[1]} dims, the encoder {emb.shape[1]}")
        pack.qvec = Fn.normalize(qv - qv.mean(0, keepdim=True), dim=-1).to(DEV)
        enc_name += f", query = live decision state ({os.path.basename(a.qvec)})"
    tag = "".join(ch if ch.isalnum() else "_" for ch in enc_name.split(",")[0]) + (f"_c{a.codec}" if a.codec else "")
    print(f"packed; H_rem known for {len(known)} cases (median {h_med:.0f}) ({time.time() - t0:.0f}s)", flush=True)

    base = dict(head="bilinear", rank=16, diag=True, lowrank=True, views="multi", gate=True, late=True, struct=True,
                quant=True, cos_skip=True, cross=False, alpha=0.05, beta=0.2, c0=100.0, tau=0.1, lr=2e-3, bs=32,
                k_neg=12, epochs=a.epochs, patience=6)
    # cross-file negatives off by default: they cost 0.024-0.047 MRR for every encoder in the 2026-10-27 grid
    configs = {"full Cortex-SubNN": {},
               "- gate (uniform over present views)": dict(gate=False),
               "- multi-view (prompt only)": dict(views="q"),
               "- multi-view (single stream prompt+snapshot)": dict(views="single"),
               "- late interaction (M = 1, mean pooled)": dict(late=False),
               "- cost weighting (alpha = beta = 0)": dict(alpha=0.0, beta=0.0),
               "- structural prior": dict(struct=False),
               "- diagonal re-weighting (low-rank only)": dict(diag=False),
               "- low-rank residual (diagonal only)": dict(lowrank=False),
               "+ cross-file negatives": dict(cross=True),
               "v2 bottleneck head (k = 32, int8, cosine skip)": dict(head="bottleneck")}
    K = a.folds
    if a.quick:
        configs, K = {"full Cortex-SubNN": {}}, 2
    elif a.configs == "full":
        configs = {"full Cortex-SubNN": {}}
    fold = folds_by_session(C, K)
    ses = lambda c: c["c"]["source"] == "sessions"  # noqa: E731
    real = lambda c: ses(c) and c["c"].get("views_at") != "edit"  # noqa: E731
    pooled, per_fold = {}, {}
    cost = {n: {rg: None for rg in REGIMES} for n in COST_RANKERS}
    gate_pi, gate_h, test_all, late_all, per_seed = [], [], [], [], {}
    for k in range(K):
        te = [c for c in C if real(c) and fold.get(c["c"]["sid"]) == k]
        le = [c for c in C if ses(c) and not real(c) and fold.get(c["c"]["sid"], -1) == k]
        train_sids = sorted({c["c"]["sid"] for c in C if ses(c) and fold.get(c["c"]["sid"], -1) != k})
        rng = np.random.default_rng(100 + k)
        val_sids = set(rng.choice(train_sids, max(1, int(0.15 * len(train_sids))), replace=False))
        va = [c for c in C if real(c) and c["c"]["sid"] in val_sids]
        tr = [c for c in C if not ses(c) or (fold.get(c["c"]["sid"], -1) != k and c["c"]["sid"] not in val_sids)]
        test_all += te
        late_all += le
        print(f"fold {k + 1}/{K}: train {len(tr)}, val {len(va)}, test {len(te)}, edit-time {len(le)}", flush=True)
        rk = make_rankers(tr, emb, emb_sym)
        for name, f in rk.items():
            for c in te + le:
                pooled.setdefault(name, {})[id(c)] = f(c)
            per_fold.setdefault(name, []).append(evaluate_rank(f, te)["mrr"])
        full = full_sc = None
        for name, ov in configs.items():
            cfg = dict(base, **ov)
            models, vbs = [], []
            for s_ in range(a.ens):                 # seed ensemble: one run is too noisy to compare configs
                m, vb = train(pack, tr, va, cfg, 10 * k + s_)
                models.append(m)
                vbs.append(vb)
            looks = [scorer(m, pack, va + te + le, cfg) for m in models]
            ens = lambda c, _l=looks: np.mean([f(c) for f in _l], 0)  # noqa: E731
            for c in te + le:
                pooled.setdefault(name, {})[id(c)] = ens(c)
            for s_, f in enumerate(looks):
                per_seed.setdefault(name, {}).setdefault(s_, {}).update({id(c): f(c) for c in te})
            per_fold.setdefault(name, []).append(evaluate_rank(ens, te)["mrr"])
            if name == "full Cortex-SubNN":
                full, full_sc = (models[0], cfg), ens
            print(f"  {name}: fold MRR {per_fold[name][-1]:.3f} (val {np.mean(vbs):.3f}) ({time.time() - t0:.0f}s)",
                  flush=True)
        m, cfg = full
        with torch.no_grad():
            ix = torch.as_tensor([pack.ix[id(c)] for c in te], device=DEV)
            gate_pi.append(m.gate_weights(pack.views(ix, cfg["views"] == "single")).cpu().numpy())
        gate_h += [c["h"] >= 0 for c in te]
        rkc = dict(rk)
        rkc["full Cortex-SubNN"] = full_sc
        for name in COST_RANKERS:
            prob = platt(rkc[name], va)
            for rg, fn in REGIMES.items():
                T = evaluate_cost(rkc[name], prob, te, fn)
                T["bypass_n"] = T["bypass_share"] * len(te)
                cost[name][rg] = T if cost[name][rg] is None else {x: cost[name][rg][x] + T[x] for x in T}
        torch.save(m.state_dict(), os.path.join(DATA, f"subnn_full_fold{k}_{tag}.pt"))

    # secondary: the deployment-like time split (train on the past, test on the most recent sessions)
    tr_c, va_c, te_c, _ = split(cases)
    pos = {id(c): i for i, c in enumerate(cases)}
    tr = [C[pos[id(c)]] for c in tr_c]
    va = [C[pos[id(c)]] for c in va_c]
    te = [C[pos[id(c)]] for c in te_c]
    time_rows = {n: evaluate_rank(f, te)["mrr"] for n, f in make_rankers(tr, emb, emb_sym).items()}
    tm = []
    for sd_ in range(1 if a.quick else a.seeds):
        looks = [scorer(train(pack, tr, va, dict(base), 100 * sd_ + e)[0], pack, te, base) for e in range(a.ens)]
        tm.append(evaluate_rank(lambda c, _l=looks: np.mean([f(c) for f in _l], 0), te)["mrr"])
    time_rows["full Cortex-SubNN"] = tm
    test_h = [c["hrem"] for c in test_all]
    keys = np.array([f"{c['c']['sid']}|{c['c']['ts']}" for c in test_all])
    np.savez(a.out + ".rr.npz", keys=keys, **{n: per_case_rr(lambda c, _n=n: pooled[_n][id(c)], test_all)
                                               for n in pooled if set(map(id, test_all)) <= set(pooled[n])})
    if a.dump:                  # out-of-fold per-symbol scores, flat with offsets (private: keys carry session ids)
        names = [n for n in pooled if set(map(id, test_all)) <= set(pooled[n])]
        off = np.cumsum([0] + [len(c["y"]) for c in test_all])
        np.savez(a.dump, names=np.array(names), keys=keys, off=off,
                 sid=np.array([c["c"]["sid"] for c in test_all]), step=np.array([c["c"]["step"] for c in test_all]),
                 fkey=np.array([str(c["key"]) for c in test_all]), y=np.concatenate([c["y"] for c in test_all]),
                 **{f"s{i}": np.concatenate([np.asarray(pooled[n][id(c)], np.float32) for c in test_all])
                    for i, n in enumerate(names)})
    write_report(pooled, per_fold, per_seed, cost, np.concatenate(gate_pi), np.array(gate_h), test_all, late_all,
                 time_rows, len(te), K, t0, enc_name, emb.shape[1], (len(known), h_med, np.median(test_h)), a.out)
    if a.export:
        export_head(pack, C, dict(base), a.export, enc_name, codec)


def export_head(pack, C, cfg, path, enc_name, codec):
    """One model on every case (15% of the sessions held out for early stopping and Platt), saved as numpy arrays
    for cortexnet/coproc.py, together with the codec when symbols were coded."""
    real = [c for c in C if c["c"]["source"] == "sessions" and c["c"].get("views_at") != "edit"]
    sids = sorted({c["c"]["sid"] for c in real})
    val_sids = set(np.random.default_rng(7).choice(sids, max(1, int(0.15 * len(sids))), replace=False))
    va = [c for c in real if c["c"]["sid"] in val_sids]
    tr = [c for c in C if c["c"]["sid"] not in val_sids]
    m, vb = train(pack, tr, va, cfg, 0)
    cal = platt(scorer(m, pack, va, cfg), va).ab              # (w_rel, w_max, w_logn, b)
    if cal[0] <= 0:
        raise SystemExit(f"not exporting: calibration weight on s - max s is {cal[0]:.3f} <= 0")
    t = lambda x: x.detach().float().cpu().numpy()  # noqa: E731
    out = dict(g=t(m.g), dg=t(m.dg), A=t(m.A), B=t(m.B), Wg_w=t(m.Wg.weight), Wg_b=t(m.Wg.bias), ws_w=t(m.ws.weight),
               ws_b=t(m.ws.bias), phi_mu=t(m.phi_mu), phi_sd=t(m.phi_sd), calib=np.array(cal, np.float32),
               flags=np.array([cfg["views"] == "multi", cfg["gate"], cfg["late"], cfg["struct"], cfg["diag"],
                               cfg["lowrank"]]), gamma=np.float32(GAMMA), features=np.array(FEATURES),
               encoder=np.array(enc_name.split(",")[0]), val_mrr=np.float32(vb))
    if codec is not None:
        out.update(codec_V=codec.V, codec_bits=codec.bits, codec_levels=codec.levels)
    np.savez(path, **out)
    print(f"exported head to {path} (validation MRR {vb:.3f}, calibration w_rel {cal[0]:.2f} w_max {cal[1]:.2f} "
          f"w_logn {cal[2]:.2f} b {cal[3]:.2f})", flush=True)


def write_report(pooled, per_fold, per_seed, cost, pi, has_h, te, le, time_rows, n_time, K, t0, enc_name, dim,
                 hrem_info, out):
    def row(name, cases):
        return evaluate_rank(lambda c: pooled[name][id(c)], cases)
    te_ids = {id(c) for c in te}
    learned = [n for n in pooled if n.startswith(("full", "-", "+", "v2")) and te_ids <= set(pooled[n])]
    base = [n for n in pooled if n not in learned]
    L = ["# Cortex-SubNN: which symbol bodies to show up front", "",
         f"Symbol-level cases from 91 local agent sessions plus git history (private data; only these aggregates are "
         f"kept). Frozen encoder: {enc_name} ({dim} dims). {K}-fold cross-validation grouped by session: every "
         f"read-time case is tested once, by a model that never saw its session ({len(te)} cases). Early stopping "
         f"(which also scores the untrained step-0 model) and calibration use 15% of the training sessions. "
         f"Rankings are within the file that was read (median ~20 leaf symbols); budget = the Reader's include budget, "
         f"max(400, 25% of file tokens). ({time.time() - t0:.0f}s)", "",
         "## Ranking (pooled over folds)", "",
         "Learned rows are the mean of a seed ensemble per fold; the single-seed column shows how much one run varies. "
         "The full model starts as the frozen zero-shot ranker exactly (identity init).",
         "", "| ranker | MRR | MRR per fold (mean +- sd) | single seed MRR (mean +- sd) | top-1 | recall in budget | "
         ">= 1 needed shown |", "|---|---|---|---|---|---|---|"]
    for name in base + learned:
        t = row(name, te)
        pf = per_fold[name]
        if name in per_seed:
            ss = [evaluate_rank(lambda c, _d=d: _d[id(c)], te)["mrr"] for d in per_seed[name].values()]
            ss = f"{np.mean(ss):.3f} +- {np.std(ss):.3f}"
        else:
            ss = "-"
        L.append(f"| {name} | {t['mrr']:.3f} | {np.mean(pf):.3f} +- {np.std(pf):.3f} | {ss} | {t['top1']:.3f} | "
                 f"{t['rec']:.3f} | {t['anyhit']:.3f} |")
    rr = {n: per_case_rr(lambda c, _n=n: pooled[_n][id(c)], te) for n in base + learned}
    L += ["", "Paired bootstrap over cases (MRR difference, 95% interval):", ""]
    for other in ("frozen zero-shot (max sub-span cosine)", "v1 include features, logistic", "random"):
        d, lo, hi = boot_diff(rr["full Cortex-SubNN"], rr[other])
        L.append(f"- full Cortex-SubNN vs {other}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for n in learned:
        if n != "full Cortex-SubNN":
            d, lo, hi = boot_diff(rr["full Cortex-SubNN"], rr[n])
            L.append(f"- full vs {n}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    snap = [c for c in te if c["h"] >= 0]
    nosnap = [c for c in te if c["h"] < 0]
    L += ["", "## By snapshot presence (MRR)", "", f"| ranker | with snapshot ({len(snap)}) | without ({len(nosnap)}) |",
          "|---|---|---|"]
    for name in base + learned:
        L.append(f"| {name} | {row(name, snap)['mrr']:.3f} | {row(name, nosnap)['mrr']:.3f} |")
    if le:
        L += ["", f"## Edit-time cases (no read before the edit; views taken at the edit, so optimistic): {len(le)}", "",
              "| ranker | MRR | top-1 |", "|---|---|---|"]
        for name in base + learned[:1]:
            t = row(name, le)
            L.append(f"| {name} | {t['mrr']:.3f} | {t['top1']:.3f} |")
    n_known, h_all, h_test = hrem_info
    L += ["", "## Gate (full model, held-out cases)", "",
          f"Mean weight on (prompt, snapshot, trail): with a snapshot {np.round(pi[has_h].mean(0), 2).tolist()}, "
          f"without {np.round(pi[~has_h].mean(0), 2).tolist()}.", "",
          "## Decision head: realised tokens on held-out cases (whole-file reads = 100%; lower is better)", "",
          "Each ranker is Platt-calibrated per fold on the validation sessions. A case costs the outline + bodies shown "
          "up front, plus one extra round trip and the missing bodies if a needed symbol was not shown. Rule A returns "
          "the whole file when that is cheaper in expectation; Rule B picks the up-front set; budget fill is the "
          "Reader's current policy with that ranker. Prefix-cache regimes: an extra round trip re-reads the cached "
          f"history at {ALPHA_CACHE}x (L_hist = logged context at the read) plus {TOOL_OVH} tokens of tool-step "
          f"overhead; with compounding, every token put into context also costs {ALPHA_CACHE}x on each later API call "
          f"until the session ends or is compacted (H_rem from the session logs for {n_known} cases, median "
          f"{h_all:.0f}; median on test cases {h_test:.0f}; missing logs use the median). Cache writes are priced at "
          "1x, not 1.25x.", ""]
    for rg in REGIMES:
        L += [f"### {rg}", "", "| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |",
              "|---|---|---|---|---|---|"]
        for name in COST_RANKERS:
            T = cost[name][rg]
            w = T["whole"]
            L.append(f"| {name} | {T['outline_only'] / w:.0%} | {T['budget_fill'] / w:.0%} | {T['rule_ab'] / w:.0%} | "
                     f"{T['bypass_n'] / len(te):.0%} | {T['oracle'] / w:.0%} |")
        L.append("")
    L += [f"## Secondary: time split (train on the past, test on the most recent {n_time} read-time cases)", "",
          "| ranker | MRR |", "|---|---|"]
    for n, v in time_rows.items():
        L.append(f"| {n} | " + (f"{np.mean(v):.3f} +- {np.std(v):.3f} ({len(v)} seeds)" if isinstance(v, list)
                                 else f"{v:.3f}") + " |")
    open(out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
