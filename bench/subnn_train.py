"""Cortex-SubNN: a small coprocessor that picks which symbol bodies to show up front, trained on the agent's own use.

Data: bench/subnn_data.py (private, outside the repo). Only aggregate numbers are written to bench/results/.

Model (one forward per read):
  views      e_q (user prompt), e_h (reasoning snapshot, presence flag delta, length/entropy H), e_prev (decayed trail of
             symbols used earlier in the session); frozen MiniLM embeddings, optionally LoRA on the top 2 layers
  gate       pi = softmax(W_g [e_q; delta e_h; e_prev; delta; delta_p; H] + log-mask), u = sum_i pi_i W_i e_i  (m dims)
  bottleneck z = norm(W_down GELU(LN(u)) + P_skip u), k dims, int8 straight-through
  symbols    M = 4 sub-span prototypes per symbol (signature + 3 body windows), own projection to k dims, int8 STE
  score      S = tau_s logsumexp_r(z.v_r / tau_s) + w_struct . phi(s) + w_c . [max_r cos(e_view, e_s,r)]
             (phi = the v1 include model's features; the last term is a skip to the frozen encoder's geometry)
  loss       cost-weighted margin InfoNCE: in-file siblings + nearest cross-file symbols of the same repo as negatives,
             margin alpha log(1 + c(s-)/c0) on negatives, weight 1 + beta log(1 + c(s+)) on positives
  decision   Platt calibration on the validation window, then Rule A (whole-file bypass) and Rule B (greedy up-front set)
             against an extra round trip of C_turn tokens

Evaluation: k-fold cross-validation grouped by session (each read-time case tested once by a model that never saw its
session; early stopping and calibration on 15% of the training sessions), plus a time split as a secondary check.
Training data = the other sessions' cases + git cases (down-weighted). Edit-time cases (no read before the edit) are
trained on but reported separately, never mixed into the main test numbers.
Usage: python -I bench/subnn_train.py [--folds 5] [--seeds 3] [--epochs 40] [--lora] [--lora-epochs 3] [--quick]
"""
import argparse
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

DATA = "D:/dev-cache/cortexnet/subnn"
HERE = os.path.dirname(os.path.abspath(__file__))
ENC = "sentence-transformers/all-MiniLM-L6-v2"
SPLIT, M, GAMMA, MAXLEN = 600, 4, 0.6, 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- data

def leaves(text, path):
    return [s for s in symbols(text, path, split_tokens=SPLIT) if not s.children]


def subspans(s, lines, path):
    """M texts per symbol: signature/doc, then three body windows."""
    head = f"{os.path.basename(path)}: {s.head} {s.doc}".strip()
    body = lines[s.start - 1:s.end]
    n = len(body)
    out = [head]
    for j in range(M - 1):
        a, b = j * n // (M - 1), (j + 1) * n // (M - 1)
        w = "\n".join(body[a:max(b, a + 1)]).strip()
        out.append(w[:700] if w else "")
    return out


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
    """MiniLM mean-pool + L2 norm (same as sentence-transformers), with optional LoRA on the top layers' Q and V."""

    def __init__(self, lora_layers=0, r=8, alpha=16):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(ENC)
        self.m = AutoModel.from_pretrained(ENC)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.lora = nn.ModuleList()
        layers = self.m.encoder.layer
        for li in range(len(layers) - lora_layers, len(layers)):
            att = layers[li].attention.self
            for name in ("query", "value"):
                lin = getattr(att, name)
                wrap = LoRA(lin, r, alpha)
                setattr(att, name, wrap)
                self.lora.append(wrap)

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


class LoRA(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x.to(self.A.dtype) @ self.A.t() @ self.B.t()).to(x.dtype) * self.scale


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


# ---------------------------------------------------------------- batching

class Batcher:
    def __init__(self, prep, emb, phi_mu, phi_sd):
        self.p, self.emb, self.mu, self.sd = prep, emb, phi_mu, phi_sd
        self.zero = torch.zeros(1, emb.shape[1], device=DEV)

    def rows(self, ids):
        ids = torch.as_tensor(np.asarray(ids), device=DEV)
        out = self.emb[ids.clamp(min=0)]
        return out * (ids >= 0).unsqueeze(-1).float()

    def views(self, cs, single=False):
        eq = self.rows([c["qh" if single else "q"] for c in cs])
        eh = self.rows([c["h"] for c in cs])
        dh = torch.tensor([float(c["h"] >= 0) for c in cs], device=DEV)
        ep, dp = [], []
        for c in cs:
            pr = [r for r in c["prev"] if r >= 0]
            if pr:
                w = torch.tensor([GAMMA ** j for j in range(len(pr))], device=DEV)
                e = (self.emb[torch.tensor(pr, device=DEV)] * w[:, None]).sum(0) / w.sum()
                ep.append(Fn.normalize(e, dim=-1))
            else:
                ep.append(self.zero[0])
            dp.append(float(bool(pr)))
        H = torch.tensor(np.stack([c["H"] for c in cs]), device=DEV)
        return eq, eh, torch.stack(ep), dh, torch.tensor(dp, device=DEV), H

    def cands(self, items):
        """items: per case a list of (file key, symbol index, phi row). Returns padded E, mask, phi, valid."""
        C = max(len(x) for x in items)
        R = np.full((len(items), C, M), -1, np.int64)
        P = np.zeros((len(items), C, len(FEATURES)), np.float32)
        valid = np.zeros((len(items), C), bool)
        for b, x in enumerate(items):
            for j, (key, si, ph) in enumerate(x):
                R[b, j] = self.p.files[key]["rows"][si]
                P[b, j] = (ph - self.mu) / self.sd
                valid[b, j] = True
        E = self.rows(R.reshape(-1)).reshape(len(items), C, M, -1)
        msk = torch.as_tensor(R >= 0, device=DEV).float()
        return E, msk, torch.tensor(P, device=DEV), torch.tensor(valid, device=DEV)


def infile(c):
    return [(c["key"], i, c["phi"][i]) for i in range(len(c["y"]))]


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


def train(prep, tr, va, emb, cfg, seed, cross, log=print):
    torch.manual_seed(seed), np.random.seed(seed), random.seed(seed)
    allphi = np.concatenate([c["phi"] for c in tr])
    mu, sd = allphi.mean(0), allphi.std(0) + 1e-6
    bt = Batcher(prep, emb, mu, sd)
    model = SubNN(len(FEATURES), views=cfg["views"], gate=cfg["gate"], late=cfg["late"], struct=cfg["struct"],
                  quant=cfg["quant"], cos_skip=cfg.get("cos_skip", True)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    idx = np.arange(len(tr))
    for ep in range(cfg["epochs"]):
        model.train()
        rng.shuffle(idx)
        for i in range(0, len(idx), cfg["bs"]):
            cs = [tr[j] for j in idx[i:i + cfg["bs"]]]
            items, pos_n = [], []
            for j, c in zip(idx[i:i + cfg["bs"]], cs):
                pos = list(np.where(c["y"])[0])
                neg = list(np.where(~c["y"])[0])
                k_in = min(len(neg), cfg["k_neg"])
                neg = list(rng.choice(neg, k_in, replace=False)) if k_in else []
                x = [(c["key"], int(s), c["phi"][s]) for s in pos + neg]
                if cfg["cross"]:
                    x += cross[j][:max(k_in, 1)]
                items.append(x)
                pos_n.append(len(pos))
            loss = batch_loss(model, bt, cs, items, pos_n, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = evaluate_rank(scorer(model, bt, va, cfg), va)["mrr"]
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
    return model, bt, best


def batch_loss(model, bt, cs, items, pos_n, cfg):
    eq, eh, ep, dh, dp, H = bt.views(cs, single=cfg["views"] == "single")
    z, _ = model.query(eq, eh, ep, dh, dp, H)
    E, msk, P, valid = bt.cands(items)
    V, vm = model.sym(E, msk)
    S = model.score(z, V, vm, P, (E, msk, (eq, eh, ep, dh, dp)))      # [B, C]
    cost = torch.zeros_like(S)
    for b, x in enumerate(items):
        cost[b, :len(x)] = torch.tensor([bt.p.files[k]["cost"][i] for k, i, _ in x], device=DEV)
    tau = cfg["tau"]
    total, wsum = 0.0, 0.0
    for b, npos in enumerate(pos_n):
        n = int(valid[b].sum())
        negS = S[b, npos:n] + cfg["alpha"] * torch.log1p(cost[b, npos:n] / cfg["c0"])
        for p in range(npos):
            omega = 1 + cfg["beta"] * torch.log1p(cost[b, p])
            logits = torch.cat([S[b, p:p + 1], negS]) / tau
            total = total + cs[b]["w"] * omega * (torch.logsumexp(logits, 0) - logits[0]) / npos
        wsum += cs[b]["w"]
    return total / max(wsum, 1e-6)


@torch.no_grad()
def model_scores(model, bt, cs, cfg):
    eq, eh, ep, dh, dp, H = bt.views(cs, single=cfg["views"] == "single")
    z, pi = model.query(eq, eh, ep, dh, dp, H)
    E, msk, P, valid = bt.cands([infile(c) for c in cs])
    V, vm = model.sym(E, msk)
    S = model.score(z, V, vm, P, (E, msk, (eq, eh, ep, dh, dp)))
    return [S[b, :len(c["y"])].float().cpu().numpy() for b, c in enumerate(cs)]


def scorer(model, bt, cases, cfg, bs=64):
    """Score many cases in batches; returns a per-case lookup usable as a ranker."""
    model.eval()
    out = {}
    for i in range(0, len(cases), bs):
        cs = cases[i:i + bs]
        for c, s in zip(cs, model_scores(model, bt, cs, cfg)):
            out[id(c)] = s
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
    """Fit p = sigmoid(a s + b) on validation symbols (temperature + bias)."""
    from sklearn.linear_model import LogisticRegression
    x = np.concatenate([np.asarray(scores(c), float) for c in cases])[:, None]
    y = np.concatenate([c["y"] for c in cases])
    lr = LogisticRegression(C=10.0).fit(x, y)
    return lambda s: lr.predict_proba(np.asarray(s, float)[:, None])[:, 1]


def case_cost(f, y, inc, c_turn):
    need = np.where(y)[0]
    miss = [j for j in need if j not in inc]
    return f["out_tok"] + sum(f["cost"][j] for j in inc) + (c_turn + sum(f["cost"][j] for j in miss) if miss else 0)


def rule_ab(f, p, c_turn):
    """Rule A (bypass to the whole file) then Rule B (greedy set that lowers expected cost). Returns set or None.

    Adding j changes the expected cost by c_j (1 - p_j) - C_turn * prod_out * p_j / (1 - p_j), where prod_out is the
    probability that nothing outside the set is needed; take the most negative change until none is negative.
    """
    cost = f["cost"].astype(float)
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


def evaluate_cost(score, prob, cases, c_turn_fn):
    """Realised tokens under several policies, summed over cases."""
    T = dict(whole=0.0, outline_only=0.0, budget_fill=0.0, rule_ab=0.0, oracle=0.0)
    bypass = 0
    for c in cases:
        f, y = PREP.files[c["key"]], c["y"]
        ct = c_turn_fn(c)
        s = np.asarray(score(c), float)
        T["whole"] += f["file_tok"]
        T["outline_only"] += case_cost(f, y, set(), ct)
        order, budget, used, inc = np.argsort(-s), max(400, 0.25 * f["file_tok"]), 0, set()
        for j in order:
            if used + f["cost"][j] <= budget:
                inc.add(j)
                used += f["cost"][j]
        T["budget_fill"] += case_cost(f, y, inc, ct)
        I = rule_ab(f, prob(s), ct)
        if I is None:
            bypass += 1
            T["rule_ab"] += f["file_tok"]
        else:
            T["rule_ab"] += case_cost(f, y, I, ct)
        T["oracle"] += min(f["file_tok"], case_cost(f, y, set(np.where(y)[0]), ct))
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


def make_rankers(tr, emb):
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([c["phi"] for c in tr])
    Y = np.concatenate([c["y"] for c in tr])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    v1 = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, Y)

    def dense(c):
        f = PREP.files[c["key"]]
        r = torch.as_tensor(f["rows"], device=DEV)
        e = emb[r.clamp(min=0)] * (r >= 0).unsqueeze(-1).float()
        qv = emb[c["qh"]] if c["qh"] >= 0 else torch.zeros(emb.shape[1], device=DEV)
        return (e @ qv).max(1).values.cpu().numpy()
    return {"random": lambda c: np.random.default_rng(len(c["y"]) * 7 + int(c["y"].sum())).random(len(c["y"])),
            "smallest first": lambda c: -PREP.files[c["key"]]["cost"],
            "lexical (name + identifier overlap with prompt/snapshot)": lambda c: c["phi"][:, 0] * 2 + c["phi"][:, 1],
            "v1 include features, logistic": lambda c: v1.decision_function((c["phi"] - mu) / sd),
            "frozen MiniLM zero-shot (max sub-span cosine)": dense}


REGIMES = {"C_turn = 500": lambda c: 500, "C_turn = 2,000": lambda c: 2000, "C_turn = 8,000": lambda c: 8000,
           "C_turn = logged context x 0.12 (cached price)": lambda c: 0.12 * c["c"]["c_turn"]}
COST_RANKERS = ["full Cortex-SubNN", "v1 include features, logistic", "frozen MiniLM zero-shot (max sub-span cosine)",
                "smallest first"]
LORA_NAME = "full Cortex-SubNN + LoRA (top 2 MiniLM layers, r = 8)"


def main():
    global PREP
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3, help="seeds for the secondary time-split run")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-epochs", type=int, default=3)
    ap.add_argument("--ens", type=int, default=3, help="seeds averaged per config and fold")
    ap.add_argument("--quick", action="store_true", help="full model only, 2 folds")
    a = ap.parse_args()
    t0 = time.time()
    cases, texts = load()
    PREP = Prep(cases, texts)
    C = PREP.case
    enc = Encoder().to(DEV).eval()
    with torch.no_grad():
        emb = enc(PREP.tab.texts)
    del enc
    print(f"{len(C)} cases, embedded {emb.shape[0]} texts ({time.time() - t0:.0f}s)", flush=True)
    cross_all = mine_cross(PREP, C, emb)
    print(f"cross-file negatives mined ({time.time() - t0:.0f}s)", flush=True)

    base = dict(views="multi", gate=True, late=True, struct=True, quant=True, cross=True, alpha=0.05, beta=0.2,
                c0=100.0, tau=0.1, lr=2e-3, bs=32, k_neg=12, epochs=a.epochs, patience=6)
    configs = {"full Cortex-SubNN": {},
               "- gate (uniform over present views)": dict(gate=False),
               "- multi-view (prompt only)": dict(views="q"),
               "- multi-view (single stream prompt+snapshot)": dict(views="single"),
               "- late interaction (M = 1, mean pooled)": dict(late=False),
               "- cost weighting (alpha = beta = 0)": dict(alpha=0.0, beta=0.0),
               "- structural prior": dict(struct=False),
               "- cross-file negatives": dict(cross=False),
               "- int8 (fp32 z and v)": dict(quant=False),
               "- pretrained-cosine skip": dict(cos_skip=False)}
    K = a.folds
    if a.quick:
        configs, K = {"full Cortex-SubNN": {}}, 2
    fold = folds_by_session(C, K)
    ses = lambda c: c["c"]["source"] == "sessions"  # noqa: E731
    real = lambda c: ses(c) and c["c"].get("views_at") != "edit"  # noqa: E731
    pooled, per_fold = {}, {}
    cost = {n: {rg: None for rg in REGIMES} for n in COST_RANKERS}
    gate_pi, gate_h, lora_hist, test_all, late_all, lora_jobs, per_seed = [], [], [], [], [], [], {}
    for k in range(K):
        te = [c for c in C if real(c) and fold.get(c["c"]["sid"]) == k]
        le = [c for c in C if ses(c) and not real(c) and fold.get(c["c"]["sid"], -1) == k]
        train_sids = sorted({c["c"]["sid"] for c in C if ses(c) and fold.get(c["c"]["sid"], -1) != k})
        rng = np.random.default_rng(100 + k)
        val_sids = set(rng.choice(train_sids, max(1, int(0.15 * len(train_sids))), replace=False))
        va = [c for c in C if real(c) and c["c"]["sid"] in val_sids]
        tr_i = [i for i, c in enumerate(C) if not ses(c) or (fold.get(c["c"]["sid"], -1) != k
                                                              and c["c"]["sid"] not in val_sids)]
        tr, cross = [C[i] for i in tr_i], [cross_all[i] for i in tr_i]
        test_all += te
        late_all += le
        print(f"fold {k + 1}/{K}: train {len(tr)}, val {len(va)}, test {len(te)}, edit-time {len(le)}", flush=True)
        rk = make_rankers(tr, emb)
        for name, f in rk.items():
            for c in te + le:
                pooled.setdefault(name, {})[id(c)] = f(c)
            per_fold.setdefault(name, []).append(evaluate_rank(f, te)["mrr"])
        full = full_sc = None
        for name, ov in configs.items():
            cfg = dict(base, **ov)
            models, vbs = [], []
            for s_ in range(a.ens):                 # seed ensemble: one run is too noisy to compare configs
                m, bt, vb = train(PREP, tr, va, emb, cfg, 10 * k + s_, cross)
                models.append((m, bt))
                vbs.append(vb)
            looks = [scorer(m, bt, va + te + le, cfg) for m, bt in models]
            ens = lambda c, _l=looks: np.mean([f(c) for f in _l], 0)  # noqa: E731
            for c in te + le:
                pooled.setdefault(name, {})[id(c)] = ens(c)
            for s_, f in enumerate(looks):
                per_seed.setdefault(name, {}).setdefault(s_, {}).update({id(c): f(c) for c in te})
            per_fold.setdefault(name, []).append(evaluate_rank(ens, te)["mrr"])
            if name == "full Cortex-SubNN":
                full, full_sc = (models[0][0], models[0][1], cfg), ens
            print(f"  {name}: fold MRR {per_fold[name][-1]:.3f} (val {np.mean(vbs):.3f}) ({time.time() - t0:.0f}s)",
                  flush=True)
        m, bt, cfg = full
        eq, eh, ep, dh, dp, H = bt.views(te)
        with torch.no_grad():
            m.eval()
            _, pi = m.query(eq, eh, ep, dh, dp, H)
        gate_pi.append(pi.cpu().numpy())
        gate_h += [c["h"] >= 0 for c in te]
        rkc = dict(rk)
        rkc["full Cortex-SubNN"] = full_sc
        for name in COST_RANKERS:
            prob = platt(rkc[name], va)
            for rg, fn in REGIMES.items():
                T = evaluate_cost(rkc[name], prob, te, fn)
                T["bypass_n"] = T["bypass_share"] * len(te)
                cost[name][rg] = T if cost[name][rg] is None else {x: cost[name][rg][x] + T[x] for x in T}
        torch.save(m.state_dict(), os.path.join(DATA, f"subnn_full_fold{k}.pt"))
        lora_jobs.append((k, tr, va, te, le, cross, full))

    # secondary: the deployment-like time split (train on the past, test on the most recent sessions)
    tr_c, va_c, te_c, _ = split(cases)
    pos = {id(c): i for i, c in enumerate(cases)}
    tr = [C[pos[id(c)]] for c in tr_c]
    va = [C[pos[id(c)]] for c in va_c]
    te = [C[pos[id(c)]] for c in te_c]
    cross = [cross_all[pos[id(c)]] for c in tr_c]
    time_rows = {n: evaluate_rank(f, te)["mrr"] for n, f in make_rankers(tr, emb).items()}
    tm = []
    for sd_ in range(1 if a.quick else a.seeds):
        looks = [scorer(*train(PREP, tr, va, emb, dict(base), 100 * sd_ + e, cross)[:2], te, base)
                 for e in range(a.ens)]
        tm.append(evaluate_rank(lambda c, _l=looks: np.mean([f(c) for f in _l], 0), te)["mrr"])
    time_rows["full Cortex-SubNN"] = tm
    args = (pooled, per_fold, per_seed, cost, np.concatenate(gate_pi), np.array(gate_h), test_all, late_all)
    write_report(*args, lora_hist, "", time_rows, len(te), K, t0)    # main results are safe before LoRA starts

    if a.lora:                  # GPU allocations count against Windows commit memory here; an OOM ends LoRA only
        note = ""
        torch.cuda.empty_cache()
        for k, tr, va, te_k, le_k, cross, full in lora_jobs:
            try:
                look, hist = run_lora(tr, va, te_k + le_k, cross, a.lora_epochs, full, t0)
            except torch.OutOfMemoryError:
                note = f"LoRA stopped by a CUDA out-of-memory error in fold {k + 1}; folds done: {len(lora_hist)}."
                torch.cuda.empty_cache()
                break
            for c in te_k + le_k:
                pooled.setdefault(LORA_NAME, {})[id(c)] = look(c)
            per_fold.setdefault(LORA_NAME, []).append(evaluate_rank(look, te_k)["mrr"])
            lora_hist.append(hist)
        write_report(*args, lora_hist, note, time_rows, len(te), K, t0)


def run_lora(tr, va, te, cross, epochs, warm, t0):
    """Fine-tune LoRA (top 2 MiniLM layers, r = 8) together with the SubNN head, texts encoded on the fly.
    Warm start from the fold's trained head; LoRA B = 0, so the start equals the frozen model."""
    m0, bt0, cfg = warm
    enc = Encoder(lora_layers=2).to(DEV)
    head = SubNN(len(FEATURES)).to(DEV)
    head.load_state_dict(m0.state_dict())
    opt = torch.optim.AdamW([{"params": head.parameters(), "lr": 3e-4},
                             {"params": enc.lora.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    rng = np.random.default_rng(0)
    texts = PREP.tab.texts

    def embed_rows(rows):
        rows = sorted({r for r in rows if r >= 0})
        e = enc([texts[r] for r in rows], bs=64)
        table = torch.zeros(len(texts), e.shape[1], device=DEV, dtype=e.dtype)
        table[torch.as_tensor(rows, device=DEV)] = e
        return table

    def rows_for(cs, items):
        rr = []
        for c, x in zip(cs, items):
            rr += [c["q"], c["h"], c["qh"]] + c["prev"]
            for k, i, _ in x:
                rr += list(PREP.files[k]["rows"][i])
        return rr

    def lookup(cs):
        enc.eval()
        with torch.no_grad():
            table = embed_rows(rows_for(cs, [infile(c) for c in cs]))
        return scorer(head, Batcher(PREP, table, bt0.mu, bt0.sd), cs, cfg)

    best, best_state, hist = evaluate_rank(lookup(va), va)["mrr"], None, []
    idx = np.arange(len(tr))
    for ep in range(epochs):
        enc.train(), head.train()
        rng.shuffle(idx)
        for i in range(0, len(idx), 4):
            cs = [tr[j] for j in idx[i:i + 4]]
            items, pos_n = [], []
            for j, c in zip(idx[i:i + 4], cs):
                pos = list(np.where(c["y"])[0])
                neg = list(np.where(~c["y"])[0])
                k_in = min(len(neg), cfg["k_neg"])
                neg = list(rng.choice(neg, k_in, replace=False)) if k_in else []
                items.append([(c["key"], int(s), c["phi"][s]) for s in pos + neg] + cross[j][:max(k_in, 1)])
                pos_n.append(len(pos))
            table = embed_rows(rows_for(cs, items))
            loss = batch_loss(head, Batcher(PREP, table, bt0.mu, bt0.sd), cs, items, pos_n, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = evaluate_rank(lookup(va), va)["mrr"]
        hist.append(round(v, 3))
        print(f"  LoRA epoch {ep + 1}: val MRR {v:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if v > best + 1e-4:
            best = v
            best_state = ({k: x.detach().clone() for k, x in head.state_dict().items()},
                          {k: x.detach().clone() for k, x in enc.lora.state_dict().items()})
    if best_state:
        head.load_state_dict(best_state[0])
        enc.lora.load_state_dict(best_state[1])
    out = lookup(te)
    del enc
    torch.cuda.empty_cache()
    return out, hist


def write_report(pooled, per_fold, per_seed, cost, pi, has_h, te, le, lora_hist, lora_note, time_rows, n_time, K,
                 t0):
    def row(name, cases):
        return evaluate_rank(lambda c: pooled[name][id(c)], cases)
    te_ids = {id(c) for c in te}
    learned = [n for n in pooled if (n.startswith("full") or n.startswith("-"))
               and te_ids <= set(pooled[n])]          # a partial LoRA run is reported per fold below
    base = [n for n in pooled if n not in learned]
    L = ["# Cortex-SubNN: which symbol bodies to show up front", "",
         f"Symbol-level cases from 91 local agent sessions plus git history (private data; only these aggregates are "
         f"kept). {K}-fold cross-validation grouped by session: every read-time case is tested once, by a model that "
         f"never saw its session ({len(te)} cases). Early stopping and calibration use 15% of the training sessions. "
         f"Rankings are within the file that was read (median ~20 leaf symbols); budget = the Reader's include budget, "
         f"max(400, 25% of file tokens). ({time.time() - t0:.0f}s)", "",
         "## Ranking (pooled over folds)", "",
         "Learned rows are the mean of a seed ensemble per fold; the single-seed column shows how much one run varies.",
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
    for other in ("frozen MiniLM zero-shot (max sub-span cosine)", "v1 include features, logistic", "random"):
        d, lo, hi = boot_diff(rr["full Cortex-SubNN"], rr[other])
        L.append(f"- full Cortex-SubNN vs {other}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for n in learned:
        if n.startswith("-") or "LoRA" in n:
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
    L += ["", "## Gate (full model, held-out cases)", "",
          f"Mean weight on (prompt, snapshot, trail): with a snapshot {np.round(pi[has_h].mean(0), 2).tolist()}, "
          f"without {np.round(pi[~has_h].mean(0), 2).tolist()}.", "",
          "## Decision head: realised tokens on held-out cases (whole-file reads = 100%; lower is better)", "",
          "Each ranker is Platt-calibrated per fold on the validation sessions. A case costs the outline + bodies shown "
          "up front, plus one extra round trip (C_turn) and the missing bodies if a needed symbol was not shown. Rule A "
          "returns the whole file when that is cheaper in expectation; Rule B picks the up-front set; budget fill is the "
          "Reader's current policy with that ranker.", ""]
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
    if lora_hist:
        L += ["", f"LoRA validation MRR per fold by epoch: {lora_hist}"]
        if LORA_NAME not in learned and LORA_NAME in per_fold:
            L += [f"LoRA test MRR on completed folds {np.round(per_fold[LORA_NAME], 3).tolist()} vs full model "
                  f"{np.round(per_fold['full Cortex-SubNN'][:len(per_fold[LORA_NAME])], 3).tolist()}."]
    if lora_note:
        L += ["", lora_note]
    out = os.path.join(HERE, "results", "subnn.md")
    open(out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
