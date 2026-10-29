"""Cortex-SubNN coprocessor at read time: numpy inference of the head that bench/subnn_train.py exports.

For one read it scores every leaf symbol of the file from
  views    e_q (prompt), e_h (reasoning snapshot), e_qh (both embedded together; or a live hidden-state vector
           supplied by the caller), e_prev (cache context: decayed mean of the symbols the session touched last)
  symbols  M = 4 sub-span vectors per symbol (signature + 3 body windows), kept as bit-plane codes when the export
           carries a codec (cortexnet/codec.py), decoded on use
  phi      the include model's features (cortexnet/learn.py)
and returns calibrated probabilities (Platt, fitted at export). Same maths as `Bilinear.forward` in
bench/subnn_train.py; MiniLM (torch + transformers) is optional, and without it the Reader keeps its include model.
"""
import hashlib
import math
import os

import numpy as np

from .codec import BitplaneCodec
from .learn import idents
from .outline import symbols, tokens

SPLIT, M, MAXLEN = 600, 4, 128
ENC = "sentence-transformers/all-MiniLM-L6-v2"


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


def trail_text(names, body):
    return " ".join(names) + "\n" + body[:800]


def calib_features(s):
    """Per symbol of one read: [s - max s, max s, log n]. The first sets the order inside the read; the other two
    let the level move with how well the best symbol matches and with file size (the base rate falls with n)."""
    s = np.asarray(s, float)
    if not len(s):
        return np.zeros((0, 3))
    return np.stack([s - s.max(), np.full_like(s, s.max()), np.full_like(s, math.log(len(s)))], 1)


class MiniLM:
    """MiniLM mean-pool + L2 norm (same as sentence-transformers), frozen. Needs torch + transformers."""

    def __init__(self, device="cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.dev = torch, device
        self.tok = AutoTokenizer.from_pretrained(ENC)
        self.m = AutoModel.from_pretrained(ENC).to(device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

    def __call__(self, texts, bs=256):
        torch = self.torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                b = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAXLEN,
                             return_tensors="pt").to(self.dev)
                with torch.autocast(self.dev, dtype=torch.float16, enabled=self.dev == "cuda"):
                    hs = self.m(**b).last_hidden_state
                msk = b["attention_mask"].unsqueeze(-1).float()
                e = (hs.float() * msk).sum(1) / msk.sum(1).clamp(min=1)
                outs.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.cat(outs).cpu().numpy() if outs else np.zeros((0, 384), np.float32)


class Coprocessor:
    def __init__(self, path, encoder=None):
        d = np.load(path)
        self.g, self.dg, self.A, self.B = d["g"], d["dg"], d["A"], d["B"]
        self.Wg_w, self.Wg_b, self.ws_w, self.ws_b = d["Wg_w"], d["Wg_b"], d["ws_w"], float(d["ws_b"][0])
        self.phi_mu, self.phi_sd = d["phi_mu"], d["phi_sd"]
        # calib = (w_rel, w_max, w_logn, b) over calib_features. Heads exported before 2026-10-28 carry only a raw
        # pooled Platt (a, b), which came out with a < 0 and reversed every read's order: refuse them.
        if "calib" not in d:
            raise ValueError(f"{path}: no calib (old raw-score Platt {tuple(d['platt']) if 'platt' in d else None}); "
                             "re-export with bench/subnn_train.py --export")
        self.calib = np.asarray(d["calib"], float)
        if self.calib[0] <= 0:
            raise ValueError(f"{path}: calibration weight on s - max s is {self.calib[0]:.3f} <= 0")
        self.multi, self.gate, self.late, self.struct, self.diag, self.lowrank = (bool(x) for x in d["flags"])
        self.gamma = float(d["gamma"])
        self.encoder_name = str(d["encoder"])
        self.codec = BitplaneCodec(d["codec_V"], d["codec_bits"], d["codec_levels"]) if "codec_V" in d else None
        self.encoder = encoder
        self.dim = self.g.shape[1]
        self.store = {}         # text hash -> code (bytes, with a codec) or vector: the symbol cache

    # ------------------------------------------------------------------ vectors
    def _embed(self, texts):
        return np.asarray(self.encoder(texts), np.float32) if texts else np.zeros((0, self.dim), np.float32)

    def _embed_query(self, texts, cap=512):
        """Query-side texts (prompt, snapshot, trail) through a cache: a turn's reads share the request and the trail,
        and the user extract primes the request before the first read. Missing texts are encoded in one batch."""
        cache = self.__dict__.setdefault("qcache", {})
        new = list(dict.fromkeys(t for t in texts if t not in cache))
        if new:
            for t, v in zip(new, self._embed(new)):
                cache[t] = v
            while len(cache) > cap:
                cache.pop(next(iter(cache)))
        return np.stack([cache[t] for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def prime(self, prompt):
        """Encode a request ahead of the reads that will use it (the texts views() derives from a prompt with no
        snapshot yet). Returns how many texts were new."""
        q = (prompt or "").strip()
        texts = [t for t in dict.fromkeys((q[:1200], q[:600].strip(), " ")) if t]
        before = len(self.__dict__.get("qcache", {}))
        self._embed_query(texts)
        return len(self.qcache) - before

    def symbol_vectors(self, path, lines, syms):
        """[n, M, d] sub-span vectors of the symbols (zero rows for empty sub-spans), via the code cache."""
        texts = [[t.strip() for t in subspans(s, lines, path)] for s in syms]
        keys = [[hashlib.sha1(t.encode("utf8", "replace")).digest()[:12] if t else None for t in row] for row in texts]
        new = sorted({(k, t) for row_k, row_t in zip(keys, texts) for k, t in zip(row_k, row_t)
                      if k is not None and k not in self.store}, key=lambda x: x[0])
        if new:
            V = self._embed([t for _, t in new])
            vals = self.codec.encode(V) if self.codec is not None else V
            for (k, _), v in zip(new, vals):
                self.store[k] = v
        E = np.zeros((len(syms), M, self.dim), np.float32)
        rows, where = [], []
        for i, row in enumerate(keys):
            for j, k in enumerate(row):
                if k is not None:
                    rows.append(self.store[k])
                    where.append((i, j))
        if rows:
            R = np.stack(rows)
            R = self.codec.decode(R) if self.codec is not None else R
            for (i, j), v in zip(where, R):
                E[i, j] = v
        return E

    def symbol_codes(self, path, lines, syms):
        """Per symbol, its stored sub-span codes [n_sub, code_bytes] (after symbol_vectors filled the cache)."""
        out = []
        for s in syms:
            ks = [hashlib.sha1(t.strip().encode("utf8", "replace")).digest()[:12]
                  for t in subspans(s, lines, path) if t.strip()]
            out.append(np.stack([self.store[k] for k in ks if k in self.store]) if ks else np.zeros((0, 1), np.uint8))
        return out

    def views(self, prompt="", snapshot="", trail=(), qvec=None):
        """trail: [(weight, text)] of recently touched symbols, newest first (the cache context)."""
        q, h = (prompt or "").strip(), (snapshot or "").strip()
        texts = [q[:1200], h[-1200:], (q[:600] + "\n" + h[-600:]).strip()] + [t for _, t in trail]
        E = self._embed_query([t if t else " " for t in texts])
        eq, eh, eqh = (E[0] if q else 0 * E[0]), (E[1] if h else 0 * E[1]), (E[2] if q or h else 0 * E[2])
        if qvec is not None:
            eqh = np.asarray(qvec, np.float32) / max(float(np.linalg.norm(qvec)), 1e-12)
        ep, dp = np.zeros(self.dim, np.float32), 0.0
        if len(trail):
            w = np.array([self.gamma ** j for j in range(len(trail))], np.float32)
            ep = (E[3:] * w[:, None]).sum(0) / w.sum()
            ep /= max(float(np.linalg.norm(ep)), 1e-12)
            dp = 1.0
        hid = idents(h)
        H = np.array([math.log1p(tokens(h)) / 8 if h else 0.0, len(hid) / max(1, len(h.split())) if h else 0.0],
                     np.float32)
        return dict(eq=eq, eh=eh, ep=ep, eqh=eqh, dh=float(bool(h)), dp=dp, H=H)

    # ------------------------------------------------------------------ scores
    def gate_weights(self, v):
        if not self.multi:
            return np.array([1.0, 0.0, 0.0], np.float32)
        mask = np.array([1.0, v["dh"], v["dp"]], np.float32)
        g = self.Wg_w @ np.concatenate([v["eq"], v["eh"] * v["dh"], v["ep"], [v["dh"], v["dp"]], v["H"]]) + self.Wg_b \
            if self.gate else np.zeros(3, np.float32)
        z = g + np.log(mask + 1e-6)
        e = np.exp(z - z.max())
        return e / e.sum()

    def scores(self, v, E, phi):
        pi = self.gate_weights(v)
        u = v["eqh"] + pi[0] * self.g[0] * v["eq"] + pi[1] * self.g[1] * v["eh"] + pi[2] * self.g[2] * v["ep"]
        if not self.multi:
            u = v["eqh"] + self.g[0] * v["eqh"]
        if not self.late:
            msk = (np.abs(E).sum(-1) > 0).astype(np.float32)
            E = (E * msk[..., None]).sum(1, keepdims=True) / np.maximum(msk.sum(1, keepdims=True), 1)[..., None]
            E /= np.maximum(np.linalg.norm(E, axis=-1, keepdims=True), 1e-12)
        s = E @ (u * (1 + self.dg))                       # [n, M]
        if self.lowrank:
            s = s + (E @ self.B.T) @ (self.A @ u)
        s = s.max(1)          # missing sub-spans are zero vectors scoring 0, exactly as in training
        if self.struct:
            s = s + ((phi - self.phi_mu) / self.phi_sd) @ self.ws_w[0] + self.ws_b
        return s

    def prob(self, path, lines, syms, phi, prompt="", snapshot="", trail=(), qvec=None):
        s = self.scores(self.views(prompt, snapshot, trail, qvec), self.symbol_vectors(path, lines, syms), phi)
        return 1 / (1 + np.exp(-(calib_features(s) @ self.calib[:3] + self.calib[3])))
