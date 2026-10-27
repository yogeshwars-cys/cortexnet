"""Bit-plane vector codec, ported from the Wormhole BPF64 engine (wormhole.cpp `Codec`, export_quantized_dual_funnel.py).

fit      orthonormal basis V = top eigenvectors of the uncentred second moment (so x ~ V z and |V z| = |z|),
         coordinates sorted by variance; a greedy rate-distortion pass gives each coordinate 0..5 bits
         (non-increasing in variance, summing to exactly 8 * code_bytes); per-coordinate Lloyd-Max levels.
encode   z = x V, index = bucket of z_j among the level midpoints, stored as bit-planes: plane p holds bit
         (b_j - 1 - p) of every coordinate with b_j > p, MSB first, planes concatenated, packed big-endian.
score    asymmetric, as in wormhole: the query stays unquantised; a per-query table of (q_j * level, level^2)
         gives cosine(q, decoded doc) = sum_a / (|q_z| sqrt(sum_b)) straight from the codes.
prefilter  plane 0 alone (one bit per coordinate, the top half / bottom half of its levels) scored with the
         query's conditional means: a coarse cosine that touches 1/8..1/2 of the code, then score() re-ranks.
cells    plane-0 bits of the top m coordinates as an integer: a semantic cell id, free from the code.

64 bytes is wormhole's code size; 128 and 256 bytes keep more coordinates and deeper planes.
"""
import numpy as np

MAX_BITS = 5
# Lloyd-Max mean squared error of a unit Gaussian at 0..5 bits
_GAUSS_MSE = np.array([1.0, 0.3634, 0.1175, 0.03454, 0.009497, 0.002499])


def _lloyd_max(z, n, iters=30):
    """1-D Lloyd-Max (k-means) levels for samples z, n levels, sorted."""
    lv = np.quantile(z, (np.arange(n) + 0.5) / n)
    for _ in range(iters):
        th = (lv[:-1] + lv[1:]) / 2
        b = np.searchsorted(th, z)
        s = np.bincount(b, z, minlength=n)
        c = np.bincount(b, minlength=n)
        new = np.where(c > 0, s / np.maximum(c, 1), lv)
        if np.allclose(new, lv):
            break
        lv = np.sort(new)
    return lv


class BitplaneCodec:
    def __init__(self, V, bits, levels):
        self.V = np.asarray(V, np.float32)                      # [d, k]
        self.bits = np.asarray(bits, np.uint8)                  # [k], non-increasing, >= 1
        self.levels = np.asarray(levels, np.float32)            # [k, 32], first 2^b_j used
        self.k = len(self.bits)
        self.n_lv = 1 << self.bits.astype(np.int64)
        self.th = np.full((self.k, 31), np.inf, np.float32)     # level midpoints
        for j in range(self.k):
            lv = self.levels[j, :self.n_lv[j]]
            self.th[j, :self.n_lv[j] - 1] = (lv[:-1] + lv[1:]) / 2
        self.width = [int((self.bits > p).sum()) for p in range(int(self.bits.max()))]
        self.code_bits = int(self.bits.astype(np.int64).sum())
        assert self.code_bits % 8 == 0
        self.code_bytes = self.code_bits // 8
        self.off = np.concatenate([[0], np.cumsum(self.width)])[:-1]
        # plane 0 = top / bottom half of each coordinate's levels: conditional means for the prefilter
        half = self.n_lv // 2
        self.m0 = np.array([self.levels[j, :half[j]].mean() for j in range(self.k)], np.float32)
        self.m1 = np.array([self.levels[j, half[j]:self.n_lv[j]].mean() for j in range(self.k)], np.float32)

    # ------------------------------------------------------------------ fit / io
    @classmethod
    def fit(cls, X, code_bytes=64, seed=0, max_samples=50000):
        X = np.asarray(X, np.float32)
        X = X[np.linalg.norm(X, axis=1) > 0]
        rng = np.random.default_rng(seed)
        if len(X) > max_samples:
            X = X[rng.choice(len(X), max_samples, replace=False)]
        d = X.shape[1]
        w, U = np.linalg.eigh((X.T @ X).astype(np.float64) / len(X))
        order = np.argsort(-w)
        w, U = np.maximum(w[order], 0), U[:, order]
        budget = 8 * code_bytes
        bits = np.zeros(d, np.int64)
        gain = w * (_GAUSS_MSE[0] - _GAUSS_MSE[1])
        for _ in range(budget):                       # greedy reverse water-filling
            j = int(np.argmax(gain))
            if gain[j] <= 0:
                break
            bits[j] += 1
            gain[j] = w[j] * (_GAUSS_MSE[bits[j]] - _GAUSS_MSE[bits[j] + 1]) if bits[j] < MAX_BITS else -1.0
        bits = np.sort(bits)[::-1]                   # keep bits non-increasing along the variance order
        if bits.sum() < budget:                      # more bits than useful variance: top up the leading coords
            for j in range(d):
                while bits[j] < MAX_BITS and bits.sum() < budget:
                    bits[j] += 1
        k = int((bits > 0).sum())
        V = U[:, :k].astype(np.float32)
        Z = X @ V
        levels = np.zeros((k, 32), np.float32)
        for j in range(k):
            levels[j, :1 << int(bits[j])] = _lloyd_max(Z[:, j], 1 << int(bits[j]))
        return cls(V, bits[:k], levels)

    def save(self, path):
        np.savez(path, V=self.V, bits=self.bits, levels=self.levels)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["V"], d["bits"], d["levels"])

    # ------------------------------------------------------------------ codes
    def project(self, X):
        return np.asarray(X, np.float32) @ self.V

    def quantize(self, Z):
        idx = np.empty(Z.shape, np.uint8)
        for j in range(self.k):
            idx[:, j] = np.searchsorted(self.th[j, :self.n_lv[j] - 1], Z[:, j])
        return idx

    def encode(self, X):
        idx = self.quantize(self.project(np.atleast_2d(X)))
        planes = []
        for p, w in enumerate(self.width):
            shift = (self.bits[:w].astype(np.int64) - 1 - p)
            planes.append(((idx[:, :w] >> shift) & 1).astype(np.uint8))
        return np.packbits(np.concatenate(planes, 1), axis=1)

    def unpack(self, codes):
        b = np.unpackbits(np.atleast_2d(codes), axis=1)[:, :self.code_bits]
        idx = np.zeros((b.shape[0], self.k), np.uint8)
        for p, w in enumerate(self.width):
            shift = (self.bits[:w].astype(np.int64) - 1 - p).astype(np.uint8)
            idx[:, :w] |= b[:, self.off[p]:self.off[p] + w] << shift
        return idx

    def decode(self, codes, unit=True):
        idx = self.unpack(codes)
        Z = np.take_along_axis(self.levels, idx.T.astype(np.int64), 1).T
        X = Z @ self.V.T
        if unit:
            X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return X

    # ------------------------------------------------------------------ search
    def lut(self, q):
        """Per-query tables: a[j, c] = q_j level_c / |q_z|, b[j, c] = level_c^2."""
        qz = self.project(q)
        qz = qz / max(float(np.linalg.norm(qz)), 1e-12)
        return qz[:, None] * self.levels, self.levels ** 2

    def score(self, codes, q=None, lut=None, idx=None):
        """Cosine between q and every decoded code (asymmetric: q is not quantised)."""
        a, b = lut if lut is not None else self.lut(q)
        idx = self.unpack(codes) if idx is None else idx
        cols = np.arange(self.k)
        sa = a[cols, idx].sum(1)
        sb = b[cols, idx].sum(1)
        return sa / np.sqrt(np.maximum(sb, 1e-12))

    def plane0(self, codes):
        """[n, k] 0/1: the first bit plane (every coordinate has one)."""
        return np.unpackbits(np.atleast_2d(codes), axis=1)[:, :self.width[0]]

    def prefilter(self, codes, q, keep, p0=None):
        """Coarse ranking from plane 0 only; returns the indices of the best `keep` codes, best first."""
        qz = self.project(q)
        p0 = self.plane0(codes) if p0 is None else p0
        s = p0.astype(np.float32) @ (qz * (self.m1 - self.m0))
        keep = min(keep, len(s))
        top = np.argpartition(-s, keep - 1)[:keep]
        return top[np.argsort(-s[top])]

    def search(self, codes, q, k=10, keep=None, p0=None):
        """Plane-0 prefilter to `keep` candidates, LUT re-rank, top k as (indices, cosines)."""
        keep = keep or max(10 * k, 200)
        cand = self.prefilter(codes, q, keep, p0) if keep < len(codes) else np.arange(len(codes))
        s = self.score(codes[cand], q)
        o = np.argsort(-s)[:k]
        return cand[o], s[o]

    def cells(self, codes, m=20):
        """Semantic cell id per code: plane-0 bits of the top m coordinates, as an int."""
        m = min(m, self.width[0], 63)
        bits = self.plane0(codes)[:, :m].astype(np.int64)
        return (bits << np.arange(m - 1, -1, -1)).sum(1)
