"""Co-need graph over semantic cells, analysable in discrete time frames.

cell     a symbol's position in code space at plane-0 resolution: the bitwise majority of plane 0 over its
         sub-span codes, top m coordinates, as an int (cortexnet/codec.py). Built from codes the cache already
         holds, so nodes cost nothing extra, and cells are shared by similar symbols across files and repos.
edge     two cells needed within `horizon` calls of each other (ranged, approved, edited or submitted together).
frames   every `frame_calls` calls the frame's edge counts are frozen into the history (one array per frame, with
         the task/session tag), and the live weights decay by `decay`. `frames()` replays the history, so the graph
         can be studied frame by frame; `neighbours()` uses the live weights.
"""
from collections import deque

import numpy as np


def symbol_cells(codec, codes, m=20):
    """codes: [n_sub, code_bytes] uint8 of one symbol's sub-spans -> its cell (majority of plane 0)."""
    p0 = codec.plane0(codes)[:, :min(m, codec.width[0], 63)]
    bits = (p0.sum(0) * 2 >= len(p0)).astype(np.int64)
    return int((bits << np.arange(len(bits) - 1, -1, -1)).sum())


class CoNeedGraph:
    def __init__(self, frame_calls=25, decay=0.9, horizon=3):
        self.frame_calls, self.decay, self.horizon = frame_calls, decay, horizon
        self.live, self.frame_counts, self.history = {}, {}, []
        self.recent = deque()           # (call, cells)
        self.calls, self.frame, self.tag = 0, 0, ""
        self.members = {}               # cell -> {symbol key: name}

    def add_symbol(self, cell, key, name=""):
        self.members.setdefault(cell, {})[key] = name

    def need(self, cells):
        """Cells needed at this call: pair them with each other and with cells needed in the last `horizon` calls."""
        cells = sorted(set(cells))
        if not cells:
            return
        prev = {c for call, cs in self.recent for c in cs}
        for i, a in enumerate(cells):
            for b in cells[i + 1:] + sorted(prev - {a}):
                e = (a, b) if a < b else (b, a)
                self.frame_counts[e] = self.frame_counts.get(e, 0) + 1
                self.live[e] = self.live.get(e, 0.0) + 1.0
        self.recent.append((self.calls, cells))

    def tick(self, tag=None):
        if tag is not None and tag != self.tag:
            self.close_frame()
            self.tag = tag
        self.calls += 1
        while self.recent and self.recent[0][0] < self.calls - self.horizon:
            self.recent.popleft()
        if self.calls % self.frame_calls == 0:
            self.close_frame()

    def close_frame(self):
        if self.frame_counts:
            e = np.array([(a, b, n) for (a, b), n in sorted(self.frame_counts.items())], np.int64)
            self.history.append(dict(frame=self.frame, tag=self.tag, calls=self.calls, edges=e))
        self.frame_counts = {}
        self.frame += 1
        self.live = {e: w * self.decay for e, w in self.live.items() if w * self.decay > 0.05}

    def neighbours(self, cells, k=10):
        """Cells most co-needed with `cells` (live weights), strongest first, excluding `cells` themselves."""
        cells = set(cells)
        score = {}
        for (a, b), w in self.live.items():
            if a in cells and b not in cells:
                score[b] = score.get(b, 0.0) + w
            elif b in cells and a not in cells:
                score[a] = score.get(a, 0.0) + w
        return sorted(score.items(), key=lambda x: -x[1])[:k]

    def frames(self):
        """Frame snapshots: (frame index, tag, calls at close, [n, 3] array of cell a, cell b, count)."""
        return [(h["frame"], h["tag"], h["calls"], h["edges"]) for h in self.history]

    def save(self, path):
        """Closed frames and live weights (the open frame is saved once it closes)."""
        np.savez(path, frames=np.array([(h["frame"], h["calls"]) for h in self.history], np.int64).reshape(-1, 2),
                 tags=np.array([h["tag"] for h in self.history]),
                 edges=np.concatenate([h["edges"] for h in self.history]) if self.history else np.zeros((0, 3), np.int64),
                 sizes=np.array([len(h["edges"]) for h in self.history], np.int64),
                 live=np.array([(a, b, w) for (a, b), w in self.live.items()], np.float64).reshape(-1, 3),
                 params=np.array([self.frame_calls, self.decay, self.horizon, self.calls, self.frame], np.float64))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        fc, dec, hor, calls, frame = d["params"]
        g = cls(int(fc), float(dec), int(hor))
        g.calls, g.frame = int(calls), int(frame)
        g.live = {(int(a), int(b)): float(w) for a, b, w in d["live"]}
        parts = np.split(d["edges"], np.cumsum(d["sizes"])[:-1]) if len(d["sizes"]) else []
        g.history = [dict(frame=int(f), calls=int(c), tag=str(t), edges=e)
                     for (f, c), t, e in zip(d["frames"], d["tags"], parts)]
        return g
