"""Latent requests and prefetch: keep CortexNET a step ahead of the agent.

After every agent turn the prefetcher extracts where the agent is heading: the `want` on its task card (an explicit
latent request), file paths it mentions, and identifiers that name symbols in the repository (through a symbol index
built in the background at start). While the model is thinking about its next step, a background thread warms those
files: their symbols are parsed and, with the coprocessor, encoded into its cache, which is the expensive part of a
CortexNET read. The agent's latest `want` also becomes part of every read's query, so the bodies picked up front follow
its stated intent.

Hit rate: a read_file of a file predicted at the previous step counts as a hit (shown by /context).
"""
import os
import re
import threading

from .outline import symbols
from .reader import SRC_EXT

PATH_RX = re.compile(r"[\w./\\-]+\.[A-Za-z0-9]{1,5}\b")
IDENT_RX = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules", ".cortexnet", ".mypy_cache", ".pytest_cache",
        "build", "dist"}
COMMON = {"self", "None", "True", "False", "return", "import", "from", "class", "def", "with", "this", "that",
          "file", "line", "lines", "code", "test", "tests", "data", "main", "init", "path", "name", "value", "type"}


class Prefetcher:
    def __init__(self, ws, max_files=3, max_index_files=2000):
        self.ws, self.max_files = ws, max_files
        self.rels, self.index = [], {}          # repository source files; symbol name -> {rel: first line}
        self.predicted, self.warm = [], set()
        self.hits = self.misses = self.warmed = 0
        self.threads = []
        self.ready, self.stop = threading.Event(), threading.Event()
        t = threading.Thread(target=self._build, args=(max_index_files,), daemon=True)
        t.start()
        self.threads.append(t)

    def _build(self, cap):
        try:
            for dp, dns, fns in os.walk(self.ws.root):
                dns[:] = [d for d in dns if d not in SKIP and not d.startswith(".")]
                if self.stop.is_set():
                    return
                for f in fns:
                    if os.path.splitext(f)[1].lower() not in SRC_EXT or len(self.rels) >= cap:
                        continue
                    p = os.path.join(dp, f)
                    if os.path.getsize(p) > 1_000_000:
                        continue
                    rel = self.ws.rel(p)
                    self.rels.append(rel)
                    try:
                        with open(p, encoding="utf8", errors="replace") as fh:
                            for s in symbols(fh.read(), p):
                                self.index.setdefault(s.name.split(":")[0], {}).setdefault(rel, s.start)
                    except (OSError, ValueError, SyntaxError, RecursionError):
                        continue
        finally:
            self.ready.set()

    def predict(self, text):
        """Repository files the agent is heading for, most likely first: paths it names, then files defining the
        symbols it names (names defined in at most 2 files, so generic ones do not count)."""
        out, score = [], {}
        for m in PATH_RX.findall(text or ""):
            m = m.replace("\\", "/").strip("./")
            hits = [r for r in self.rels if r == m or r.endswith("/" + m)]
            if len(hits) == 1 and hits[0] not in out:
                out.append(hits[0])
        for w in IDENT_RX.findall(text or ""):
            if w in COMMON:
                continue
            files = self.index.get(w, ())
            if 0 < len(files) <= 2:
                for r in files:
                    score[r] = score.get(r, 0) + 1
        out += [r for r, _ in sorted(score.items(), key=lambda x: -x[1]) if r not in out]
        return out[:self.max_files]

    def locate(self, name):
        """{rel: first line} of the files defining symbol `name` (empty if unknown or the index is still building)."""
        return dict(self.index.get(name, {}))

    def step(self, text):
        """Called after each agent turn with its words (and its card's `want`): predict and warm in the background."""
        return self.prime(self.predict(text))

    def prime(self, rels):
        """Warm these files in the background and count the next read against them (user extract, step)."""
        self.predicted = list(rels)[:self.max_files]
        todo = [r for r in self.predicted if r not in self.warm]
        if todo and not self.stop.is_set():
            t = threading.Thread(target=self._warm, args=(todo,), daemon=True)
            t.start()
            self.threads.append(t)
        return self.predicted

    def _warm(self, rels):
        for rel in rels:
            if self.stop.is_set():
                return
            try:
                with self.ws.lock:
                    if self.ws.reader.warm(self.ws.path(rel)):
                        self.warmed += 1
                self.warm.add(rel)
            except (OSError, ValueError):
                continue

    def observe(self, path):
        """A read_file happened: count it against the last prediction."""
        try:
            rel = self.ws.rel(self.ws.path(path))
        except ValueError:
            return
        if rel in self.predicted:
            self.hits += 1
        else:
            self.misses += 1

    def wait(self, timeout=10):
        for t in list(self.threads):
            t.join(timeout)

    def close(self, timeout=10):
        """Stop the background work and wait for it (files are open while it runs)."""
        self.stop.set()
        self.wait(timeout)
