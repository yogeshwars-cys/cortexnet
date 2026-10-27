"""Outline-first reads with learned up-front includes, a session trail, a symbol graph and a connected-symbol menu.

read(path, query)    small file: the whole file. Otherwise the outline plus the symbol bodies the include model
                     expects to be needed (within a token budget), then a short menu of connected symbols in
                     neighbouring files that the caller can pull with approve(ids) instead of reading those files.
range(path, a, b)    numbered lines a..b. Lines not shown before are a miss: they label the symbols that were
                     needed and not shown, and train the include model.
approve(ids)         bodies of menu items. Approving is a positive label; menu items never asked for are negatives.

Session state the scores use:
  trail    decayed identifier memory: query words, lines returned by ranges/approvals, included bodies, edits
  active   recently touched symbols (ranged, approved, included, edited) for the symbol-graph feature
  edits    every call checks the files seen so far for changes on disk, so edits made by other tools count too
Every call is appended to <state_dir>/events.jsonl.
"""
import difflib
import json
import os
import time

import numpy as np

from .learn import FEATURES, IncludeModel, featurize, idents
from .outline import outline, symbols, tokens

DECAY = 0.7
SRC_EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".java",
           ".kt", ".dart", ".swift", ".cs", ".rb", ".php", ".lua", ".md", ".html", ".css", ".vue", ".svelte"}


def numbered(lines, a, b):
    return "\n".join(f"{i:6d}\t{lines[i - 1]}" for i in range(a, b + 1))


class Reader:
    def __init__(self, state_dir=None, full_below=800, budget_frac=0.25, budget_min=400, tau=0.5, horizon=10,
                 features=None, menu=False, menu_size=3, tau_menu=0.35, neighbours=40, split_tokens=0):
        self.state_dir = state_dir or os.path.join(os.getcwd(), ".cortexnet")
        os.makedirs(self.state_dir, exist_ok=True)
        self.model = IncludeModel(os.path.join(self.state_dir, "include.json"), horizon=horizon)
        self.full_below, self.budget_frac, self.budget_min, self.tau = full_below, budget_frac, budget_min, tau
        self.mask = None if features is None else np.array([f in features for f in FEATURES])
        self.menu_on, self.menu_size, self.tau_menu, self.max_nb = menu, menu_size, tau_menu, neighbours
        self.split_tokens = split_tokens       # > 0: sub-blocks under leaf symbols bigger than this
        self.files, self.asked, self.events, self.calls = {}, {}, [], 0
        self.trail, self.active, self.menu, self.menu_n = {}, [], {}, 0
        self.index = {}
        self.log_path = os.path.join(self.state_dir, "events.jsonl")

    # ---------------------------------------------------------------- bookkeeping
    def _symbols(self, text, p):
        return symbols(text, p, self.split_tokens)

    def _log(self, **kw):
        kw["ts"] = time.time()
        with open(self.log_path, "a", encoding="utf8") as f:
            f.write(json.dumps(kw) + "\n")

    def _remember(self, words, amount=1.0):
        for w in words:
            self.trail[w] = min(1.0, self.trail.get(w, 0.0) + (1 - DECAY) * amount)

    def _touch(self, p, a, b, weight=1.0):
        st = self.files.get(p)
        if not st:
            return
        for s in st["syms"]:
            if s.start <= b and s.end >= a and not s.children:
                body = idents("\n".join(st["lines"][s.start - 1:s.end]))
                self.active.append([weight, (p, s.start, s.end), s.name.lower(), body])
        self.active = self.active[-30:]

    def _tick(self):
        self.calls += 1
        self.trail = {w: v * DECAY for w, v in self.trail.items() if v * DECAY > 0.02}
        for a in self.active:
            a[0] *= DECAY
        self.active = [a for a in self.active if a[0] > 0.05]
        for p, st in list(self.files.items()):
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if mt != st.get("mtime"):
                _, _, lines, changed = self._load(p)
                st.update(lines=lines, mtime=mt, syms=self._symbols("\n".join(lines), p), changed=changed)
                if changed:
                    lo, hi = min(changed), max(changed)
                    self._remember(idents("\n".join(lines[lo - 1:hi])))
                    self._touch(p, lo, hi)
        for ev in [e for e in self.events if e["call"] + self.model.horizon < self.calls]:
            self._finalize(ev)

    def _finalize(self, ev):
        if ev in self.events:
            self.events.remove(ev)
        idx = [k for k in range(len(ev["spans"])) if not ev["included"][k]]
        if not idx:
            return
        y = []
        for k in idx:
            s0, s1 = ev["spans"][k]
            y.append(int(any(min(s1, b) - max(s0, a) + 1 >= max(1, 0.5 * min(s1 - s0 + 1, b - a + 1))
                             for a, b in ev["requests"])))
        self.model.update(ev["X"][idx], y)
        self.model.save()

    def _request(self, p, a, b):
        for ev in self.events:
            if ev["path"] == p:
                ev["requests"].append((a, b))

    def _load(self, path):
        p = os.path.abspath(path)
        with open(p, encoding="utf8", errors="replace") as f:
            text = f.read()
        lines = text.split("\n")
        old = self.files.get(p)
        changed = set()
        if old is not None and old["lines"] != lines:
            sm = difflib.SequenceMatcher(a=old["lines"], b=lines, autojunk=False)
            for op, _, _, j1, j2 in sm.get_opcodes():
                if op != "equal":
                    changed.update(range(j1 + 1, max(j2, j1 + 1) + 1))
        return p, text, lines, changed

    def _state(self, p, lines, syms):
        st = self.files.setdefault(p, dict(shown=set()))
        try:
            st["mtime"] = os.path.getmtime(p)
        except OSError:
            st["mtime"] = None
        st.update(lines=lines, syms=syms)
        return st

    def _features(self, p, syms, lines, query, changed):
        return featurize(syms, lines, query, changed, self.asked.get(p, []), self.trail,
                         [(w, k, n, b) for w, k, n, b in self.active], key=lambda s: (p, s.start, s.end),
                         mask=self.mask)

    # ---------------------------------------------------------------- neighbours and menu
    def _neighbours(self, p):
        d = os.path.dirname(p)
        try:
            names = sorted(os.listdir(d))
        except OSError:
            return []
        out = []
        for n in names:
            q = os.path.join(d, n)
            if q == p or os.path.splitext(n)[1].lower() not in SRC_EXT or not os.path.isfile(q):
                continue
            try:
                mt, sz = os.path.getmtime(q), os.path.getsize(q)
            except OSError:
                continue
            if sz > 300_000:
                continue
            ent = self.index.get(q)
            if ent is None or ent["mtime"] != mt:
                with open(q, encoding="utf8", errors="replace") as f:
                    text = f.read()
                lines = text.split("\n")
                ent = dict(mtime=mt, lines=lines, syms=[s for s in self._symbols(text, q) if s.kind not in ("section", "imports")])
                self.index[q] = ent
            out.append((q, ent))
            if len(out) >= self.max_nb:
                break
        return out

    def _menu(self, p, query):
        if not self.menu_on or (self.mask is not None and not self.mask[FEATURES.index("conn")]):
            return []
        cands = []
        for q, ent in self._neighbours(p):
            syms = [s for s in ent["syms"] if not s.children]
            if not syms:
                continue
            X, _ = self._features(q, syms, ent["lines"], query, set())
            if not X[:, FEATURES.index("conn")].any() and not X[:, FEATURES.index("trail")].any():
                continue
            P = self.model.prob(X)
            for k, s in enumerate(syms):
                cost = tokens("\n".join(ent["lines"][s.start - 1:s.end]))
                if P[k] >= self.tau_menu and (q, s.start, s.end) not in {m["key"] for m in self.menu.values()}:
                    cands.append((P[k] / cost ** 0.5, P[k], q, s, cost, X[k]))
        cands.sort(key=lambda c: -c[0])
        picked = []
        for _, pk, q, s, cost, x in cands[:self.menu_size]:
            self.menu_n += 1
            mid = f"#{self.menu_n}"
            self.menu[mid] = dict(path=q, a=s.start, b=s.end, key=(q, s.start, s.end), name=s.name, cost=cost)
            self.events.append(dict(path=q, call=self.calls, X=x[None, :], spans=[(s.start, s.end)], included=[False],
                                    requests=[], menu=mid))
            picked.append(f"  {mid} {os.path.basename(q)}:{s.name} (lines {s.start}-{s.end}, ~{cost} tok)")
        return picked

    # ---------------------------------------------------------------- tools
    def read(self, path, query="", approve=()):
        pre = self.approve(approve) if approve else ""
        self._tick()
        p, text, lines, changed = self._load(path)
        changed |= self.files.get(p, {}).pop("changed", set())
        full = tokens(text)
        self._remember(idents(query))
        for ev in [e for e in self.events if e["path"] == p and "menu" not in e]:
            self._finalize(ev)
        if full <= self.full_below:
            out = numbered(lines, 1, len(lines))
            st = self._state(p, lines, self._symbols(text, p))
            st["shown"] = set(range(1, len(lines) + 1))
            self._log(op="read", path=p, file_tokens=full, shown_tokens=tokens(out), mode="whole", included=0)
            return pre + out
        syms = self._symbols(text, p)
        X, _ = self._features(p, syms, lines, query, changed)
        P = self.model.prob(X)
        budget = max(self.budget_min, self.budget_frac * full)
        inc, used, taken = [False] * len(syms), 0, []
        for k in sorted(range(len(syms)), key=lambda k: -P[k]):
            s = syms[k]
            if P[k] < self.tau or s.children:
                continue
            cost = tokens("\n".join(lines[s.start - 1:s.end]))
            if used + cost > budget or any(not (s.end < a or s.start > b) for a, b in taken):
                continue
            inc[k], used = True, used + cost
            taken.append((s.start, s.end))
        parts = [outline(text, p, syms)]
        st = self._state(p, lines, syms)
        st["shown"] = set()
        for a, b in sorted(taken):
            parts.append(f"--- lines {a}-{b} ---\n" + numbered(lines, a, b))
            st["shown"].update(range(a, b + 1))
            self._remember(idents("\n".join(lines[a - 1:b])), 0.5)
            self._touch(p, a, b, 0.5)
        self.events.append(dict(path=p, call=self.calls, X=X, spans=[(s.start, s.end) for s in syms], included=inc,
                                requests=[]))
        menu = self._menu(p, query)
        if menu:
            parts.append("[connected] symbols in neighbouring files; pull with cortex_approve(ids) instead of reading "
                         "those files:\n" + "\n".join(menu))
        out = "\n".join(parts)
        self._log(op="read", path=p, file_tokens=full, shown_tokens=tokens(out), mode="outline", included=sum(inc),
                  symbols=len(syms), menu=len(menu), menu_tokens=tokens("\n".join(menu)) if menu else 0)
        return pre + out

    def range(self, path, start, end, approve=()):
        pre = self.approve(approve) if approve else ""
        self._tick()
        p, text, lines, changed = self._load(path)
        a, b = max(1, int(start)), min(len(lines), int(end))
        if a > b:
            return pre + f"empty range: file has {len(lines)} lines"
        out = numbered(lines, a, b)
        st = self.files.get(p) or self._state(p, lines, self._symbols(text, p))
        miss = any(i not in st["shown"] for i in range(a, b + 1))
        st["shown"].update(range(a, b + 1))
        self.asked.setdefault(p, []).append((a, b))
        self._request(p, a, b)
        self._remember(idents(out))
        self._touch(p, a, b)
        self._log(op="range", path=p, file_tokens=tokens(text), shown_tokens=tokens(out), miss=miss, start=a, end=b)
        return pre + out

    def approve(self, ids):
        self._tick()
        parts = []
        for mid in ids:
            m = self.menu.get(mid if str(mid).startswith("#") else f"#{mid}")
            if not m:
                parts.append(f"unknown menu id {mid}")
                continue
            p = m["path"]
            _, text, lines, _ = self._load(p)
            a, b = m["a"], min(m["b"], len(lines))
            body = numbered(lines, a, b)
            parts.append(f"--- {os.path.basename(p)} lines {a}-{b} ({m['name']}) ---\n" + body)
            st = self.files.get(p) or self._state(p, lines, self._symbols(text, p))
            st["shown"].update(range(a, b + 1))
            self.asked.setdefault(p, []).append((a, b))
            self._request(p, a, b)
            self._remember(idents(body))
            self._touch(p, a, b)
            self._log(op="approve", path=p, file_tokens=tokens(text), shown_tokens=tokens(body), id=mid)
        return "\n".join(parts) + ("\n" if parts else "")

    def outline(self, path):
        self._tick()
        p, text, lines, _ = self._load(path)
        syms = self._symbols(text, p)
        out = outline(text, p, syms)
        self._state(p, lines, syms)
        self._log(op="outline", path=p, file_tokens=tokens(text), shown_tokens=tokens(out))
        return out

    def stats(self):
        ev = []
        if os.path.exists(self.log_path):
            with open(self.log_path, encoding="utf8") as f:
                ev = [json.loads(l) for l in f]
        reads = [e for e in ev if e["op"] == "read"]
        ranges = [e for e in ev if e["op"] == "range"]
        shown = sum(e["shown_tokens"] for e in ev)
        whole = sum(e["file_tokens"] for e in reads)
        return (f"reads {len(reads)}, ranges {len(ranges)} (misses {sum(e['miss'] for e in ranges)}), approvals "
                f"{sum(e['op'] == 'approve' for e in ev)}, outlines {sum(e['op'] == 'outline' for e in ev)}; tokens shown "
                f"{shown:,} vs whole-file reads {whole:,} ({shown / max(whole, 1):.0%}); include model updates "
                f"{self.model.updates}")
