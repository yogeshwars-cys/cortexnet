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

Optional (all off by default):
  coproc   a cortexnet.coproc.Coprocessor: adds semantic context (the prompt, the reasoning snapshot or a live hidden
           state, and the symbols the session touched last) on top of the include model. A leaf symbol's probability
           is 1 - (1 - p_include)(1 - p_coproc): either can bring a body up front, and the coprocessor never takes
           away a body the include model would show. The include model keeps learning from the same events.
           On top of that the coprocessor's own top `coproc_top` symbols (default 1) go up front when each is at most
           `coproc_max` tokens (default 400), by rank, since its honest probabilities rarely clear a threshold.
  policy   "rule_ab": choose the up-front set (or the whole file) by Rule A/B under the prefix-cache cost
           (cortexnet/policy.py) instead of filling a fixed budget.
  graph    a cortexnet.graph.CoNeedGraph over the coprocessor's code cells: ranged/approved symbols are co-needs,
           and the [connected] menu also offers co-needed symbols from any file seen before.

Neural-dynamics options (all off by default; bench/replay_sessions.py measures each):
  homeostasis  gain beta > 0: the include threshold is a thermostat on the session's miss rate. H_miss, a leaky
           average of "a read was followed by a range request for a body it did not show", moves the threshold:
           tau_t = tau * exp(-beta * (H_miss - miss_target)): more misses lower it (show more), fewer raise it.
  refractory   a body this session already sent, unchanged since, is not sent again by a later read of the file: a
           one-line back-reference instead. `forget(path)` (called by the fade when it shrinks a read) ends it.
  menu_rule    "wta": menu candidates compete; their scores are normalised against the winner (softmax over the log
           scores) and only those above `menu_floor` are offered, up to menu_size, so a flat field offers nothing.
  learn_rule   "nlms": the include model's per-feature normalised steps (cortexnet/learn.py).
"""
import bisect
import difflib
import json
import os
import re
import time

import numpy as np

from .learn import FEATURES, IncludeModel, featurize, idents
from .outline import OUTLINE_MAX, outline, skeleton, symbols, tokens
from .policy import H_REM_PRIOR, effective_turn, rule_ab

DECAY = 0.7
SRC_EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".java",
           ".kt", ".dart", ".swift", ".cs", ".rb", ".php", ".lua", ".md", ".html", ".css", ".vue", ".svelte"}


QUOTED = re.compile(r'"([^"\n]{3,})"|`([^`\n]{2,})`|\'([^\'\n]{3,})\'')


def anchors(lines, syms, query, max_hits=6):
    """Lines holding the request's literal anchors, best first: quoted text (an error message, a string value)
    matched by its first 24 characters, then `code` names at their definition or assignment (inside the named class
    when a dotted name gives one). A small model asked about one line of a big file rarely asks for a range when the
    outline does not show it, so these lines go up front."""
    hits, names = [], []
    for m in QUOTED.finditer(query or ""):
        t = next(g for g in m.groups() if g).strip()
        if " " in t or len(t) > 24 or not re.fullmatch(r"[\w.]+", t):
            frag = t[:24]
            hits += [i for i, l in enumerate(lines, 1) if frag in l][:2]
        else:
            names.append(t.split("."))
    bare = QUOTED.sub(" ", query or "")
    for t in re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b|\b\w*_\w+\b|\b[A-Z][a-z0-9]+[A-Z]\w*\b", bare):
        names.append(t.split("."))          # code-like words without quotes: Index.suffix, compute_hash, QuerySet
    for parts in list(dict.fromkeys(map(tuple, names))):
        name, owner = parts[-1], parts[-2] if len(parts) > 1 else None
        rx = re.compile(rf"^\s*(?:(?:async\s+)?def|class)\s+{re.escape(name)}\b|^\s*{re.escape(name)}\s*(?::[^=]*)?=(?!=)")
        found = [i for i, l in enumerate(lines, 1) if rx.match(l)]
        if owner:
            spans = [(s.start, s.end) for s in syms if s.name == owner and s.kind == "class"]
            inside = [i for i in found if any(a <= i <= b for a, b in spans)]
            found = inside or found
        hits += found[:2]
    return list(dict.fromkeys(hits))[:max_hits]


class Where:
    """The def or class a line sits in, for every span of one read: parent links between the defs and classes,
    built once, instead of a scan of every symbol per span (a read can show thousands of small bodies)."""

    def __init__(self, syms):
        self.defs = sorted((s for s in syms if s.kind in ("def", "class")), key=lambda s: (s.start, s.depth))
        self.starts = [s.start for s in self.defs]
        self.heads = set(self.starts)
        self.parent, stack = [], []
        for i, s in enumerate(self.defs):
            while stack and self.defs[stack[-1]].end < s.start:
                stack.pop()
            self.parent.append(stack[-1] if stack else None)
            stack.append(i)

    def _inner(self, a):
        i = bisect.bisect_left(self.starts, a) - 1
        while i is not None and i >= 0:
            if self.defs[i].end >= a:
                return i
            i = self.parent[i]
        return None

    def inner(self, a):
        """The innermost def or class that starts before line a and still holds it, or None."""
        i = self._inner(a)
        return None if i is None else self.defs[i]

    def __call__(self, a, b):
        """' (in Class.method, L60-120)' for a span that starts inside a def or class rather than at its first line,
        so a window around an anchor names the function it belongs to; '' otherwise."""
        if a in self.heads:
            return ""
        i = self._inner(a)
        if i is None:
            return ""
        s = self.defs[i]
        owner = self.parent[i]
        while owner is not None and not (self.defs[owner].kind == "class" and self.defs[owner].depth < s.depth):
            owner = self.parent[owner]
        name = f"{self.defs[owner].name}.{s.name}" if owner is not None else s.name
        return f" (in {name}, L{s.start}-{s.end})"


def where(syms, a, b):
    return Where(syms)(a, b)


class Spans:
    """Disjoint line spans kept sorted: an O(log n) overlap test. Spans are added only after that test, so they stay
    disjoint, and among the spans starting at or before b the last one reaches furthest."""

    def __init__(self):
        self.starts, self.ends = [], []

    def overlaps(self, a, b):
        i = bisect.bisect_right(self.starts, b) - 1
        return i >= 0 and self.ends[i] >= a

    def add(self, a, b):
        i = bisect.bisect_left(self.starts, a)
        self.starts.insert(i, a)
        self.ends.insert(i, b)


def changed_lines(old, new, max_cells=4_000_000):
    """1-based lines of `new` that an edit touched. The common prefix and suffix are skipped first (edits are local),
    then only the middle is diffed; a middle too big to diff (difflib is quadratic on repetitive lines) counts as
    changed as a whole."""
    n = min(len(old), len(new))
    lo = 0
    while lo < n and old[lo] == new[lo]:
        lo += 1
    hi = 0
    while hi < n - lo and old[-1 - hi] == new[-1 - hi]:
        hi += 1
    a, b = old[lo:len(old) - hi], new[lo:len(new) - hi]
    if len(a) * len(b) > max_cells:
        return set(range(lo + 1, lo + max(len(b), 1) + 1))
    out = set()
    for op, _, _, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "equal":
            out.update(range(lo + j1 + 1, lo + max(j2, j1 + 1) + 1))
    return out


def numbered(lines, a, b):
    return "\n".join(f"{i:6d}\t{lines[i - 1]}" for i in range(a, b + 1))


class Reader:
    def __init__(self, state_dir=None, full_below=800, budget_frac=0.25, budget_min=400, tau=0.5, horizon=10,
                 features=None, menu=False, menu_size=3, tau_menu=0.35, neighbours=40, split_tokens=0,
                 coproc=None, policy="budget", ctx_base=20000, h_rem=H_REM_PRIOR, graph=None, coproc_top=1,
                 coproc_max=400, xray_top=0, xray_lines=12, xray_budget=400, mmr_beta=0.0, homeostasis=0.0,
                 miss_target=0.3, homeo_rate=0.1, refractory=False, menu_rule="topk", menu_floor=0.25,
                 learn_rule="logistic", max_tokens=None, anchor=True, anchor_max=400, focus=False):
        """max_tokens: a cap on one read's output (outline at most half of it, bodies the rest); set it from the
        context window (e.g. a quarter of it) so one read of a big file cannot fill a small context. None = no cap."""
        self.max_tokens = max_tokens
        self.anchor, self.anchor_max = anchor, anchor_max      # anchors(): the request's literal lines go up front
        # focus: anchored lines are shown first, before the outline, labelled as matching the request, and a window
        # on a def/class line starts at that line (small models read the top of a tool result)
        self.focus = focus
        self.coproc, self.policy, self.ctx_base, self.h_rem, self.graph = coproc, policy, ctx_base, h_rem, graph
        self.homeostasis, self.miss_target, self.homeo_rate = homeostasis, miss_target, homeo_rate
        self.h_miss = miss_target              # the thermostat starts at its set point
        self.refractory, self.live, self.reads = refractory, {}, 0     # live: (path, a, b) -> (read no., body hash)
        self.menu_rule, self.menu_floor = menu_rule, menu_floor
        self.coproc_top, self.coproc_max = coproc_top, coproc_max
        self.xray_top, self.xray_lines, self.xray_budget = xray_top, xray_lines, xray_budget
        self.mmr_beta = mmr_beta              # > 0: the coprocessor's picks avoid near-duplicates of each other
        self.served, self.text_of, self.cell_of = 0, {}, {}      # tokens returned; trail texts; symbol -> cell
        self.state_dir = state_dir or os.path.join(os.getcwd(), ".cortexnet")
        os.makedirs(self.state_dir, exist_ok=True)
        self.model = IncludeModel(os.path.join(self.state_dir, "include.json"), horizon=horizon, rule=learn_rule)
        self.full_below, self.budget_frac, self.budget_min, self.tau = full_below, budget_frac, budget_min, tau
        self.mask = None if features is None else np.array([f in features for f in FEATURES])
        self.menu_on, self.menu_size, self.tau_menu, self.max_nb = menu, menu_size, tau_menu, neighbours
        self.split_tokens = split_tokens       # > 0: sub-blocks under leaf symbols bigger than this
        self.files, self.asked, self.events, self.calls = {}, {}, [], 0
        self.trail, self.active, self.menu, self.menu_n = {}, [], {}, 0
        self.index, self.parsed = {}, {}
        self.last = {}                         # path -> why the last read showed what it did (the CLI's /why)
        self.log_path = os.path.join(self.state_dir, "events.jsonl")

    # ---------------------------------------------------------------- bookkeeping
    def _symbols(self, text, p):
        key = (p, len(text), hash(text))                 # parse once per file content (prefetch warms this)
        hit = self.parsed.get(key)
        if hit is None:
            hit = self.parsed[key] = tuple(symbols(text, p, self.split_tokens))
            if len(self.parsed) > 256:
                self.parsed.pop(next(iter(self.parsed)))
        return list(hit)

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
                text = "\n".join(st["lines"][s.start - 1:s.end])
                self.active.append([weight, (p, s.start, s.end), s.name.lower(), idents(text)])
                self.text_of[(p, s.start, s.end)] = s.name + "\n" + text[:800]
        self.active = self.active[-30:]

    def _touch_many(self, p, spans, weight=1.0):
        """_touch for each span in order. Only the newest 30 entries of self.active survive a touch, so only the last
        spans that fill them are scanned (one read can take thousands of tiny bodies, each scan is over every symbol)."""
        st = self.files.get(p)
        if not st:
            return
        keep, n = [], 0
        for a, b in reversed(spans):
            keep.append((a, b))
            n += sum(1 for s in st["syms"] if s.start <= b and s.end >= a and not s.children)
            if n >= 30:
                break
        for a, b in reversed(keep):
            self._touch(p, a, b, weight)

    def _cells(self, p, a, b):
        st = self.files.get(p) or {}
        return [self.cell_of[(p, s.start, s.end)] for s in st.get("syms", [])
                if s.start <= b and s.end >= a and not s.children and (p, s.start, s.end) in self.cell_of]

    def _trail(self):
        """Cache context for the coprocessor: the last touched symbols, newest first, as (weight, text)."""
        out, seen = [], set()
        for w, k, _, _ in reversed(self.active):
            if k not in seen and k in self.text_of:
                seen.add(k)
                out.append((w, self.text_of[k]))
        return out[:6]

    def _tick(self):
        self.calls += 1
        if self.graph is not None:
            frames = len(self.graph.history)
            self.graph.tick()
            if len(self.graph.history) > frames:
                self.graph.save(os.path.join(self.state_dir, "graph.npz"))
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
        if self.homeostasis and "menu" not in ev:              # the thermostat: did this read miss a needed body?
            self.h_miss += self.homeo_rate * (float(any(y)) - self.h_miss)
        self.model.update(ev["X"][idx], y)
        self.model.save()

    def tau_now(self):
        """The include threshold in force (homeostatic when enabled)."""
        if not self.homeostasis:
            return self.tau
        return float(np.clip(self.tau * np.exp(-self.homeostasis * (self.h_miss - self.miss_target)), 0.05, 0.95))

    def forget(self, path):
        """The fade shrank a read of `path`: its bodies are no longer in context, so they may be sent again."""
        p = os.path.abspath(path)
        for k in [k for k in self.live if k[0] == p]:
            del self.live[k]

    def _request(self, p, a, b):
        for ev in self.events:
            if ev["path"] == p:
                ev["requests"].append((a, b))

    def warm(self, path):
        """Do a later read's expensive work now (prefetch): parse the file's symbols and, with a coprocessor, encode
        their sub-spans into its cache. Returns True if anything was computed. Changes no reader state."""
        p = os.path.abspath(path)
        with open(p, encoding="utf8", errors="replace") as f:
            text = f.read()
        if tokens(text) <= self.full_below:
            return False                                 # read whole: nothing to precompute
        key = (p, len(text), hash(text))
        fresh = key not in self.parsed
        syms = self._symbols(text, p)
        if self.coproc is None:
            return fresh
        leaf = [s for s in syms if not s.children]
        before = len(self.coproc.store)
        self.coproc.symbol_vectors(p, text.split("\n"), leaf)
        return fresh or len(self.coproc.store) > before

    def _load(self, path):
        p = os.path.abspath(path)
        with open(p, encoding="utf8", errors="replace") as f:
            text = f.read()
        lines = text.split("\n")
        old = self.files.get(p)
        changed = set()
        if old is not None and old["lines"] != lines:
            changed = changed_lines(old["lines"], lines)
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
        for _, pk, q, s, cost, x in self._compete(cands, lambda c: c[0]):
            self.menu_n += 1
            mid = f"#{self.menu_n}"
            self.menu[mid] = dict(path=q, a=s.start, b=s.end, key=(q, s.start, s.end), name=s.name, cost=cost)
            self.events.append(dict(path=q, call=self.calls, X=x[None, :], spans=[(s.start, s.end)], included=[False],
                                    requests=[], menu=mid))
            picked.append(f"  {mid} {os.path.basename(q)}:{s.name} (lines {s.start}-{s.end}, ~{cost} tok)")
        return picked

    def _need(self, p, a, b):
        if self.graph is not None:
            self.graph.need(self._cells(p, a, b))

    def _graph_menu(self, p):
        """Symbols from any file seen before: the cells most co-needed with the touched symbols' cells first, then
        the touched cells themselves (semantically adjacent at plane-0 resolution)."""
        if self.graph is None or not self.cell_of:
            return []
        cells = list(dict.fromkeys(self.cell_of[k] for _, k, _, _ in reversed(self.active) if k in self.cell_of))
        taken = {m["key"] for m in self.menu.values()}
        cands = []                                  # (co-need weight or None for the same cell, key, name, cost)
        limit = self.menu_size * (4 if self.menu_rule == "wta" else 1)
        for cell, w in self.graph.neighbours(cells, k=10) + [(c, None) for c in cells]:   # cells newest first
            for key, name in self.graph.members.get(cell, {}).items():
                q, a, b = key
                if q == p or key in taken or a in self.files.get(q, {}).get("shown", set()) or not os.path.isfile(q):
                    continue
                with open(q, encoding="utf8", errors="replace") as f:
                    cost = tokens("\n".join(f.read().split("\n")[a - 1:b]))
                taken.add(key)
                cands.append((w, key, name, cost))
                if len(cands) >= limit:
                    break
            if len(cands) >= limit:
                break
        known = [c[0] for c in cands if c[0] is not None]
        floor_w = min(known) if known else 1.0      # a same-cell item competes as the weakest co-needed one
        picked = []
        for w, key, name, cost in self._compete(cands, lambda c: c[0] if c[0] is not None else floor_w):
            q, a, b = key
            self.menu_n += 1
            mid = f"#{self.menu_n}"
            self.menu[mid] = dict(path=q, a=a, b=b, key=key, name=name, cost=cost)
            why = "same semantic cell" if w is None else f"co-needed {w:.1f}"
            picked.append(f"  {mid} {os.path.basename(q)}:{name} (lines {a}-{b}, ~{cost} tok, {why})")
        return picked

    def _compete(self, items, score):
        """The menu's selection. topk: the first menu_size. wta: scores normalised against the winner (softmax over
        log scores); only items holding at least menu_floor of the total survive, up to menu_size."""
        if self.menu_rule != "wta" or not items:
            return items[:self.menu_size]
        z = np.log(np.maximum(np.array([score(c) for c in items], float), 1e-12))
        w = np.exp(z - z.max())
        w /= w.sum()
        order = np.argsort(-w, kind="stable")
        return [items[i] for i in order if w[i] >= self.menu_floor][:self.menu_size]

    # ---------------------------------------------------------------- tools
    def _whole(self, pre, p, text, lines, full, mode):
        out = numbered(lines, 1, len(lines))
        st = self._state(p, lines, self._symbols(text, p))
        st["shown"] = set(range(1, len(lines) + 1))
        self._log(op="read", path=p, file_tokens=full, shown_tokens=tokens(out), mode=mode, included=0)
        self.served += tokens(out)
        return pre + out

    def _semantic(self, p, lines, syms, X, query, snapshot, qvec):
        """Coprocessor probabilities for the leaf symbols; registers their cells in the graph."""
        leaf = [k for k, s in enumerate(syms) if not s.children]
        ls = [syms[k] for k in leaf]
        pc = self.coproc.prob(p, lines, ls, X[leaf], prompt=query, snapshot=snapshot, trail=self._trail(), qvec=qvec)
        if self.graph is not None and self.coproc.codec is not None:
            from .graph import symbol_cells
            for s, codes in zip(ls, self.coproc.symbol_codes(p, lines, ls)):
                if len(codes):
                    cell = symbol_cells(self.coproc.codec, codes)
                    self.cell_of[(p, s.start, s.end)] = cell
                    self.graph.add_symbol(cell, (p, s.start, s.end), s.name)
        return leaf, pc

    def _mmr(self, p, lines, syms, leaf, pc, order, top=10):
        """Redundancy-aware re-ordering of the coprocessor's top `top` symbols (maximal marginal relevance): the next
        pick maximises p / p_max - mmr_beta * (highest cosine to the symbols already picked), on the coprocessor's
        own mean-pooled sub-span vectors (from its cache, no new encoding). The rest keep their order."""
        head = order[:top]
        pos = {k: i for i, k in enumerate(leaf)}
        E = self.coproc.symbol_vectors(p, lines, [syms[k] for k in head])          # [n, M, d], cached
        V = E.sum(1)
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        rel = np.array([pc[pos[k]] for k in head], float)
        rel = rel / max(rel.max(), 1e-12)
        picked, left = [], list(range(len(head)))
        while left:
            sim = (V[left] @ V[picked].T).max(1) if picked else np.zeros(len(left))
            j = left[int(np.argmax(rel[left] - self.mmr_beta * sim))]
            picked.append(j)
            left.remove(j)
        return [head[j] for j in picked] + order[top:]

    def _xray(self, syms, lines, inc, taken, query, order, st):
        """X-ray tier: skeletons (signature, lines with the query's rare words, control-flow spine) of symbols that are
        not shown in full: the top `xray_top` of the ranking, then any symbol holding a rare query word. Rare: in at
        most max(3, 2% of) the file's lines. Stops at `xray_budget` tokens."""
        qw = {w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)}
        cnt = {w: 0 for w in qw}
        for l in lines:
            for w in qw & set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", l)):
                cnt[w] += 1
        rare = {w for w, c in cnt.items() if 0 < c <= max(3, 0.02 * len(lines))}
        hits = sorted((k for k, s in enumerate(syms) if not s.children and rare & set(
            re.findall(r"[A-Za-z_][A-Za-z0-9_]*", "\n".join(lines[s.start - 1:s.end])))),
            key=lambda k: min(cnt[w] for w in rare & set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*",
                                                                   "\n".join(lines[syms[k].start - 1:syms[k].end])))))
        out, used, seen = [], 0, set()
        for k in list(order[:self.xray_top]) + hits:
            s = syms[k]
            if k in seen or inc[k] or any(a <= s.start and s.end <= b for a, b in taken):
                continue
            seen.add(k)
            keep = [ln for ln in skeleton(s, lines, rare, self.xray_lines) if ln not in st["shown"]]
            if not keep:
                continue
            runs, run = [], [keep[0]]
            for ln in keep[1:]:
                if ln == run[-1] + 1:
                    run.append(ln)
                else:
                    runs.append(run)
                    run = [ln]
            runs.append(run)
            text = f"--- x-ray of {s.name} (lines {s.start}-{s.end}; read_lines for the rest) ---\n" + \
                "\n      ...\n".join(numbered(lines, r[0], r[-1]) for r in runs)
            if used + tokens(text) > self.xray_budget:
                break
            out.append(text)
            used += tokens(text)
            st["shown"].update(keep)
        return out

    def read(self, path, query="", approve=(), snapshot="", qvec=None):
        """snapshot: the agent's current reasoning; qvec: a live hidden-state query vector (coprocessor only)."""
        pre = self.approve(approve) if approve else ""
        self._tick()
        p, text, lines, changed = self._load(path)
        changed |= self.files.get(p, {}).pop("changed", set())
        full = tokens(text)
        self._remember(idents(query))
        for ev in [e for e in self.events if e["path"] == p and "menu" not in e]:
            self._finalize(ev)
        if full <= self.full_below:
            self.last[p] = dict(mode="whole", tokens=full, rows=[])
            return self._whole(pre, p, text, lines, full, "whole")
        syms = self._symbols(text, p)
        X, _ = self._features(p, syms, lines, query, changed)
        P = self.model.prob(X)
        cop_order, pcop = [], {}
        if self.coproc is not None:
            leaf, pc = self._semantic(p, lines, syms, X, query, snapshot, qvec)
            pcop = {k: float(v) for k, v in zip(leaf, pc)}
            P = np.array(P, float)
            P[leaf] = 1 - (1 - P[leaf]) * (1 - np.asarray(pc, float))   # additive: either one can include a body
            cop_order = [leaf[i] for i in np.argsort(-np.asarray(pc, float), kind="stable")]
            if self.mmr_beta > 0:
                cop_order = self._mmr(p, lines, syms, leaf, pc, cop_order)
        head = outline(text, p, syms, None if not self.max_tokens else
                       min(OUTLINE_MAX, max(full // 2, 400), self.max_tokens // 2))
        inc, taken, cause, anchored = [False] * len(syms), [], [""] * len(syms), {}
        if self.policy == "rule_ab":
            leaf = [k for k, s in enumerate(syms) if not s.children]
            cost = [tokens("\n".join(lines[syms[k].start - 1:syms[k].end])) for k in leaf]
            pick = rule_ab(dict(cost=cost, out_tok=tokens(head), file_tok=full), [P[k] for k in leaf],
                           effective_turn(self.ctx_base + self.served, self.h_rem))
            if pick is None:
                self.last[p] = dict(mode="bypass", tokens=full, rows=[])
                return self._whole(pre, p, text, lines, full, "bypass")
            for i in pick:
                inc[leaf[i]] = True
                cause[leaf[i]] = "rule_ab"
                taken.append((syms[leaf[i]].start, syms[leaf[i]].end))
        else:
            budget, used, tau, spans = max(self.budget_min, self.budget_frac * full), 0, self.tau_now(), Spans()
            if self.max_tokens:
                budget = min(budget, self.max_tokens - tokens(head))
            for ln in anchors(lines, syms, query) if self.anchor else ():
                # the smallest leaf holding the line when it is cheap, else a window around it
                k = min((k for k, s in enumerate(syms) if not s.children and s.start <= ln <= s.end),
                        key=lambda k: syms[k].end - syms[k].start, default=None)
                a, b = (syms[k].start, syms[k].end) if k is not None else (ln, ln)
                if k is None or tokens("\n".join(lines[a - 1:b])) > min(self.anchor_max, budget / 3):
                    top = self.focus and re.match(r"\s*(?:async\s+def|def|class)\s", lines[ln - 1])
                    a, b = (ln, min(len(lines), ln + 5)) if top else (max(1, ln - 3), min(len(lines), ln + 3))
                    k = None
                cost = tokens("\n".join(lines[a - 1:b]))
                if used + cost > budget or spans.overlaps(a, b):
                    continue
                used += cost
                taken.append((a, b))
                spans.add(a, b)
                anchored[(a, b)] = ln
                if k is not None:
                    inc[k], cause[k] = True, "anchor"
            for k in sorted(range(len(syms)), key=lambda k: -P[k]):
                s = syms[k]
                if P[k] < tau or s.children:
                    continue
                cost = tokens("\n".join(lines[s.start - 1:s.end]))
                if used + cost > budget or spans.overlaps(s.start, s.end):
                    continue
                inc[k], used, cause[k] = True, used + cost, "budget"
                taken.append((s.start, s.end))
                spans.add(s.start, s.end)
        # the coprocessor's own pick: its top-ranked symbols go up front when small, whatever the thresholds say (its
        # calibrated probabilities are honest but low, ~1-2 needed of 100+ symbols, so a threshold never lets it act)
        for k in cop_order[:self.coproc_top]:
            s = syms[k]
            if inc[k] or any(not (s.end < a or s.start > b) for a, b in taken):
                continue
            if tokens("\n".join(lines[s.start - 1:s.end])) <= self.coproc_max:
                inc[k], cause[k] = True, "coproc_top"
                taken.append((s.start, s.end))
        parts, first, wh = [head], [], Where(syms)
        st = self._state(p, lines, syms)
        st["shown"] = set()
        self.reads += 1
        for a, b in sorted(taken):
            body = numbered(lines, a, b)
            h = hash(body)
            seen = self.live.get((p, a, b)) if self.refractory else None
            if seen is not None and seen[1] == h:              # refractory: already in context, unchanged
                parts.append(f"--- lines {a}-{b}: unchanged since read #{seen[0]} above ---")
            else:
                ln = anchored.get((a, b))
                if self.focus and ln is not None:     # small models: what matches the request comes first
                    s = wh.inner(a)
                    if s is not None:                 # the enclosing def line on top: a raise shows whose it is
                        body = numbered(lines, s.start, s.start) + "\n   ...\n" + body
                    first.append(f"--- matches your request: lines {a}-{b}{wh(ln, ln)} ---\n" + body)
                else:
                    parts.append(f"--- lines {a}-{b}{wh(a, b)} ---\n" + body)
                if self.refractory:
                    self.live[(p, a, b)] = (self.reads, h)
            st["shown"].update(range(a, b + 1))
            self._remember(idents("\n".join(lines[a - 1:b])), 0.5)
        self._touch_many(p, sorted(taken), 0.5)
        if first:
            top, _, rest = head.partition("\n")
            parts = [top] + first + [rest] + parts[1:]
        if self.xray_top:
            parts += self._xray(syms, lines, inc, taken, query, cop_order or
                                sorted((k for k, s in enumerate(syms) if not s.children), key=lambda k: -P[k]), st)
        self.events.append(dict(path=p, call=self.calls, X=X, spans=[(s.start, s.end) for s in syms], included=inc,
                                requests=[]))
        menu = self._graph_menu(p) + self._menu(p, query)
        if menu:
            parts.append("[connected] symbols in neighbouring files; pull with cortex_approve(ids) instead of reading "
                         "those files:\n" + "\n".join(menu))
        out = "\n".join(parts)
        self.last[p] = dict(mode="outline", tokens=full, shown=tokens(out), policy=self.policy, tau=self.tau_now(),
                            menu=len(menu), rows=[dict(name=s.name, start=s.start, end=s.end, p=float(P[k]),
                                                       pc=pcop.get(k), included=inc[k], cause=cause[k])
                                                  for k, s in enumerate(syms) if not s.children])
        self._log(op="read", path=p, file_tokens=full, shown_tokens=tokens(out), mode="outline", included=sum(inc),
                  symbols=len(syms), menu=len(menu), menu_tokens=tokens("\n".join(menu)) if menu else 0)
        self.served += tokens(out)
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
        self._need(p, a, b)
        self._log(op="range", path=p, file_tokens=tokens(text), shown_tokens=tokens(out), miss=miss, start=a, end=b)
        self.served += tokens(out)
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
            self._need(p, a, b)
            self._log(op="approve", path=p, file_tokens=tokens(text), shown_tokens=tokens(body), id=mid)
            self.served += tokens(body)
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
