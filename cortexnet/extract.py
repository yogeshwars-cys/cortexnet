"""User extract: what a request names, handed on before the agent's first model call.

`extract(text, prefetcher, ws)` reads the user's message for the repository files it names, the repository symbols it
names (with the file and line where each is defined, from the prefetcher's symbol index), quoted text (error messages,
literals) and the kind of task. The CLI's `Agent.bridge()` then hands that on while the model's first call is in flight:

  - the cache manager (prefetch.py) warms the files the request points at: parsed and, with a coprocessor, their
    symbols' sub-span codes encoded, the expensive part of a CortexNET read;
  - the coprocessor encodes the request into its query-embedding cache (coproc.py), so the turn's reads reuse it;
  - the turn's user message carries a one-line note of where the named things are defined, which can save the model
    a grep or list round trip. It names locations only, never what the code does.

Nothing here reads file bodies or calls the model; the extract itself costs a regex pass over the message.
"""
import os
import re
from dataclasses import dataclass, field

from .prefetch import COMMON, IDENT_RX, PATH_RX

MARKED_RX = re.compile(r"\b([A-Za-z_]\w*)\s*\(|\b(?:def|class|function|method|func|fn|symbol|struct|interface)\s+"
                       r"`?([A-Za-z_]\w*)")
QUOTE_RX = re.compile(r"`([^`\n]{3,200})`|\"([^\"\n]{6,200})\"|'([^'\n]{6,200})'")
KINDS = (("fix", r"\b(fix|bug|broken|fail\w*|error|crash\w*|wrong|regression|traceback|exception)\b"),
         ("test", r"\b(tests?|pytest|unittest|coverage|spec)\b"),
         ("add", r"\b(add|implement|create|support|new|feature|write)\b"),
         ("refactor", r"\b(refactor|rename|clean ?up|simplify|move|extract)\b"),
         ("explain", r"\b(how|why|what|where|explain|understand|walk me|describe)\b"))


@dataclass
class Extract:
    text: str
    kind: str = "other"
    paths: list = field(default_factory=list)       # repository files the request names (relative paths)
    symbols: list = field(default_factory=list)     # [(name, rel, line)] defined in at most 2 files
    quoted: list = field(default_factory=list)      # quoted strings: error messages, literals
    files: list = field(default_factory=list)       # files to warm, most likely first

    def note(self, max_items=6):
        """A one-line note for the model: where the named things are. Empty when nothing was found."""
        items = [f"{p}" for p in self.paths] + [f"{n} -> {r}:{ln}" for n, r, ln in self.symbols]
        if not items:
            return ""
        more = f" (+{len(items) - max_items} more)" if len(items) > max_items else ""
        return ("[CortexNET: your request names " + "; ".join(items[:max_items]) + more
                + ". Locations only, from the symbol index; nothing has been read yet.]")

    def summary(self):
        bits = [f"{len(self.paths)} file{'s' * (len(self.paths) != 1)}" if self.paths else "",
                f"{len(self.symbols)} symbol{'s' * (len(self.symbols) != 1)}" if self.symbols else "",
                f"warming {', '.join(self.files)}" if self.files else ""]
        return f"{self.kind} request: " + ", ".join(b for b in bits if b) if any(bits) else ""


def extract(text, prefetcher=None, ws=None, wait=0.3, max_files=3):
    """The request's files, symbols, quotes and kind. `wait`: seconds to let the background symbol index finish."""
    x = Extract(text=text or "")
    low = x.text.lower()
    for kind, rx in KINDS:
        if re.search(rx, low):
            x.kind = kind
            break
    x.quoted = [next(g for g in m.groups() if g) for m in QUOTE_RX.finditer(x.text)][:5]
    if prefetcher is None:
        return x
    prefetcher.ready.wait(wait)
    rels = list(prefetcher.rels)
    for m in PATH_RX.findall(x.text):
        m = m.replace("\\", "/").strip("./")
        hits = [r for r in rels if r == m or r.endswith("/" + m)]
        if len(hits) == 1 and hits[0] not in x.paths:
            x.paths.append(hits[0])
    if ws is not None:                                   # a path to a file the index skipped (not source) still counts
        for m in PATH_RX.findall(x.text):
            try:
                p = ws.path(m)
            except ValueError:
                continue
            rel = ws.rel(p)
            if rel not in x.paths and _is_file(p):
                x.paths.append(rel)
    # A plain English word is a symbol only when the request marks it as code: "explain prefetch command" once matched a
    # command() helper in bench/ and sent the agent there. Code-shaped words always count; a plain word that names
    # exactly one file's stem ("prefetch" -> cortexnet/prefetch.py) counts as that file.
    marked = {m.group(1) or m.group(2) for m in MARKED_RX.finditer(x.text)} | \
        {w for q in re.findall(r"`([^`\n]+)`", x.text) for w in IDENT_RX.findall(q)}
    stems = {}
    for r in rels:
        stems.setdefault(os.path.splitext(os.path.basename(r))[0], []).append(r)
    seen = set()
    for m in IDENT_RX.finditer(x.text):
        w = m.group(0)
        if w in COMMON or w in seen or w.lower() in ("this", "that", "with", "from", "what", "where"):
            continue
        seen.add(w)
        opens = not x.text[:m.start()].strip() or x.text[:m.start()].rstrip()[-1] in ".!?"
        capital = w[0].isupper() and not opens          # "Reader" mid-sentence is a name; "Can" opening one is not
        if not (code_like(w) or capital or w in marked):
            hits = stems.get(w, [])
            if len(hits) == 1 and hits[0] not in x.paths:
                x.paths.append(hits[0])
            continue
        where = prefetcher.locate(w)
        if 0 < len(where) <= 2:
            for rel, line in sorted(where.items()):
                x.symbols.append((w, rel, line))
    score = {}
    for _, rel, _ in x.symbols:
        score[rel] = score.get(rel, 0) + 1
    x.files = (x.paths + [r for r, _ in sorted(score.items(), key=lambda kv: -kv[1]) if r not in x.paths])[:max_files]
    return x


def code_like(w):
    """snake_case, CamelCase / mixedCase, or with digits: not an ordinary English word."""
    return "_" in w or any(c.isupper() for c in w[1:]) or any(c.isdigit() for c in w)


def _is_file(p):
    return os.path.isfile(p)
