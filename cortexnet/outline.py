"""File sketches: the cheap tier a model reads before it asks for line ranges.

A sketch lists the file's symbols with their line spans and signatures, so the model can see where things are
without paying for the bodies. Python goes through `ast`; other text files use line patterns; anything with no
recognisable structure falls back to fixed-size sections labelled by their first line.
"""
import ast
import textwrap
import re
from dataclasses import dataclass, field


def tokens(text):
    return max(1, len(text) // 4)


@dataclass
class Symbol:
    name: str
    kind: str
    start: int          # 1-based, inclusive
    end: int
    head: str           # signature line(s) as written
    doc: str = ""
    depth: int = 0
    refs: set = field(default_factory=set)
    children: bool = False      # another symbol is nested inside this one

    def label(self):
        d = f"  # {self.doc}" if self.doc else ""
        return f"{'  ' * self.depth}L{self.start}-{self.end} {self.head}{d}"


def _py(src, lines):
    tree = ast.parse(src)
    out = []

    def head(node):
        first = getattr(node, "decorator_list", None)
        s = min([d.lineno for d in first] + [node.lineno]) if first else node.lineno
        body = node.body[0].lineno if getattr(node, "body", None) else node.lineno
        h = " ".join(l.strip() for l in lines[s - 1:max(s, body - 1)] if l.strip())
        return (h[:157] + "...") if len(h) > 160 else h, s

    def walk(body, depth):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                h, s = head(node)
                doc = (ast.get_docstring(node) or "").strip().split("\n")[0][:80]
                kind = "class" if isinstance(node, ast.ClassDef) else "def"
                sym = Symbol(node.name, kind, s, node.end_lineno, h, doc, depth)
                sym.refs = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
                           {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
                out.append(sym)
                if isinstance(node, ast.ClassDef) and depth < 1:
                    walk(node.body, depth + 1)
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if imports:
        names = []
        for n in imports:
            names += [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else [n.module or "."]
        out.append(Symbol("imports", "imports", imports[0].lineno, imports[-1].end_lineno,
                          "imports: " + ", ".join(dict.fromkeys(names))[:150]))
    consts = [n for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))]
    for n in consts:
        tg = n.targets if isinstance(n, ast.Assign) else [n.target]
        nm = ", ".join(t.id for t in tg if isinstance(t, ast.Name))
        if nm:
            out.append(Symbol(nm, "const", n.lineno, n.end_lineno, lines[n.lineno - 1].strip()[:120]))
    walk(tree.body, 0)
    return sorted(out, key=lambda s: (s.start, s.depth))


DEF = re.compile(
    r"^(?P<ind>\s*)(?:export\s+)?(?:default\s+)?(?:public\s+|private\s+|protected\s+|static\s+|async\s+|override\s+|"
    r"inline\s+|suspend\s+|pub(?:\(\w+\))?\s+)*"
    r"(?:(?:function|def|fn|func|fun|class|struct|enum|interface|trait|impl|object|module|type)\s+[\w<>:.,\s]*?\w"
    r"|(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>"
    r"|[\w:<>*&\[\]]+\s+[\w:~]+\s*\([^;{}]*\)\s*(?:const\s*)?\{?\s*$)")
HEADING = re.compile(r"^(#{1,6})\s+\S")


def _generic(lines, path):
    out = []
    md = path.lower().endswith((".md", ".markdown", ".rst", ".txt"))
    starts = []
    for i, l in enumerate(lines, 1):
        if md:
            m = HEADING.match(l)
            if m:
                starts.append((i, len(m.group(1)) - 1, l.strip()))
        else:
            m = DEF.match(l)
            if m and not l.strip().startswith(("//", "#", "*", "return", "else", "if", "for", "while")):
                starts.append((i, len(m.group("ind").expandtabs(4)) // 2, l.strip()))
    for k, (i, d, h) in enumerate(starts):
        end = len(lines)
        for j, dj, _ in starts[k + 1:]:
            if dj <= d:
                end = j - 1
                break
        while end > i and not lines[end - 1].strip():
            end -= 1
        out.append(Symbol(h[:60], "heading" if md else "def", i, end, h[:160], depth=min(d, 3)))
    return out


def sections(lines, size=40):
    out = []
    for s in range(1, len(lines) + 1, size):
        e = min(len(lines), s + size - 1)
        first = next((l.strip() for l in lines[s - 1:e] if l.strip()), "")
        out.append(Symbol(f"section {s}", "section", s, e, first[:100]))
    return out


COMPOUND = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match)


def _py_blocks(src_lines, s):
    """Statement groups inside a big Python function: compound statements stand alone, runs of simple ones merge."""
    try:
        node = ast.parse(textwrap.dedent("\n".join(src_lines[s.start - 1:s.end]).expandtabs(4)))
    except SyntaxError:
        node = None
    body = None
    if node and node.body and isinstance(node.body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        body = node.body[0].body
    if not body:
        return None
    off = s.start - 1
    spans, run = [], None
    for st in body:
        a, b = st.lineno + off, st.end_lineno + off
        if isinstance(st, COMPOUND):
            if run:
                spans.append(run)
                run = None
            spans.append([a, b])
        elif run and b - run[0] + 1 > 20:          # long runs of simple statements are cut at ~20 lines
            spans.append(run)
            run = [a, b]
        else:
            run = [run[0], b] if run else [a, b]
    if run:
        spans.append(run)
    return spans


def _para_blocks(lines, s, min_lines=6, max_lines=40):
    """Blank-line paragraphs inside a big symbol, small ones merged forward, long ones cut into chunks."""
    spans, cur = [], None
    for i in range(s.start + 1, s.end + 1):
        blank = not lines[i - 1].strip()
        if blank:
            if cur and cur[1] - cur[0] + 1 >= min_lines:
                spans.append(cur)
                cur = None
            continue
        cur = [cur[0], i] if cur else [i, i]
    if cur:
        spans.append(cur)
    out = []
    for a, b in spans:
        while b - a + 1 > max_lines:
            out.append([a, a + max_lines - 1])
            a += max_lines
        out.append([a, b])
    return out


def split_big(syms, lines, path, min_tokens):
    """Add labelled sub-blocks under every leaf symbol bigger than min_tokens, so a miss can fetch a block."""
    out = []
    for s in syms:
        out.append(s)
        if s.children or s.kind in ("imports", "const") or tokens("\n".join(lines[s.start - 1:s.end])) <= min_tokens:
            continue
        spans = (_py_blocks(lines, s) if path.lower().endswith(".py") and s.kind in ("def", "class") else None) \
            or _para_blocks(lines, s)
        merged = []
        for a, b in spans:                       # fold tiny blocks into a small neighbour, keep real blocks apart
            prev = merged[-1][1] - merged[-1][0] + 1 if merged else 0
            if merged and ((b - a + 1 < 4 and prev < 15) or prev < 4):
                merged[-1][1] = b
            else:
                merged.append([a, b])
        spans = merged
        if len(spans) < 2:
            continue
        for a, b in spans:
            head = next((l.strip() for l in lines[a - 1:b] if l.strip()), "")[:80]
            out.append(Symbol(f"{s.name}:{a}", "block", a, b, head, depth=s.depth + 1))
        s.children = True
    return out


def symbols(text, path="", split_tokens=0):
    lines = text.split("\n")
    syms = []
    if path.lower().endswith(".py"):
        try:
            syms = _py(text, lines)
        except (SyntaxError, ValueError):
            syms = []
    if not syms:
        syms = _generic(lines, path)
    covered = sum(s.end - s.start + 1 for s in syms if s.depth == 0)
    if not syms or covered < 0.3 * len(lines):
        syms = sorted(syms + sections(lines), key=lambda s: (s.start, s.depth))
    for s in syms:
        s.children = any(c is not s and c.depth > s.depth and s.start <= c.start <= s.end for c in syms)
    if split_tokens:
        syms = split_big(syms, lines, path, split_tokens)
    return syms


def outline(text, path="", syms=None):
    syms = syms if syms is not None else symbols(text, path)
    n = text.count("\n") + 1
    head = f"{path or '<file>'}: {n} lines, ~{tokens(text)} tokens. Outline (ask for line ranges to see bodies):"
    return "\n".join([head] + [s.label() for s in syms])


FLOW = re.compile(r"^\s*(if|elif|else|for|while|try|except|finally|with|return|raise|yield|match|case|switch|catch|"
                  r"throw|break|continue)\b")


def skeleton(s, lines, words=(), max_lines=12, window=2):
    """An x-ray of a symbol: line numbers to show, at most max_lines, by priority: the signature (until the line that
    closes it, at most 4 lines), lines holding any of `words` with `window` lines around them, then the control-flow
    spine (if/for/return/raise..., shallowest first). Lines are 1-based, sorted."""
    a, b = s.start, s.end
    pick = []

    def add(ln):
        if a <= ln <= b and ln not in pick and len(pick) < max_lines:
            pick.append(ln)
    for ln in range(a, min(b, a + 3) + 1):
        add(ln)
        if lines[ln - 1].rstrip().endswith((":", "{")):
            break
    words = {w for w in words if w}
    for ln in range(a, b + 1):
        if words and words & set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", lines[ln - 1])):
            for k in range(ln - window, ln + window + 1):
                add(k)
    depth = lambda ln: len(lines[ln - 1]) - len(lines[ln - 1].lstrip())  # noqa: E731
    for ln in sorted((ln for ln in range(a + 1, b + 1) if FLOW.match(lines[ln - 1])), key=lambda ln: (depth(ln), ln)):
        add(ln)
    return sorted(pick)
