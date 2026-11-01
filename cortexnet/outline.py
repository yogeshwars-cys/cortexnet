"""File sketches: the cheap tier a model reads before it asks for line ranges.

A sketch lists the file's symbols with their line spans and signatures, so the model can see where things are
without paying for the bodies. Python goes through `ast`; other text files use line patterns; anything with no
recognisable structure falls back to fixed-size sections labelled by their first line.
"""
import ast
import bisect
import textwrap
import re
import warnings
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
    with warnings.catch_warnings():         # e.g. invalid escape sequences in old code: not the reader's to report
        warnings.simplefilter("ignore", SyntaxWarning)
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
                sym.refs = {n.id if isinstance(n, ast.Name) else n.attr for n in ast.walk(node)
                            if isinstance(n, (ast.Name, ast.Attribute))}
                out.append(sym)
                if isinstance(node, ast.ClassDef) and depth < 1:
                    walk(node.body, depth + 1)
            elif depth and isinstance(node, (ast.Assign, ast.AnnAssign)):   # class attributes (name = "design", ...)
                tg = node.targets if isinstance(node, ast.Assign) else [node.target]
                nm = ", ".join(t.id for t in tg if isinstance(t, ast.Name))
                if nm:
                    out.append(Symbol(nm, "const", node.lineno, node.end_lineno, lines[node.lineno - 1].strip()[:120],
                                      depth=depth))
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
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


def _mark_children(syms):
    """s.children = some deeper symbol starts inside s's span: a range-max query over the symbols sorted by start
    (a sparse table), so a file with tens of thousands of symbols stays linear-log instead of quadratic."""
    order = sorted(syms, key=lambda s: s.start)
    starts = [s.start for s in order]
    table = [[s.depth for s in order]]
    while 2 ** len(table) <= len(order):
        prev, h = table[-1], 2 ** (len(table) - 1)
        table.append([max(prev[i], prev[i + h]) for i in range(len(prev) - h)])
    for s in syms:
        lo, hi = bisect.bisect_left(starts, s.start), bisect.bisect_right(starts, s.end)
        k = (hi - lo).bit_length() - 1
        s.children = hi > lo and max(table[k][lo], table[k][hi - 2 ** k]) > s.depth


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
    _mark_children(syms)
    if split_tokens:
        syms = split_big(syms, lines, path, split_tokens)
    return syms


OUTLINE_MAX = 4000      # tokens: an outline never costs more than this, nor more than half the file


def _cost(lines):
    return sum(tokens(l) + 1 for l in lines)


def _listing(group, depth, what, width):
    """Lines of `name:line` entries for a group of symbols, each line at most `width` characters (0 = no limit)."""
    head = f"{'  ' * depth}L{group[0].start}-{max(s.end for s in group)} {len(group)} {what}: "
    items = [f"{s.name}:{s.start}" for s in group]
    if not width:
        return [head + ", ".join(items)]
    out, cur = [], head
    for k, it in enumerate(items):
        if len(cur) + len(it) + 2 > width and cur != head:
            if len(out) + 1 >= max(1, width // 40):           # this group's share of the budget is used up
                out.append(cur.rstrip(", ") + f", ... {len(items) - k} more")
                return out
            out.append(cur.rstrip(", "))
            cur = f"{'  ' * (depth + 1)}"
        cur += it + ", "
    out.append(cur.rstrip(", "))
    return out


def _kind_word(group):
    kinds = dict.fromkeys(s.kind for s in group)
    return f"{next(iter(kinds))}s" if len(kinds) == 1 else "symbols"


def fold(syms, max_tokens):
    """Outline lines that fit in max_tokens, giving up the least useful detail first: sub-blocks; then the
    signatures of a class's members (kept as `name:line`, enough to ask for a range); then runs of same-kind
    top-level symbols (listed the same way); then the length of every list, trimmed evenly. Returns (lines, folded)."""
    lines = [s.label() for s in syms]
    if _cost(lines) <= max_tokens:
        return lines, False
    syms = [s for s in syms if s.kind != "block"]
    lines = [s.label() for s in syms]
    if _cost(lines) <= max_tokens:
        return lines, True
    groups, k = [], 0                   # (top-level symbol, its members) in file order
    while k < len(syms):
        j = k + 1
        while j < len(syms) and syms[j].depth > syms[k].depth and syms[j].start <= syms[k].end:
            j += 1
        groups.append((syms[k], syms[k + 1:j]))
        k = j

    def render(width, runs):
        out, i = [], 0
        while i < len(groups):
            top, members = groups[i]
            j = i + 1
            if runs and not members:            # a run of 3+ same-kind top-level symbols without members
                while j < len(groups) and not groups[j][1] and groups[j][0].kind == top.kind:
                    j += 1
            if j - i >= 3:
                run = [g[0] for g in groups[i:j]]
                out += _listing(run, 0, _kind_word(run), width)
            else:
                j = i + 1
                out.append(top.label())
                if len(members) <= 2:                   # a listing would save nothing
                    out += [m.label() for m in members]
                else:
                    out += _listing(members, top.depth + 1, "members", width)
            i = j
        return out
    for runs in (False, True):
        lines = render(0, runs)
        if _cost(lines) <= max_tokens:
            return lines, True
    lo, hi, best = 40, 4000, None                # the widest list lines that still fit
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = render(mid, True)
        if _cost(cand) <= max_tokens:
            best, lo = cand, mid + 1
        else:
            hi = mid - 1
    if best is None:                             # not even one short line per group: fixed groups of symbols
        top = [g[0] for g in groups]
        per = max(3, -(-len(top) // max(1, max_tokens // 50)))
        best = [l for k in range(0, len(top), per) for l in _listing(top[k:k + per], 0, _kind_word(top[k:k + per]), 120)]
        while best and _cost(best) > max_tokens:
            best.pop()
    return best, True


def outline(text, path="", syms=None, max_tokens=None):
    """max_tokens: fold the outline to fit (default: OUTLINE_MAX, and at most half the file, or 400 tokens for small
    files); 0 = never fold."""
    syms = syms if syms is not None else symbols(text, path)
    n = text.count("\n") + 1
    full = tokens(text)
    cap = min(OUTLINE_MAX, max(full // 2, 400)) if max_tokens is None else max_tokens
    lines, folded = fold(syms, cap) if cap else ([s.label() for s in syms], False)
    how = (f"folded to ~{cap} tokens: runs of symbols are listed by name, ask for line ranges to see more"
           if folded else "ask for line ranges to see bodies")
    head = f"{path or '<file>'}: {n} lines, ~{full} tokens. Outline ({how}):"
    return "\n".join([head] + lines)


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
