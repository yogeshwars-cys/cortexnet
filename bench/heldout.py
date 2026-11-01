"""Held-out session stress test: machine-written questions about repositories CortexNET was never tuned on.

The 20 questions of bench/live_ab.py were written by hand about this repository and the standard library, by the same
people who tuned the reader on them. This harness removes both: the questions come from the AST of package subtrees of
SWE-bench Lite repositories (each pinned at an instance's base commit), and only facts with one unambiguous answer are
kept. Each session asks one subtree's questions in one conversation per arm, as live_ab --session does.

  python -I bench/heldout.py gen                       # write bench/heldout/questions.jsonl (deterministic)
  python -I bench/heldout.py run --openai <url>/v1 --ctx 8192 --rep 0 [--subtrees django,sympy] [--arms ...]

Arms: baseline = whole-file reads; paged = the usual agent file tool without CortexNET (first 200 lines, then line
ranges); cortex = CortexNET reads; +fade = watermark fade on top (cortex+fade against paged+fade isolates the read).
A CortexNET read and a page are capped alike at a quarter of the context window (live_ab.READ_MAX).
  python -I bench/heldout.py report                    # bench/results/heldout.md from every run's jsonl

Question kinds (all checked against the AST, the answer graded by a regular expression on the reply):
  default   the default value of a parameter of a uniquely named function or method (numbers and strings only;
            None/True/False/0/1 and one-letter strings are left out: too easy to guess)
  const     the value a uniquely assigned module or class constant is set to
  raises    which function raises an error whose message starts with a given text (message unique in the subtree)
  base      the first base class of a class (object / Exception and the like left out)
Files below 1,500 tokens are skipped: every arm reads them whole, so they only add ties.
"""
import argparse
import ast
import glob
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import live_ab  # noqa: E402
from cortexnet.outline import tokens  # noqa: E402

SNAP = os.path.join(os.environ.get("CORTEXNET_SWE", "D:/dev-cache/swe"), "snap")
OUT = os.path.join(HERE, "heldout")
QFILE = os.path.join(OUT, "questions.jsonl")
RUNS = os.path.join(HERE, "results", "traces", "heldout")
# name: (snapshot = instance at whose base commit the repository is pinned, package subtree)
SUBTREES = {
    "django": ("django__django-11001", "django/db/models"),
    "sympy": ("sympy__sympy-11400", "sympy/core"),
    "xarray": ("pydata__xarray-4094", "xarray/core"),
    "sklearn": ("scikit-learn__scikit-learn-10297", "sklearn/linear_model"),
    "pylint": ("pylint-dev__pylint-7080", "pylint/checkers"),
    "sphinx": ("sphinx-doc__sphinx-10451", "sphinx/domains"),
    "pytest": ("pytest-dev__pytest-11143", "src/_pytest"),
    "seaborn": ("mwaskom__seaborn-3190", "seaborn"),
}
# v2, the test set: v1's failures were studied to improve the reader (anchored reads, class attributes in the
# outline), which makes v1 a development set. v2 takes 8 other subtrees nobody looked at, and half its questions
# name code without backticks, as people often do.
SUBTREES_V2 = {
    "fits": ("astropy__astropy-14365", "astropy/io/fits"),
    "flask": ("pallets__flask-4992", "src/flask"),
    "sphinxutil": ("sphinx-doc__sphinx-10451", "sphinx/util"),
    "forms": ("django__django-11039", "django/forms"),
    "functions": ("sympy__sympy-11870", "sympy/functions"),
    "preprocessing": ("scikit-learn__scikit-learn-10508", "sklearn/preprocessing"),
    "backends": ("pydata__xarray-3364", "xarray/backends"),
    "axes": ("matplotlib__matplotlib-23299", "lib/matplotlib/axes"),
}
# v3, multi-hop for the small-model workflow test: every question names a class in one file and asks about its base
# class, which is defined in another file of the package (and the answer is nowhere in the first file), so each
# answer takes a lookup in two files. Subtrees used by neither v1 nor v2.
SUBTREES_V3 = {
    "admin": ("django__django-11001", "django/contrib/admin"),
    "printing": ("sympy__sympy-11400", "sympy/printing"),
    "modeling": ("astropy__astropy-12907", "astropy/modeling"),
    "autodoc": ("sphinx-doc__sphinx-10325", "sphinx/ext"),
    "extensions": ("pylint-dev__pylint-6506", "pylint/extensions"),
    "featext": ("scikit-learn__scikit-learn-10949", "sklearn/feature_extraction"),
    "plot": ("pydata__xarray-4248", "xarray/plot"),
    "mpl": ("matplotlib__matplotlib-22711", "lib/matplotlib/backends"),
}
SET = "v1"


def use_set(name):
    global QFILE, RUNS, SUBTREES, SET, PER_SESSION
    if name in ("v2", "v3"):
        QFILE = os.path.join(OUT, f"questions_{name}.jsonl")
        RUNS = os.path.join(HERE, "results", "traces", f"heldout_{name}")
        SUBTREES = SUBTREES_V2 if name == "v2" else SUBTREES_V3
    if name == "v3":
        PER_SESSION = 60
    SET = name


def hop_facts(files, repo):
    """Two-file questions: class C (file F) has a first base B defined in another file G; ask for a default of a
    parameter of one of its methods, or a class attribute it sets, that C does not define itself. B is not named and
    the answer does not occur in C's source, so it takes a lookup in G."""
    classes, texts = {}, {}
    for p in files:
        with open(p, encoding="utf8", errors="replace") as f:
            src = f.read()
        rel = os.path.relpath(p, repo).replace("\\", "/")
        texts[rel] = src
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                classes.setdefault(n.name, []).append((rel, n))
    out = []
    for name, defs in classes.items():
        if len(defs) != 1:
            continue
        rel, c = defs[0]
        b0 = c.bases[0] if c.bases else None
        bname = b0.id if isinstance(b0, ast.Name) else b0.attr if isinstance(b0, ast.Attribute) else None
        if not bname:
            continue
        own = {m.name for m in c.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))} | \
            {t.id for m in c.body if isinstance(m, ast.Assign) for t in m.targets if isinstance(t, ast.Name)}
        c_src = "\n".join(texts[rel].split("\n")[c.lineno - 1:c.end_lineno])
        base = classes.get(bname)
        if not base or len(base) != 1 or base[0][0] == rel:
            continue
        brel, b = base[0]
        cands = []
        for m in b.body:
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name not in own:
                args = m.args
                pos = args.posonlyargs + args.args
                pairs = list(zip(pos[len(pos) - len(args.defaults):], args.defaults)) + \
                    [(x, d) for x, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
                for x, d in pairs:
                    if isinstance(d, ast.Constant) and literal_rx(d.value):
                        cands.append(dict(kind="hop_default", answer=repr(d.value), rx=literal_rx(d.value),
                                          q=f"In {rel}, the class `{name}` has a first base class that is defined "
                                            f"in another file of this package. What is the default value of the "
                                            f"`{x.arg}` parameter of that base class's `{m.name}` method?"))
            elif isinstance(m, ast.Assign) and len(m.targets) == 1 and isinstance(m.targets[0], ast.Name) \
                    and m.targets[0].id not in own and isinstance(m.value, ast.Constant) and literal_rx(m.value.value):
                cands.append(dict(kind="hop_const", answer=repr(m.value.value), rx=literal_rx(m.value.value),
                                  q=f"In {rel}, the class `{name}` has a first base class that is defined in another "
                                    f"file of this package. What value does that base class set `{m.targets[0].id}` "
                                    f"to?"))
        for f in cands:
            if re.search(f["rx"], c_src, re.I) or re.search(f["rx"], f["q"], re.I) or \
                    any(re.search(f["rx"], d, re.I) for d in DISTRACTORS):
                continue                                 # the answer must take the hop to the base class's file
            f.update(file=rel, base_file=brel, file_tokens=tokens(texts[rel]))
            out.append(f)
    # follow a call: f (file F) calls g, a top-level function defined in exactly one other file G
    funcs = {}
    for rel, src in texts.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.setdefault(n.name, []).append((rel, n))
    seen = set()
    for rel, src in texts.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(src)
        except SyntaxError:
            continue
        local = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        callers = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))] + \
            [m for c in tree.body if isinstance(c, ast.ClassDef) for m in c.body
             if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for fn in callers:
            f_src = "\n".join(src.split("\n")[fn.lineno - 1:fn.end_lineno])
            for call in (x for x in ast.walk(fn) if isinstance(x, ast.Call)):
                g = call.func.id if isinstance(call.func, ast.Name) else \
                    call.func.attr if isinstance(call.func, ast.Attribute) else None
                defs = funcs.get(g, [])
                if not g or g in local or len(defs) != 1 or defs[0][0] == rel or (rel, g) in seen:
                    continue
                grel, gn = defs[0]
                args = gn.args
                pos = args.posonlyargs + args.args
                pairs = list(zip(pos[len(pos) - len(args.defaults):], args.defaults)) + \
                    [(x, d) for x, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
                for x, d in pairs:
                    rx = literal_rx(d.value) if isinstance(d, ast.Constant) else None
                    if not rx or re.search(rx, f_src, re.I) or any(re.search(rx, t, re.I) for t in DISTRACTORS):
                        continue
                    q = (f"In {rel}, `{fn.name}` calls `{g}`, which is defined in another file of this package. "
                         f"What is the default value of the `{x.arg}` parameter of `{g}`?")
                    if re.search(rx, q, re.I):
                        continue
                    seen.add((rel, g))
                    out.append(dict(kind="hop_call", answer=repr(d.value), rx=rx, q=q, file=rel, base_file=grel,
                                    file_tokens=tokens(src)))
                    break
    return out


PER_SESSION = 40
MIN_FILE_TOKENS = 1500
MIX = {"default": 0.35, "const": 0.2, "raises": 0.25, "base": 0.2}
TRIVIAL_BASES = {"object", "Exception", "ValueError", "TypeError", "RuntimeError", "Enum", "NamedTuple", "Protocol",
                 "Generic", "ABC", "dict", "list", "tuple", "str", "int", "type"}
DISTRACTORS = ["ANSWER: The default value is None.", "ANSWER: I could not find it in the file.",
               "ANSWER: It is set to True.", "ANSWER: The method raises a ValueError.",
               "ANSWER: The class inherits from object.", "ANSWER: It is an empty string.",
               "ANSWER: The function is defined in the module and returns a value."]


def py_files(root):
    for p in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        rel = os.path.relpath(p, root).replace("\\", "/")
        if "/tests/" in "/" + rel or os.path.basename(p).startswith("test_") or "/testing/" in "/" + rel:
            continue
        yield p


def literal_rx(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return None if abs(v) < 2 else r"(?<![\w.])" + re.escape(str(v)) + r"(?![\w])"
    if isinstance(v, float):
        r = repr(v)
        if "e" in r or "inf" in r or "nan" in r or v in (0.0, 1.0):
            return None
        return r"(?<![\w.])" + re.escape(r).replace(r"0\.", r"0?\.", 1) + r"(?![\d])"
    if isinstance(v, str):
        if not (3 <= len(v) <= 40) or not re.search(r"[A-Za-z]", v) or "\n" in v or not v.isprintable():
            return None
        return r"(?<![\w])" + re.escape(v) + r"(?![\w])"
    return None


def name_rx(n):
    return r"(?<![\w])" + re.escape(n) + r"(?![\w])"


def facts(path, rel, subtree_text):
    """Every askable fact of one file: dicts with kind, q, rx, answer."""
    with open(path, encoding="utf8", errors="replace") as f:
        src = f.read()
    if tokens(src) < MIN_FILE_TOKENS:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return []
    out, funcs, assigns = [], {}, {}

    def scope(body, cls=None):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.setdefault(f"{cls}.{n.name}" if cls else n.name, []).append(n)
            elif isinstance(n, ast.ClassDef) and cls is None:
                funcs.setdefault(n.name, []).append(n)
                scope(n.body, n.name)
            elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                assigns.setdefault(f"{cls}.{n.targets[0].id}" if cls else n.targets[0].id, []).append(n)
    scope(tree.body)
    for qual, nodes in funcs.items():
        if len(nodes) != 1:
            continue
        n = nodes[0]
        if isinstance(n, ast.ClassDef):
            if n.bases:
                b = n.bases[0]
                base = b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else None
                if base and base not in TRIVIAL_BASES and base != n.name:
                    out.append(dict(kind="base", answer=base, rx=name_rx(base),
                                    q=f"In {rel}, what is the first base class of the class `{qual}`?"))
            continue
        what = "method" if "." in qual else "function"
        args = n.args
        pos = args.posonlyargs + args.args
        pairs = list(zip(pos[len(pos) - len(args.defaults):], args.defaults)) + \
            [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
        for a, d in pairs:
            if isinstance(d, ast.Constant):
                rx = literal_rx(d.value)
                if rx:
                    out.append(dict(kind="default", answer=repr(d.value), rx=rx,
                                    q=f"In {rel}, what is the default value of the `{a.arg}` parameter of the "
                                      f"{what} `{qual}`?"))
        inner = [x for x in ast.walk(n) if isinstance(x, ast.Raise)]
        for r in inner:
            c = r.exc
            if isinstance(c, ast.Call) and c.args:
                m = c.args[0]
                text = m.value if isinstance(m, ast.Constant) and isinstance(m.value, str) else \
                    m.values[0].value if isinstance(m, ast.JoinedStr) and m.values and \
                    isinstance(m.values[0], ast.Constant) else None
                if text and len(text.strip()) >= 16:
                    head = " ".join(text.split())[:48].rstrip()
                    head = head[:head.rfind(" ")] if " " in head[20:] else head
                    if len(head) >= 16 and subtree_text.count(head) == 1:
                        short = qual.split(".")[-1]
                        rx = name_rx(short) if not short.startswith("__") else                             f"(?s)(?=.*{name_rx(qual.split('.')[0])}).*{name_rx(short)}"   # __init__ alone: too generic
                        out.append(dict(kind="raises", answer=qual, rx=rx,
                                        q=f"In {rel}, which function or method raises an error whose message "
                                          f"starts with \"{head}\"?"))
    for qual, nodes in assigns.items():
        if len(nodes) != 1 or not isinstance(nodes[0].value, ast.Constant):
            continue
        rx = literal_rx(nodes[0].value.value)
        if rx:
            out.append(dict(kind="const", answer=repr(nodes[0].value.value), rx=rx,
                            q=f"In {rel}, what value is `{qual}` set to?"))
    keep = []
    for f in out:
        if re.search(f["rx"], f["q"], re.I) or any(re.search(f["rx"], d, re.I) for d in DISTRACTORS):
            continue                                     # the answer is in the question or in a stock wrong reply
        f.update(file=rel, file_tokens=tokens(src))
        keep.append(f)
    return keep


def gen(a):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, (inst, sub) in SUBTREES.items():
        repo = os.path.join(SNAP, inst)
        files = list(py_files(os.path.join(repo, sub)))
        subtree_text = "".join(open(p, encoding="utf8", errors="replace").read() for p in files)
        by_kind = {}
        found = list(hop_facts(files, repo)) if SET == "v3" else []      # v3: two-file questions, all of them
        found += [f for p in files for f in facts(p, os.path.relpath(p, repo).replace("\\", "/"), subtree_text)]
        for f in found:
            by_kind.setdefault(f["kind"], []).append(f)
        rnd = random.Random(f"heldout-{name}")
        picked = []
        mix = dict({"hop_call": 0.5, "hop_default": 0.5, "hop_const": 0.5}, **MIX) if SET == "v3" else MIX
        for kind, share in mix.items():                  # spread over files: round-robin by file within a kind
            pool = by_kind.get(kind, [])
            rnd.shuffle(pool)
            per_file = {}
            for f in pool:
                per_file.setdefault(f["file"], []).append(f)
            order = sorted(per_file, key=lambda _: rnd.random())
            want, got = round(PER_SESSION * share), []
            while len(got) < want and any(per_file.values()):
                for fl in order:
                    if per_file[fl] and len(got) < want:
                        got.append(per_file[fl].pop())
            picked += got
        while len(picked) < PER_SESSION:                 # top up from whatever kinds have facts left
            rest = [f for fs in by_kind.values() for f in fs if f not in picked]
            if not rest:
                break
            picked.append(rnd.choice(rest))
        picked = picked[:PER_SESSION]                    # v3: the two-file questions first, then single-file ones
        rnd.shuffle(picked)
        if SET == "v2":                                  # half the questions name code without backticks
            for f in picked:
                if rnd.random() < 0.5:
                    f["q"] = f["q"].replace("`", "")
        for i, f in enumerate(picked):
            rows.append(dict(subtree=name, inst=inst, root=sub, n=i, **f))
        print(f"{name}: {len(files)} files, facts {({k: len(v) for k, v in by_kind.items()})}, picked {len(picked)} "
              f"from {len({f['file'] for f in picked})} files", flush=True)
    with open(QFILE, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} questions -> {QFILE}")


def workspace(inst, sub):
    """A copy of one package subtree (tests left out), paths kept repository-relative."""
    d = tempfile.mkdtemp(dir=os.environ.get("CORTEXNET_TMP", "D:/dev-cache/tmp"))
    repo = os.path.join(SNAP, inst)
    for p in py_files(os.path.join(repo, sub)):
        dst = os.path.join(d, os.path.relpath(p, repo))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(p, dst)
    return d


def run(a):
    qs = [json.loads(l) for l in open(QFILE, encoding="utf8")]
    subs = a.subtrees.split(",") if a.subtrees else list(SUBTREES)
    arms = a.arms.split(",")
    backend = live_ab.OpenAIBackend(a.openai, a.model, os.environ.get(a.key_env) if a.key_env else None,
                                   repair=True)
    fade = dict(limit=a.ctx, mark=0.85, low=0.5, keep=4, keep_frac=0.25)
    fade["turns"] = a.turns
    live_ab.READ_MAX = a.ctx // 4                   # one read (CortexNET or a page) may hold a quarter of the window
    os.makedirs(RUNS if SET != "v3" else os.path.join(RUNS, a.model), exist_ok=True)
    for name in subs:
        mine = [q for q in qs if q["subtree"] == name][:a.questions or PER_SESSION]
        for arm in arms:
            out = os.path.join(RUNS if SET != "v3" else os.path.join(RUNS, a.model),
                               f"{name}_{arm.replace('+', '_')}_ctx{a.ctx}_rep{a.rep}.jsonl")
            if os.path.exists(out):                     # one file per session: parallel runs never share a file
                print(f"{name} {arm}: done already", flush=True)
                continue
            root = workspace(mine[0]["inst"], mine[0]["root"])
            asked = {q["file"] for q in qs if q["subtree"] == name}
            example = next(f for f in sorted(live_ab.os.path.relpath(p, root).replace("\\", "/")
                                             for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
                           if f not in asked)
            live_ab.HINT = live_ab.HINT.replace("cortexnet/graph.py", example) if "cortexnet/graph.py" in                 live_ab.HINT else re.sub(r'path "[^"]+"', f'path "{example}"', live_ab.HINT)
            sess, t0 = {}, time.time()
            rows = []
            try:
                for q in mine:
                    before = backend.repairs
                    st = live_ab.run_task(backend, root, arm, q["q"], os.path.join(root, ".state"),
                                          max_calls=a.max_calls, count_errors=True, hint=True, session=sess,
                                          fade=fade if arm.endswith("+fade") else None, preread=a.preread,
                                          recover=a.recover)
                    st.update(subtree=name, arm=arm, ctx=a.ctx, rep=a.rep, model=a.model, n=q["n"], kind=q["kind"],
                              ok=not st["error"] and bool(re.search(q["rx"], st["answer"], re.I)),
                              answer=st["answer"][:300], file_tokens=q["file_tokens"],
                              repairs=backend.repairs - before)
                    rows.append(st)
            finally:
                shutil.rmtree(root, ignore_errors=True)
            with open(out, "a", encoding="utf8") as f:      # a session is written only once it finished
                for st in rows:
                    f.write(json.dumps(st) + "\n")
            print(f"{name} ctx={a.ctx} rep={a.rep} {arm}: {sum(r['ok'] for r in rows)}/{len(rows)} correct, "
                  f"{sum(bool(r['error']) for r in rows)} errors, {time.time() - t0:.0f} s", flush=True)


def evidence(q, text):
    """Does a read's output show what answers q? raises: the message line itself (the outline always names every
    function, so the name alone proves nothing); other kinds: the answer's text."""
    if q["kind"] == "raises":
        return re.search(r'starts with "(.+)"\?$', q["q"]).group(1) in text
    return bool(re.search(q["rx"], text, re.I))


def offline(a):
    """No model: one first read per question (the question as the query, a fresh reader), at each context's read
    cap: how often does that read show the answer? The paged arm's first page alongside."""
    qs = [json.loads(l) for l in open(QFILE, encoding="utf8")]
    if a.subtrees:
        qs = [q for q in qs if q["subtree"] in a.subtrees.split(",")]
    tmp = tempfile.mkdtemp(dir=os.environ.get("CORTEXNET_TMP", "D:/dev-cache/tmp"))
    try:
        for cap in (2048, 4096, 8192):
            hit, page, kinds = 0, 0, {}
            for i, q in enumerate(qs):
                path = os.path.join(SNAP, q["inst"], q["file"])
                r = live_ab.Reader(os.path.join(tmp, f"s{cap}_{i}"), split_tokens=600, max_tokens=cap)
                ok = evidence(q, r.read(path, q["q"]))
                live_ab.READ_MAX = cap
                pg = live_ab.Tools(os.path.join(SNAP, q["inst"]), "paged", tmp).call("read_file", {"path": q["file"]})
                pok = evidence(q, pg)
                hit, page = hit + ok, page + pok
                k = kinds.setdefault(q["kind"], [0, 0, 0])
                k[0], k[1], k[2] = k[0] + ok, k[1] + pok, k[2] + 1
            print(f"read cap {cap}: CortexNET first read shows the answer {hit}/{len(qs)} ({hit / len(qs):.0%}), "
                  f"first page {page}/{len(qs)} ({page / len(qs):.0%}); by kind (cortex/page/n): "
                  + ", ".join(f"{k} {v[0]}/{v[1]}/{v[2]}" for k, v in kinds.items()), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def boot_diff(x, y, n=4000, seed=0):
    """95% interval of mean(x - y), paired bootstrap over questions."""
    import numpy as np
    d = np.asarray(x, float) - np.asarray(y, float)
    r = np.random.default_rng(seed)
    m = d[r.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def report(a):
    rows = [json.loads(l) for p in glob.glob(os.path.join(RUNS, "**", "*.jsonl"), recursive=True) for l in open(p, encoding="utf8")]
    if not rows:
        raise SystemExit("no runs yet")
    arms = [x for x in ("baseline", "baseline+fade", "paged", "paged+fade", "cortex", "cortexna+fade", "cortex+fade",
                        "cortexf+fade")
            if any(r["arm"] == x for r in rows)]
    ctxs = sorted({(r.get("model", "llama-3.1-8b"), r["ctx"]) for r in rows})
    key = lambda r: (r["subtree"], r["rep"], r["n"])  # noqa: E731
    L = ["# Held-out session stress test", "",
         f"`bench/heldout.py`: {len({(r['subtree'], r['n']) for r in rows})} machine-written questions over "
         f"{len({r['subtree'] for r in rows})} package subtrees of SWE-bench Lite repositories (none used to build or "
         "tune CortexNET), one conversation per subtree and arm (up to 40 questions in a row).", ""]
    for model, ctx in ctxs:
        R = [r for r in rows if r["ctx"] == ctx and r.get("model", "llama-3.1-8b") == model]
        by = {arm: {key(r): r for r in R if r["arm"] == arm} for arm in arms}
        reps = sorted({r["rep"] for r in R})
        cell = lambda arm, ks: (f"{sum(by[arm][k]['ok'] for k in ks if k in by[arm])}/"  # noqa: E731
                                f"{sum(k in by[arm] for k in ks)}" if any(k in by[arm] for k in ks) else "-")
        L += [f"## {model}, context {ctx:,} tokens (repeats {reps})", "",
              "| arm | correct | questions lost to an overflow | sessions alive at the end | first overflow (median "
              "question) | input tokens per question | tool calls per question | mean peak context | tool-call "
              "repairs |", "|---|---|---|---|---|---|---|---|---|"]
        for arm in arms:
            rr = list(by[arm].values())
            if not rr:
                continue
            sess = {}
            for r in rr:
                sess.setdefault((r["subtree"], r["rep"]), []).append(r)
            first = sorted(min([r["n"] for r in v if r["error"]] or [len(v)]) + 1 for v in sess.values())
            alive = sum(not v[-1]["error"] for v in (sorted(v, key=lambda r: r["n"]) for v in sess.values()))
            L.append(f"| {arm} | {sum(r['ok'] for r in rr)}/{len(rr)} ({sum(r['ok'] for r in rr) / len(rr):.0%}) | "
                     f"{sum(bool(r['error']) for r in rr)} | {alive}/{len(sess)} | "
                     f"{first[len(first) // 2] if any(r['error'] for r in rr) else 'never'} | "
                     f"{sum(r['input_tokens'] for r in rr) / len(rr):,.0f} | {sum(r['calls'] for r in rr) / len(rr):.1f} | "
                     f"{sum(r['peak_context'] for r in rr) / len(rr):,.0f} | {sum(r.get('repairs', 0) for r in rr)} |")
        L.append("")
        for x, ref in (("cortex", "paged"), ("cortex+fade", "paged+fade"), ("cortex+fade", "baseline"),
                       ("cortex+fade", "cortex"), ("paged+fade", "paged"), ("cortex+fade", "cortexna+fade"),
                       ("cortexf+fade", "cortex+fade"), ("cortexf+fade", "paged+fade")):
            ks = sorted(set(by.get(x, {})) & set(by.get(ref, {})))
            if ks:
                m, lo, hi = boot_diff([by[x][k]["ok"] for k in ks], [by[ref][k]["ok"] for k in ks])
                L.append(f"- {x} minus {ref}: {m:+.1%} correct (95% CI {lo:+.1%} to {hi:+.1%}, paired over {len(ks)} "
                         "questions)")
        allk = sorted({k for v in by.values() for k in v})
        L += ["", "| subtree | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
        for s in sorted({k[0] for k in allk}):
            ks = [k for k in allk if k[0] == s]
            L.append(f"| {s} | " + " | ".join(cell(arm, ks) for arm in arms) + " |")
        kinds = {k: next(v[k]["kind"] for v in by.values() if k in v) for k in allk}
        L += ["", "| question kind | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
        for kind in ("default", "const", "raises", "base", "hop_call", "hop_default", "hop_const"):
            ks = [k for k in allk if kinds[k] == kind]
            if ks:
                L.append(f"| {kind} | " + " | ".join(cell(arm, ks) for arm in arms) + " |")
        L.append("")
    with open(os.path.join(HERE, "results", "heldout.md" if SET == "v1" else f"heldout_{SET}.md"), "w",
              encoding="utf8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gen", "run", "report", "offline"])
    ap.add_argument("--set", default="v1", choices=["v1", "v2", "v3"],
                    help="v1: the development set; v2: the test set; v3: two-file questions, 60 per session")
    ap.add_argument("--openai", default="")
    ap.add_argument("--model", default="llama-3.1-8b")
    ap.add_argument("--key-env", default="CORTEXNET_LLAMA_KEY")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--subtrees", default="")
    ap.add_argument("--arms", default="baseline,paged,paged+fade,cortex,cortex+fade")
    ap.add_argument("--questions", type=int, default=0, help="per session (default: the set's session length)")
    ap.add_argument("--max-calls", type=int, default=8)
    ap.add_argument("--turns", action="store_true", help="the fade may also drop the oldest whole exchanges when "
                    "shrinking tool results is not enough (long sessions in a small window)")
    ap.add_argument("--recover", action="store_true", help="learn the fade's token ratio from an overflow error's "
                    "prompt count (otherwise one overflow can leave the fade estimate low for the rest of the session)")
    ap.add_argument("--preread", action="store_true", help="read the files a question names before the model's first "
                    "turn, through the arm's own read tool (for small models that rarely call tools)")
    a = ap.parse_args()
    use_set(a.set)
    {"gen": gen, "run": run, "report": report, "offline": offline}[a.what](a)


if __name__ == "__main__":
    main()
