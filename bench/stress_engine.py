"""Stress test of the CortexNET engine itself (no model, no GPU): pathological files, whole real repositories, the MCP
server under abuse.

  python -I bench/stress_engine.py files            # generated pathological files, each in a child with a time limit
  python -I bench/stress_engine.py corpus <dir>...  # read every source file of real repositories, record latency,
                                                     # output size against the whole file, and invariant failures
  python -I bench/stress_engine.py mcp              # the stdio MCP server: malformed input, bad arguments, a burst
  python -I bench/stress_engine.py all <dir>...     # all three, report to results/stress_engine.md

Invariants checked on every read: no exception; the outline's line spans lie inside the file; a range returns exactly
the lines asked for; a read never returns more than the whole numbered file would (plus a fixed header allowance).
Temporary files go to $CORTEXNET_TMP (default D:/dev-cache/tmp/cortexnet-stress, else the system temp dir) and are
removed afterwards. `mcp` exits non-zero if the server loses a call or does not exit cleanly (CI runs it).
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cortexnet.outline import symbols, tokens  # noqa: E402
from cortexnet.reader import Reader, numbered  # noqa: E402

TMP = os.environ.get("CORTEXNET_TMP") or os.path.join(
    "D:/dev-cache/tmp" if os.path.isdir("D:/dev-cache") else tempfile.gettempdir(), "cortexnet-stress")
RESULTS = os.path.join(HERE, "results")
EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cc", ".cpp", ".hpp", ".cu", ".go", ".rs", ".java", ".kt",
        ".swift", ".rb", ".php", ".cs", ".m", ".mm", ".sh", ".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml",
        ".html", ".css", ".sql", ".lua", ".zig", ".metal", ".glsl", ".comp", ".proto", ".cmake"}
HEADER = 200          # tokens a read may add on top of the whole file (outline header, markers)
LIMIT = 120.0         # seconds per generated case (read, outline, ranges, an edit and a re-read, then a memory pass)


def numbered_tokens(text):
    """What a whole-file read returns: every line with its number (as every read tool here numbers lines)."""
    return tokens(numbered(text.split("\n"), 1, text.count("\n") + 1))


# ------------------------------------------------------------------ generated pathological files
def cases():
    r = random.Random(0)
    big_py = "\n".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(20000))
    nested = "".join("    " * i + f"def g{i}():\n" for i in range(90)) + "    " * 90 + "pass\n"
    deep_expr = "x = " + "(" * 400 + "1" + ")" * 400 + "\n"
    proto = "\n".join(f"int fn_{i}(int a, int b);" for i in range(15000))
    consts = "\n".join(f"K{i} = {i}" for i in range(30000))
    huge_fn = "def big():\n" + "".join(f"    v{i} = {i}\n" for i in range(40000))
    regex_bomb = "int " + "a" * 5000 + " (" + "b, " * 20000 + "\n"
    regex_bomb2 = "public static " + "x " * 30000 + "\n"
    yield "empty.py", b""
    yield "blank_lines.py", b"\n" * 200000
    yield "one_line_minified.js", ("var a=function(){return 1};" * 200000).encode()
    yield "long_line_no_newline.txt", b"x" * 5_000_000
    yield "binary_random.py", bytes(r.getrandbits(8) for _ in range(2_000_000))
    yield "png_header.png", b"\x89PNG\r\n\x1a\n" + bytes(r.getrandbits(8) for _ in range(500_000))
    yield "nul_bytes.py", b"def a():\n    return 1\n\x00\x00\x00" * 20000
    yield "latin1.py", ("# caf\xe9\n" + "def f():\n    return '\xe9'\n" * 5000).encode("latin-1")
    yield "utf16_bom.py", ("def f():\n    return 1\n" * 5000).encode("utf-16")
    yield "crlf.py", ("def f(x):\r\n    return x\r\n\r\n" * 8000).encode()
    yield "tabs.py", ("class A:\n\tdef f(self):\n\t\treturn 1\n" * 6000).encode()
    yield "syntax_error.py", (big_py[:200000] + "\ndef broken(:\n").encode()
    yield "deep_nesting.py", nested.encode()
    yield "deep_expression.py", (deep_expr * 50 + "def f():\n    pass\n" * 2000).encode()
    yield "many_functions.py", big_py.encode()
    yield "many_consts.py", consts.encode()
    yield "one_huge_function.py", huge_fn.encode()
    yield "c_prototypes.h", proto.encode()
    yield "regex_bomb.c", (regex_bomb * 3).encode()
    yield "regex_bomb2.java", (regex_bomb2 * 3).encode()
    yield "markdown_headings.md", "".join(f"{'#' * (i % 6 + 1)} h{i}\ntext\n" for i in range(30000)).encode()
    yield "only_comments.c", ("// comment line\n" * 100000).encode()
    yield "unicode_names.py", "".join(f"def función_{i}(ñ):\n    return 'λ{i}'\n" for i in range(5000)).encode()
    yield "surrogates.py", ("def f():\n    return '\ud800'\n" * 5000).encode("utf8", "surrogatepass")


def _one(path, q):
    """Child process: every operation on one file, invariants checked; the result goes back on a queue."""
    out = dict(ok=True, errors=[])
    try:
        st = os.path.join(os.path.dirname(path), ".state_" + os.path.basename(path))
        rd = Reader(state_dir=st, split_tokens=600)
        t = time.perf_counter()
        text = rd.read(path, "return value")
        out["read_s"] = time.perf_counter() - t
        with open(path, encoding="utf8", errors="replace") as f:
            whole = f.read()
        n = whole.count("\n") + 1
        out.update(file_tokens=tokens(whole), shown_tokens=tokens(text), lines=n)
        if tokens(text) > numbered_tokens(whole) + HEADER:
            out["errors"].append(f"read larger than the whole file: {tokens(text)} > {numbered_tokens(whole)}")
        for s in symbols(whole, path, 600):
            if not (1 <= s.start <= s.end <= n):
                out["errors"].append(f"span out of file: {s.name} L{s.start}-{s.end} of {n}")
                break
        t = time.perf_counter()
        rd.outline(path)
        out["outline_s"] = time.perf_counter() - t
        a, b = max(1, n // 2), min(n, n // 2 + 9)
        got = rd.range(path, a, b)
        if not got.startswith("empty range") and len(got.split("\n")) != b - a + 1:
            out["errors"].append(f"range {a}-{b} returned {len(got.split(chr(10)))} lines")
        for bad in ((0, 0), (-5, 3), (n + 10, n + 20), (10, 2), (1, 10**9)):
            rd.range(path, *bad)
        with open(path, "ab") as f:              # an edit, then a re-read: the change detector runs
            f.write(b"\n# edited\n")
        t = time.perf_counter()
        rd.read(path, "edited")
        out["reread_s"] = time.perf_counter() - t
        tracemalloc.start()                      # memory in a separate, untimed read (tracing slows Python severalfold)
        Reader(state_dir=st + "_mem", split_tokens=600).read(path, "return value")
        out["peak_mb"] = tracemalloc.get_traced_memory()[1] / 1e6
    except Exception as e:  # noqa: BLE001 - every failure is a finding
        out.update(ok=False, errors=out["errors"] + [f"{type(e).__name__}: {str(e)[:200]}"])
    q.put(out)


def run_files():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    rows = []
    ctx = mp.get_context("spawn")
    for name, data in cases():
        path = os.path.join(TMP, name)
        with open(path, "wb") as f:
            f.write(data)
        q = ctx.Queue()
        p = ctx.Process(target=_one, args=(path, q))
        t = time.perf_counter()
        p.start()
        p.join(LIMIT)
        if p.is_alive():
            p.terminate()
            p.join()
            res = dict(ok=False, errors=[f"timeout: over {LIMIT:.0f} s"])
        else:
            res = q.get() if not q.empty() else dict(ok=False, errors=[f"child died, exit {p.exitcode}"])
        res.update(name=name, bytes=len(data), wall=time.perf_counter() - t)
        rows.append(res)
        flag = "ok " if res["ok"] and not res["errors"] else "BAD"
        print(f"{flag} {name:28s} {len(data) / 1e6:6.2f} MB  read {res.get('read_s', 0):6.2f}s  reread "
              f"{res.get('reread_s', 0):6.2f}s  shown {res.get('shown_tokens', 0):>8} / {res.get('file_tokens', 0):>8}"
              f"  {'; '.join(res['errors'])}", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    return rows


# ------------------------------------------------------------------ real repositories
def source_files(root):
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".") and x not in ("node_modules", "__pycache__", "build")]
        for f in fs:
            if os.path.splitext(f)[1].lower() in EXTS:
                yield os.path.join(d, f)


def run_corpus(roots, max_files=0):
    os.makedirs(TMP, exist_ok=True)
    rd = Reader(state_dir=os.path.join(TMP, "corpus_state"), split_tokens=600)
    files = [p for r in roots for p in source_files(r)]
    if max_files:
        files = random.Random(0).sample(files, min(max_files, len(files)))
    rows, errors = [], []
    t0 = time.perf_counter()
    for i, p in enumerate(files):
        try:
            size = os.path.getsize(p)
            with open(p, encoding="utf8", errors="replace") as f:
                whole = f.read()
            t = time.perf_counter()
            out = rd.read(p, "")
            dt = time.perf_counter() - t
            n = whole.count("\n") + 1
            if any(not (1 <= s.start <= s.end <= n) for s in rd._symbols(whole, os.path.abspath(p))):
                errors.append((p, "span out of file"))
            if tokens(out) > numbered_tokens(whole) + HEADER:
                errors.append((p, f"read larger than whole: {tokens(out)} > {numbered_tokens(whole)}"))
            a = max(1, n // 3)
            got = rd.range(p, a, a + 4)
            if len(got.split("\n")) != min(n, a + 4) - a + 1:
                errors.append((p, "range line count"))
            rows.append(dict(path=p, ext=os.path.splitext(p)[1].lower(), bytes=size, file=tokens(whole),
                             shown=tokens(out), s=dt, outlined=tokens(whole) > rd.full_below))
        except Exception as e:  # noqa: BLE001
            errors.append((p, f"{type(e).__name__}: {str(e)[:160]}"))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(files)} files, {time.perf_counter() - t0:.0f} s", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    return rows, errors, time.perf_counter() - t0


def corpus_summary(rows, errors, wall):
    import numpy as np
    s = np.array([r["s"] for r in rows]) * 1000
    big = [r for r in rows if r["outlined"]]
    ratio = np.array([r["shown"] / r["file"] for r in big]) if big else np.zeros(1)
    worse = [r for r in big if r["shown"] >= r["file"]]
    by_ext = {}
    for r in big:
        e = by_ext.setdefault(r["ext"], [0, 0, 0])
        e[0] += 1
        e[1] += r["file"]
        e[2] += r["shown"]
    slow = sorted(rows, key=lambda r: -r["s"])[:5]
    lines = [f"- files read: {len(rows):,} ({len(big):,} above the whole-file threshold, outlined) in {wall:.0f} s; "
             f"errors / invariant failures: {len(errors)}",
             f"- read latency: median {np.median(s):.1f} ms, p95 {np.percentile(s, 95):.1f} ms, p99 "
             f"{np.percentile(s, 99):.1f} ms, max {s.max():.0f} ms",
             f"- outlined files, tokens shown / whole file: median {np.median(ratio):.2f}, p90 "
             f"{np.percentile(ratio, 90):.2f}, total {sum(r['shown'] for r in big) / max(1, sum(r['file'] for r in big)):.2f}; "
             f"reads that showed at least the whole file: {len(worse)} ({len(worse) / max(1, len(big)):.1%})",
             "", "| ext | outlined files | tokens shown / whole |", "|---|---|---|"]
    for e, (n, f, sh) in sorted(by_ext.items(), key=lambda kv: -kv[1][0])[:14]:
        lines.append(f"| {e} | {n} | {sh / f:.2f} |")
    lines += ["", "Slowest reads: " + "; ".join(f"`{os.path.basename(r['path'])}` {r['s'] * 1000:.0f} ms "
                                                 f"({r['bytes'] / 1e3:.0f} kB)" for r in slow)]
    if worse:
        lines.append("Reads no smaller than the file (worst 5): " + "; ".join(
            f"`{os.path.basename(r['path'])}` {r['shown']}/{r['file']}"
            for r in sorted(worse, key=lambda r: r['file'] - r['shown'])[:5]))
    for p, e in errors[:15]:
        lines.append(f"- error `{p}`: {e}")
    return lines


# ------------------------------------------------------------------ MCP server
def run_mcp():
    os.makedirs(TMP, exist_ok=True)
    env = dict(os.environ, CORTEXNET_STATE=os.path.join(TMP, "mcp_state"))
    proc = subprocess.Popen([sys.executable, "-m", "cortexnet.mcp_server"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(HERE), env=env)
    me = os.path.join(os.path.dirname(HERE), "cortexnet", "reader.py")
    call = lambda i, name, a: dict(jsonrpc="2.0", id=i, method="tools/call", params=dict(name=name, arguments=a))  # noqa
    msgs = ["not json at all", "{\"jsonrpc\": \"2.0\"", "[]", "null", "42", json.dumps({"id": 1}),
            json.dumps(dict(jsonrpc="2.0", id=2, method="nope")),
            json.dumps(call(3, "cortex_read", {})), json.dumps(call(4, "cortex_read", {"path": 5})),
            json.dumps(call(5, "cortex_read", {"path": "Z:/no/such/file.py"})),
            json.dumps(call(6, "cortex_range", {"path": me, "start": "a", "end": "b"})),
            json.dumps(call(7, "cortex_range", {"path": me, "start": None, "end": 3})),
            json.dumps(call(8, "cortex_read", {"path": os.path.dirname(me)})),
            json.dumps(call(9, "cortex_approve", {"ids": "#1"})), json.dumps(call(10, "cortex_approve", {"ids": None})),
            json.dumps(call(11, "cortex_read", {"path": me, "approve": 7})),
            json.dumps(dict(jsonrpc="2.0", method="notifications/initialized")),
            json.dumps(dict(jsonrpc="2.0", id=12, method="tools/call", params=None))]
    burst = 2000
    msgs += [json.dumps(call(100 + i, ("cortex_read", "cortex_range", "cortex_outline")[i % 3],
                             {"path": me, "query": "tick", "start": i % 500 + 1, "end": i % 500 + 20}))
             for i in range(burst)]
    t = time.perf_counter()
    try:
        out, err = proc.communicate(("\n".join(msgs) + "\n").encode(), timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    wall = time.perf_counter() - t
    replies = []
    for l in out.decode("utf8", "replace").splitlines():
        try:
            replies.append(json.loads(l))
        except ValueError:
            replies.append({"unparsable": l[:80]})
    ids = {r.get("id") for r in replies}
    lost = [i for i in range(100, 100 + burst) if i not in ids]
    shutil.rmtree(TMP, ignore_errors=True)
    res = dict(sent=len(msgs), replies=len(replies), exit=proc.returncode, lost_burst=len(lost), wall=wall,
               errors=sum("error" in r for r in replies), tool_errors=sum(bool((r.get("result") or {}).get("isError"))
                                                                          for r in replies if isinstance(r, dict)),
               unparsable=sum("unparsable" in r for r in replies), stderr=err.decode("utf8", "replace")[-600:])
    print(json.dumps({k: v for k, v in res.items() if k != "stderr"}), "\n", res["stderr"][-400:])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["files", "corpus", "mcp", "all"])
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--max-files", type=int, default=0)
    a = ap.parse_args()
    report = ["# CortexNET engine stress test", "", f"`bench/stress_engine.py`, Python {sys.version.split()[0]}, "
              f"{time.strftime('%Y-%m-%d')}. No model: the reader alone, on generated pathological files, whole "
              "repositories and the stdio MCP server.", ""]
    if a.what in ("files", "all"):
        rows = run_files()
        report += ["## Pathological files", "", "| file | MB | ok | read s | re-read after edit s | tokens shown / whole "
                   "| peak MB | problems |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            report.append(f"| {r['name']} | {r['bytes'] / 1e6:.2f} | {'yes' if r['ok'] and not r['errors'] else 'NO'} | "
                          f"{r.get('read_s', 0):.2f} | {r.get('reread_s', 0):.2f} | {r.get('shown_tokens', '-')} / "
                          f"{r.get('file_tokens', '-')} | {r.get('peak_mb', 0):.0f} | {'; '.join(r['errors'])} |")
        report.append("")
    if a.what in ("corpus", "all") and a.roots:
        rows, errors, wall = run_corpus(a.roots, a.max_files)
        report += ["## Whole repositories", "", "Roots: " + ", ".join(f"`{os.path.basename(r.rstrip('/'))}`"
                                                                     for r in a.roots), ""]
        report += corpus_summary(rows, errors, wall) + [""]
        print("\n".join(corpus_summary(rows, errors, wall)))
    if a.what in ("mcp", "all"):
        m = run_mcp()
        if a.what == "mcp" and (m["lost_burst"] or m["exit"] != 0 or m["unparsable"]):
            raise SystemExit("MCP server lost calls or did not exit cleanly")   # a failing check for CI
        report += ["## MCP server", "", f"{m['sent']} messages in one pipe (18 malformed or with bad arguments, then a "
                   f"burst of 2000 tool calls): {m['replies']} replies in {m['wall']:.1f} s, {m['lost_burst']} burst "
                   f"calls unanswered, {m['errors']} JSON-RPC errors, {m['tool_errors']} tool errors, "
                   f"{m['unparsable']} unparsable lines, exit code {m['exit']}.", ""]
    if a.what == "all":
        with open(os.path.join(RESULTS, "stress_engine.md"), "w", encoding="utf8") as f:
            f.write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
