"""SWE-bench training cases for Cortex-SubNN: issue text -> the symbols its gold patch changes.

The coprocessor picks which symbol bodies a read shows when the request is natural language (an issue, a bug report)
rather than code names, which the anchored reads already handle. Its first head was trained on the user's sessions and
git history only. Here every SWE-bench instance outside SWE-bench Lite (the read evaluation, bench/read_eval.py, is
built from Lite episodes) becomes one case per changed Python source file: the file at the instance's base commit,
the issue as the query, and as labels the leaf symbols holding a removed line or an insertion point of the gold patch.
Test files are skipped (the agent reads source, and test patches are not part of the gold fix).

Files are fetched one by one from raw.githubusercontent.com at the base commit (no clone), cached under CACHE.
Output: SUBNN_DATA-style cases.jsonl + texts.json in OUT, with the existing session and git cases copied in, so
bench/subnn_train.py trains on all of them (non-session cases are training-only; the session cross-validation and
read_eval stay the tests).

  python -I bench/subnn_swe.py [--max 3000] [--out D:/dev-cache/cortexnet/subnn_swe]
"""
import argparse
import json
import os
import re
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from cortexnet import symbols  # noqa: E402
from subnn_data import SPLIT, Store  # noqa: E402

BASE = os.environ.get("SUBNN_DATA", "D:/dev-cache/cortexnet/subnn")
CACHE = "D:/dev-cache/cortexnet/swe_files"
HF = os.environ.get("HF_HOME", "D:/dev-cache/hf")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


def gold_lines(patch):
    """{path: set of pre-image line numbers}: removed lines, and the line before each insertion."""
    out, path, old = {}, None, 0
    for l in patch.split("\n"):
        if l.startswith("diff --git"):
            path = None
        elif l.startswith("--- "):
            p = l[4:].strip()
            path = p[2:] if p.startswith("a/") else None
        elif path and (m := HUNK.match(l)):
            old = int(m.group(1))
        elif path and l.startswith("-") and not l.startswith("---"):
            out.setdefault(path, set()).add(old)
            old += 1
        elif path and l.startswith("+") and not l.startswith("+++"):
            out.setdefault(path, set()).add(max(1, old - 1))
        elif path and l.startswith(" "):
            old += 1
    return out


def fetch(repo, commit, path):
    key = os.path.join(CACHE, f"{repo.replace('/', '__')}__{commit[:12]}__{path.replace('/', '__')}")
    if os.path.exists(key):
        return open(key, encoding="utf8").read()
    try:
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}", timeout=30) as r:
            text = r.read().decode("utf8", "replace").replace("\r\n", "\n")
    except OSError:
        return None
    with open(key, "w", encoding="utf8") as f:
        f.write(text)
    return text


def old_symbols(ref="867a95a"):
    """symbols() as of `ref`: the base cases' labels are indices into the leaf list of that outline. The Python outline
    has listed class attributes since 2026-11-01, which shifts those indices, so they are carried over by line span."""
    import importlib.util
    import subprocess
    import tempfile
    src = subprocess.run(["git", "-C", os.path.dirname(HERE), "show", f"{ref}:cortexnet/outline.py"],
                         capture_output=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(dir=os.environ.get("CORTEXNET_TMP", "D:/dev-cache/tmp")), "outline_old.py")
    with open(path, "wb") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("outline_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.symbols


def remap(cases, texts, old):
    """Labels of cases built with the old outline, re-pointed at today's leaves covering the same lines."""
    out, lost = [], 0
    for c in cases:
        text = texts[c["text"]]
        before = [s for s in old(text, c["path"], split_tokens=SPLIT) if not s.children]
        spans = [(before[i].start, before[i].end) for i in c["y"] if i < len(before)]
        now = [s for s in symbols(text, c["path"], split_tokens=SPLIT) if not s.children]
        y = [j for j, s in enumerate(now) if any(s.start <= b and s.end >= a for a, b in spans)]
        if not y or len(y) == len(now):
            lost += 1
            continue
        out.append(dict(c, y=y, spans=spans))
    return out, lost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=3000, help="instances to use")
    ap.add_argument("--out", default="D:/dev-cache/cortexnet/subnn_swe")
    ap.add_argument("--base-only", default="", help="also write the re-mapped base cases alone here (the control)")
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", HF)
    from datasets import load_dataset
    full = load_dataset("princeton-nlp/SWE-bench", split="test")
    lite = {r["instance_id"] for r in load_dataset("princeton-nlp/SWE-bench_Lite", split="test")}
    rows = [r for r in full if r["instance_id"] not in lite][:a.max]
    print(f"SWE-bench test: {len(full)}, outside Lite: {len(rows)}", flush=True)
    os.makedirs(CACHE, exist_ok=True)
    jobs = [(r, p, g) for r in rows for p, g in gold_lines(r["patch"]).items()
            if p.endswith(".py") and not re.search(r"(^|/)(tests?|testing)(/|_)|(^|/)test_|_test\.py$|conftest\.py$", p)]
    with ThreadPoolExecutor(16) as ex:
        texts = list(ex.map(lambda j: fetch(j[0]["repo"], j[0]["base_commit"], j[1]), jobs))
    store, cases, why = Store(), [], {}
    for (r, p, g), text in zip(jobs, texts):
        if not text or len(text) > 400_000:
            why["fetch"] = why.get("fetch", 0) + 1
            continue
        syms = [s for s in symbols(text, p, split_tokens=SPLIT) if not s.children]
        y = [i for i, s in enumerate(syms) if any(s.start <= ln <= s.end for ln in g)]
        if len(syms) < 3 or not y or len(y) == len(syms):
            why["label"] = why.get("label", 0) + 1
            continue
        cases.append(dict(source="swe", sid=r["instance_id"], ts="", step=0, kind="issue", src="swe", path=p,
                          text=store.put(text), y=y, spans=[[ln, ln] for ln in sorted(g)], lo=0, hi=0,
                          q=r["problem_statement"], h="", prev=[], c_turn=0,
                          repo=r["repo"]))
    print(f"{len(jobs)} changed source files, {len(cases)} cases from {len({c['sid'] for c in cases})} instances "
          f"(dropped: {why})", flush=True)
    os.makedirs(a.out, exist_ok=True)
    base = [json.loads(l) for l in open(os.path.join(BASE, "cases.jsonl"), encoding="utf8")]
    texts0 = json.load(open(os.path.join(BASE, "texts.json"), encoding="utf8"))
    base, lost = remap(base, texts0, old_symbols())
    print(f"base cases re-mapped to today's outline: {len(base)} kept, {lost} lost", flush=True)
    if a.base_only:
        os.makedirs(a.base_only, exist_ok=True)
        with open(os.path.join(a.base_only, "cases.jsonl"), "w", encoding="utf8") as f:
            for c in base:
                f.write(json.dumps(c) + "\n")
        with open(os.path.join(a.base_only, "texts.json"), "w", encoding="utf8") as f:
            json.dump(texts0, f)
        if os.path.exists(os.path.join(BASE, "hrem.json")):
            shutil.copy(os.path.join(BASE, "hrem.json"), os.path.join(a.base_only, "hrem.json"))
    texts0.update(store.texts)
    with open(os.path.join(a.out, "cases.jsonl"), "w", encoding="utf8") as f:
        for c in base + cases:
            f.write(json.dumps(c) + "\n")
    with open(os.path.join(a.out, "texts.json"), "w", encoding="utf8") as f:
        json.dump(texts0, f)
    for extra in ("hrem.json",):
        if os.path.exists(os.path.join(BASE, extra)):
            shutil.copy(os.path.join(BASE, extra), os.path.join(a.out, extra))
    print(f"wrote {len(base)} base + {len(cases)} SWE cases, {len(texts0)} texts to {a.out}")


if __name__ == "__main__":
    main()
