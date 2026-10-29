"""The learning loop: Cortex CLI sessions become training cases for the coprocessor.

Every read_file the agent makes is logged (.cortexnet/agent/cases.jsonl: the task, the read's query, the agent's
latest `want`, and the file text by hash), and so is every read_lines of a file after it. A range after a read means
the read did not show something the agent needed, so the symbols it covers are that read's labels. This is the same
case format bench/subnn_train.py trains on (`source: sessions`, `y` = indices of the file's leaf symbols that were
needed), so `cortex learn` can write a dataset from your own sessions, alone or merged with an existing one.

Reads that were never followed by a range carry no label here (they may have shown everything, or the agent gave up),
so they are counted but not written.
"""
import datetime
import json
import os
import sys

from .coproc import leaves

MIN_CASES = 200        # below this a training run mostly fits noise; bench/subnn_train.py splits by time


def load(cases_dir):
    """(read events by id, range events), skipping lines that do not parse (a run killed mid-write)."""
    reads, ranges = {}, []
    path = os.path.join(cases_dir, "cases.jsonl")
    if not os.path.exists(path):
        return reads, ranges
    with open(path, encoding="utf8") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("ev") == "read" and ev.get("id"):
                reads[ev["id"]] = ev
            elif ev.get("ev") == "range":
                ranges.append(ev)
    return reads, ranges


def build(cases_dir):
    """(cases, texts, stats) in subnn_train's format."""
    reads, ranges = load(cases_dir)
    need = {}
    for r in ranges:
        need.setdefault(r.get("read"), []).append((int(r.get("start", 0)), int(r.get("end", 0))))
    cases, texts = [], {}
    stats = dict(reads=len(reads), ranges=len(ranges), labelled=0, files=0, missing_text=0)
    for rid, ev in sorted(reads.items(), key=lambda kv: kv[1].get("t", 0)):
        spans = need.get(rid)
        if not spans:
            continue
        h = ev.get("text")
        tp = os.path.join(cases_dir, "texts", f"{h}.txt")
        if h not in texts:
            try:
                with open(tp, encoding="utf8", newline="") as f:
                    texts[h] = f.read()
            except OSError:
                stats["missing_text"] += 1
                continue
        syms = leaves(texts[h], ev["path"])
        y = sorted({k for k, s in enumerate(syms) for a, b in spans if s.start <= b and s.end >= a})
        if not y:
            continue
        ts = datetime.datetime.fromtimestamp(ev.get("t", 0), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        q = "\n".join(x for x in (ev.get("q"), ev.get("intent"), ev.get("query")) if x)
        cases.append(dict(source="sessions", sid="cortex-cli", ts=ts, kind="range", src="disk", path=ev["path"],
                          text=h, y=y, q=q, h=ev.get("h") or "", prev=[], views_at="range"))
    stats["labelled"], stats["files"] = len(cases), len({c["text"] for c in cases})
    return cases, texts, stats


def write(out_dir, cases, texts, base=None):
    """Write cases.jsonl + texts.json to out_dir, after the cases of an existing dataset `base` if given."""
    os.makedirs(out_dir, exist_ok=True)
    all_cases, all_texts = [], {}
    if base:
        with open(os.path.join(base, "cases.jsonl"), encoding="utf8") as f:
            all_cases = [json.loads(line) for line in f if line.strip()]
        with open(os.path.join(base, "texts.json"), encoding="utf8") as f:
            all_texts = json.load(f)
    all_cases += cases
    all_texts.update(texts)
    with open(os.path.join(out_dir, "cases.jsonl"), "w", encoding="utf8") as f:
        f.writelines(json.dumps(c) + "\n" for c in all_cases)
    with open(os.path.join(out_dir, "texts.json"), "w", encoding="utf8") as f:
        json.dump(all_texts, f)
    return len(all_cases)


def summary(stats):
    s = (f"{stats['reads']} reads logged, {stats['ranges']} ranges after them; "
         f"{stats['labelled']} labelled cases over {stats['files']} files")
    if stats["missing_text"]:
        s += f" ({stats['missing_text']} reads lost their file text)"
    if stats["reads"]:
        s += f". {100 * stats['labelled'] / stats['reads']:.0f}% of reads needed a range afterwards"
    return s


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="cortex learn", description="Turn Cortex CLI sessions into coprocessor "
                                 "training cases (the read -> range pairs logged in .cortexnet/agent).")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", help="write the dataset here (default: <repo>/.cortexnet/agent/subnn)")
    ap.add_argument("--base", help="an existing dataset to merge with (a dir with cases.jsonl and texts.json)")
    a = ap.parse_args(argv)
    cdir = os.path.join(os.path.realpath(a.repo), ".cortexnet", "agent")
    cases, texts, stats = build(cdir)
    print(summary(stats))
    if not cases:
        print("nothing to write yet: run some Cortex CLI sessions in this repository first")
        return 0
    out = a.out or os.path.join(cdir, "subnn")
    n = write(out, cases, texts, a.base)
    print(f"wrote {n} cases to {out}" + (f" ({len(cases)} new, merged with {a.base})" if a.base else ""))
    if stats["labelled"] < MIN_CASES and not a.base:
        print(f"note: {stats['labelled']} cases is too few to train on alone (want {MIN_CASES}+); "
              "merge with --base to fine-tune on top of an existing dataset")
    print("train (needs torch; from the cortexnet repository):\n"
          f'  SUBNN_DATA="{out}" python bench/subnn_train.py --configs full --codec 64 --export <head.npz>\n'
          "then run Cortex CLI with --coproc <head.npz>")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
