"""Offline read evaluation on SWE-bench Lite: what a single CortexNET read shows of the code the fix needs.

For every read_file case of the Glimmer A/B episodes (D:/dev-cache/cortexnet/ab, public SWE-bench Lite tasks), the
file as the agent saw it is written to a temp dir and read once per Reader configuration, with the issue text as the
query and the agent's reasoning before the read as the snapshot (the views at read time). Metrics per configuration:
  gold-line recall   share of the file's gold-patch lines (removed, or next to an insertion) visible with their line
                     number in the read's output (outline labels do not count, bodies and x-ray lines do)
  needed hit         share of reads where some line of a needed symbol (gold lines, later ranges, submitted symbol)
                     is visible
  tokens             output tokens of the read, and as a share of the whole file
A fresh Reader per case, so nothing learnt from one task leaks into another. The coprocessor head was trained on the
user's sessions and git history only, so this is out of domain for it.
Usage: python -I bench/read_eval.py --coproc <head.npz> [--configs v1,coproc,...] [--out results/read_eval.md]
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from cortexnet import Reader, tokens  # noqa: E402
from cortexnet.coproc import leaves  # noqa: E402
from swe_agent import gold_lines  # noqa: E402

AB = "D:/dev-cache/cortexnet/ab"
NUM = re.compile(r"^\s*(\d+)\t", re.M)

CONFIGS = {   # name -> Reader kwargs (coproc=True means the loaded head)
    "v1": dict(),
    "v1 + x-ray": dict(xray_top=2),
    "coprocessor": dict(coproc=True),
    "coprocessor + x-ray": dict(coproc=True, xray_top=2),
    "coprocessor top-3": dict(coproc=True, coproc_top=3),
    "coprocessor top-3 + MMR": dict(coproc=True, coproc_top=3, mmr_beta=0.5),
}


def load_cases():
    inst = {i["instance_id"]: i for i in json.load(open(os.path.join(AB, "instances.json"), encoding="utf8"))}
    out = []
    for arm in ("whole", "cortex", "fade"):
        for f in sorted(glob.glob(os.path.join(AB, arm, "*.json"))):
            ep = json.load(open(f, encoding="utf8"))
            texts = ep.get("texts") or {}
            for c in ep.get("cases") or []:
                i = inst.get(c["sid"])
                if i is None or c["text"] not in texts:
                    continue
                rel = c["path"][len(i["repo"]) + 1:] if c["path"].startswith(i["repo"] + "/") else c["path"]
                gold = set(gold_lines(i["patch"]).get(rel, set()))
                out.append(dict(sid=c["sid"], arm=arm, rel=rel, text=texts[c["text"]], q=c["q"], h=c["h"], y=c["y"],
                                gold=gold))
    return out


def visible(out):
    return {int(m) for m in NUM.findall(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coproc", default="D:/dev-cache/cortexnet/subnn/head_minilm_c64_v2.npz")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "read_eval.md"))
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (smoke test; nothing is written)")
    a = ap.parse_args()
    names = [n.strip() for n in a.configs.split(",")]
    t0 = time.time()
    cases = load_cases()[:a.limit or None]
    cp = None
    if any(CONFIGS[n].get("coproc") for n in names):
        from cortexnet.coproc import Coprocessor, MiniLM
        cp = Coprocessor(a.coproc)
        cp.encoder = MiniLM()
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMP"))
    res = {n: dict(rec=[], hit=[], tok=[], share=[]) for n in names}
    res_sid, gold_ix = [], []
    try:
        for j, c in enumerate(cases):
            p = os.path.join(tmp, f"c{j}", c["rel"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf8") as f:
                f.write(c["text"])
            lines = c["text"].split("\n")
            syms = leaves(c["text"], p)
            need = set(c["gold"])
            for k in c["y"]:
                if k < len(syms):
                    need.update(range(syms[k].start, syms[k].end + 1))
            full = tokens(c["text"])
            if c["gold"]:
                gold_ix.append(len(res_sid))
            res_sid.append(c["sid"])
            for n in names:
                kw = dict(CONFIGS[n])
                if kw.pop("coproc", False):
                    kw["coproc"] = cp
                r = Reader(os.path.join(tmp, f"s{j}_{len(res[n]['tok'])}_{n.replace(' ', '')}"), split_tokens=600,
                           ctx_base=2000, **kw)
                out = r.read(p, c["q"], snapshot=c["h"])
                v = visible(out)
                if c["gold"]:
                    res[n]["rec"].append(len(v & c["gold"]) / len(c["gold"]))
                res[n]["hit"].append(bool(v & need))
                res[n]["tok"].append(tokens(out))
                res[n]["share"].append(tokens(out) / max(full, 1))
            if (j + 1) % 20 == 0:
                print(f"{j + 1}/{len(cases)} cases ({time.time() - t0:.0f}s)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_gold = len(res[names[0]]["rec"])
    L = ["# Offline read evaluation: SWE-bench Lite read cases", "",
         f"{len(cases)} read_file cases from the Glimmer A/B episodes ({n_gold} with gold-patch lines in the read file), "
         f"one read each, query = issue text, snapshot = the agent's reasoning before the read. Coprocessor head "
         f"`{os.path.basename(a.coproc)}` (trained without SWE data). ({time.time() - t0:.0f}s)", "",
         "| configuration | gold-line recall (mean) | reads with all gold lines visible | needed symbol visible | "
         "tokens per read (mean) | share of the whole file (median) |", "|---|---|---|---|---|---|"]
    for n in names:
        R = res[n]
        L.append(f"| {n} | {np.mean(R['rec']):.3f} | {np.mean([x >= 1 - 1e-9 for x in R['rec']]):.0%} | "
                 f"{np.mean(R['hit']):.0%} | {np.mean(R['tok']):,.0f} | {np.median(R['share']):.0%} |")
    base = res[names[0]]
    sid = np.array(res_sid)
    tasks = sorted(set(res_sid))
    groups = [np.where(sid == t)[0] for t in tasks]
    L += ["", f"Paired differences against `{names[0]}`, 95% intervals from a bootstrap over the {len(tasks)} tasks "
          f"(reads of one task resampled together, since they are not independent):", ""]
    rng = np.random.default_rng(0)
    for n in names[1:]:
        for key, lab in (("rec", "gold-line recall"), ("tok", "tokens per read")):
            ix = gold_ix if key == "rec" else np.arange(len(res_sid))
            d = np.full(len(res_sid), np.nan)
            d[ix] = np.array(res[n][key]) - np.array(base[key])
            bs = []
            for _ in range(2000):
                pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                v = d[pick]
                bs.append(np.nanmean(v))
            L.append(f"- {n}, {lab}: {np.nanmean(d):+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    if not a.limit:                                   # a smoke test never overwrites the report
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
