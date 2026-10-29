"""Offline test of the spiking file manager (cortexnet/filesnn.py): does it know which file a real model wants next?

Two models, their real file operations in time order, one manager each (learned weights carry across sessions, as
they would live; the session traces start empty every session):
  claude   local Claude Code logs (bench/replay_sessions.load_sessions): reads, edits and writes; the model's words
           before an operation are the user prompt in force + its last two text blocks (the replay's query). Neurons:
           the files the session has touched (no repository index is logged).
  llama    the recorded laptop sessions of Llama 3.1 8B (bench/results/traces/live_ab_llama_3_1_8b_session8k*.jsonl,
           one stream per arm): its tool calls with a path; its words = the question + its text of that turn.
           Neurons: the fixture's files, known from the start (the repository index).
The Llama runs repeat one set of 20 questions, so a per-question split is the honest held-out set: only `eta` is
chosen, on the first half of the questions (Llama) and the first half of the sessions (Claude); the table reports the
other half. The manager's two populations are reported together (the gate picks) and apart (the innate one is
"learning off"; the plastic one alone). Baselines: recency (the last touched file first), mention then recency (files the words name by path, then
recency), the manager with learning off. "new file hits": operations on a file the session
had not touched yet, the only ones warming can speed up (a touched file is already in the cache).

Usage: python -I bench/files_replay.py      ->  bench/results/files_replay.md
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from cortexnet.filesnn import SpikingFileManager  # noqa: E402
from cortexnet.outline import symbols  # noqa: E402

REPO = os.path.dirname(HERE)
STDLIB = os.path.dirname(os.__file__)
ETAS = (0.0, 0.02, 0.05, 0.1, 0.2)


def names(text, path):
    try:
        return [s.name for s in symbols(text, path) if not s.children][:60]
    except Exception:
        return []


def claude_streams():
    from replay_sessions import load_sessions
    out = []
    for _, _, evs in load_sessions():
        steps, words = [], ""
        for e in evs:
            if e["op"] == "read":
                words = e.get("query", words)
                text = "\n".join(t for _, t in e["rows"]) if not (e["off"] or e["lim"]) else None
            elif e["op"] == "write":
                text = e["text"]
            else:
                text = None
            steps.append(dict(path=e["path"], words=words, text=text, q=None))
        if len(steps) >= 4:
            out.append(steps)
    return out, {}


def llama_streams():
    """One stream per (trace, arm) of the session runs; the fixture's files are known from the start."""
    known = {}
    for rel in [f for f in os.listdir(os.path.join(REPO, "cortexnet")) if f.endswith(".py")]:
        known["cortexnet/" + rel] = open(os.path.join(REPO, "cortexnet", rel), encoding="utf8").read()
    for f in ("argparse.py", "textwrap.py", "shlex.py", "configparser.py", "difflib.py", "logging/handlers.py"):
        known["stdlib/" + f] = open(os.path.join(STDLIB, f), encoding="utf8").read()
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, "results", "traces", "live_ab_llama_3_1_8b_session8k*.jsonl"))):
        by_arm = {}
        for line in open(f, encoding="utf8"):
            r = json.loads(line)
            if r["arm"] == "baseline":
                continue
            for c in r.get("calls") or []:
                args = c.get("arguments")
                path = args.get("path") if isinstance(args, dict) else None
                if path:
                    path = path.replace("\\", "/").lstrip("./")
                    hit = [k for k in known if k == path or k.endswith("/" + path)]
                    by_arm.setdefault(r["arm"], []).append(
                        dict(path=hit[0] if len(hit) == 1 else path, words=r["q"] + "\n" + (r.get("content") or ""),
                             text=None, q=r["q"]))
        out += [s for s in by_arm.values() if len(s) >= 4]
    return out, known


class PlasticOnly(SpikingFileManager):
    def plastic_drives(self):
        return True


def run(streams, known, eta, learn=True, plastic_only=False):
    fm = (PlasticOnly if plastic_only else SpikingFileManager)(eta=eta, learn=learn)
    known_names = {p: names(t, p) for p, t in known.items()}
    res = dict(snn=[], recency=[], mention=[], fresh=[], session=[], q=[], ops=0)
    for si, steps in enumerate(streams):
        fm.new_session()
        for p, n in known_names.items():
            fm.know(p, n)
        last = {}                                                  # file -> step it was last touched
        for st in steps:
            res["ops"] += 1
            p = st["path"]
            if p in fm.idx:
                X = fm.inputs(st["words"])
                pred = fm.predict(st["words"])
                rec = sorted(last, key=lambda f: -last[f])
                ment = sorted((fm.files[i] for i in np.flatnonzero(X[:, 1])), key=lambda f: (-last.get(f, -1), f))
                ment += [f for f in rec if f not in ment]
                res["snn"].append(pred.index(p) if p in pred else None)
                res["recency"].append(rec.index(p) if p in rec else None)
                res["mention"].append(ment.index(p) if p in ment else None)
                res["fresh"].append(p not in fm.touched)
                res["session"].append(si)
                res["q"].append(st["q"])
            else:
                fm.pending = None
            fm.used(p)
            if st["text"]:
                fm.know(p, names(st["text"], p))
            last[p] = res["ops"]
    return res, fm


def hit(ranks, k, mask=None):
    r = [x for i, x in enumerate(ranks) if mask is None or mask[i]]
    return float(np.mean([x is not None and x < k for x in r])) if r else float("nan")


def boot(a, b, k, mask, n=2000, seed=0):
    """Paired bootstrap over predictions in `mask`: hit@k(a) - hit@k(b)."""
    x = np.array([(u is not None and u < k) - (v is not None and v < k) for u, v, m in zip(a, b, mask) if m], float)
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def split(res, streams, model):
    """(tuning mask, held-out mask): Llama by question, Claude by session."""
    if model.startswith("Llama"):
        qs = sorted({q for q in res["q"]})
        first = set(qs[:len(qs) // 2])
        tune = [q in first for q in res["q"]]
    else:
        tune = [s < len(streams) // 2 for s in res["session"]]
    return tune, [not t for t in tune]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "files_replay.md"))
    a = ap.parse_args()
    t0 = time.time()
    data = (("Claude (Claude Code sessions)", claude_streams()), ("Llama 3.1 8B (laptop sessions)", llama_streams()))
    # eta: the best mean hit@3 over both models' tuning halves
    score = {}
    for eta in ETAS:
        s = []
        for model, (streams, known) in data:
            res, _ = run(streams, known, eta)
            tune, _ = split(res, streams, model)
            s.append(hit(res["snn"], 3, tune))
        score[eta] = float(np.mean(s))
    eta = max(ETAS, key=lambda e: (score[e], -e))
    L = ["# Spiking file manager: which file a real model wants next", "",
         "Each model's real file operations in time order; before each one the manager predicts from the model's "
         "words, then learns from the truth. Held out: Llama by question (its runs repeat 20 questions), Claude by "
         f"session. eta chosen on the tuning halves: {eta} (mean hit@3 by eta: "
         + ", ".join(f"{e}: {score[e]:.3f}" for e in ETAS) + ").", "",
         "| model | streams | file ops | predictable | method | hit@1 | hit@3 | new file hits@3 (n) |",
         "|---|---|---|---|---|---|---|---|"]
    notes = []
    for model, (streams, known) in data:
        live, fm = run(streams, known, eta)
        zero, _ = run(streams, known, eta, learn=False)
        plastic, _ = run(streams, known, eta, plastic_only=True)
        _, held = split(live, streams, model)
        n_held = sum(held)
        fresh = [h and f for h, f in zip(held, live["fresh"])]
        for name, ranks in (("spiking manager (gated)", live["snn"]), ("innate population (learning off)", zero["snn"]),
                            ("plastic population alone", plastic["snn"]),
                            ("mention, then recency", live["mention"]), ("recency", live["recency"])):
            L.append(f"| {model} | {len(streams)} | {live['ops']:,} | {len(live['snn']) / max(live['ops'], 1):.0%} "
                     f"(held out {n_held:,}) | {name} | {hit(ranks, 1, held):.1%} | {hit(ranks, 3, held):.1%} | "
                     f"{hit(ranks, 3, fresh):.1%} ({sum(fresh)}) |")
        for name, other in (("learning off", zero["snn"]), ("plastic alone", plastic["snn"]),
                            ("mention, then recency", live["mention"]),
                            ("recency", live["recency"])):
            m, lo, hi = boot(live["snn"], other, 3, held)
            notes.append(f"- {model}: manager vs {name}, hit@3 on the held-out half: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        r = fm.agreement()
        notes.append(f"- {model}: plastic weights at the end {fm.habits()} (innate {dict(zip(fm.habits(), fm.w0.tolist()))}); "
                     f"driving at the end: {'plastic' if r['plastic'] else 'innate'}; {r['spikes']:.1f} spikes per "
                     f"prediction, {r['silent']:.0%} of neurons silent")
    L += ["", "Paired differences (bootstrap over predictions, 95% interval) and the network:", ""] + notes
    L += ["", f"({time.time() - t0:.0f}s)"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
