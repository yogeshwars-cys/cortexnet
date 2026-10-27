"""Zero-shot proxy for a learned snapshot encoder: dense similarity between the model's reasoning and candidate files.

Same transitions as snapshot_signal.py. Each candidate file is embedded from its name, its symbol names and its first
lines; each snapshot (thinking + text before the call) is embedded as is (MiniLM-L6, cached locally). If a dense
encoder finds no signal here, a small learned bottleneck trained on these labels has little to work with.
Only transitions with a non-empty snapshot are scored. Usage: python -I bench/dense_snapshot.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import snapshot_signal as ss  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def doc(p):
    try:
        with open(p, encoding="utf8", errors="replace") as f:
            head = f.read(1500)
    except OSError:
        head = ""
    return f"{os.path.basename(p)} {' '.join(sorted(ss.defs(p) or []))[:600]} {head}"


def main():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    S = ss.sessions()
    half = len(S) // 2
    te = [x for _, rs in S[half:] for x in rs if x["snap"]]
    tr = [x for _, rs in S[:half] for x in rs if x["snap"]]
    files = sorted({q for x in tr + te for q in x["cands"]})
    E = dict(zip(files, m.encode([doc(q) for q in files], batch_size=64, normalize_embeddings=True)))
    for x in tr + te:
        z = m.encode([x["text"]], normalize_embeddings=True)[0]
        x["dense"] = np.array([E[q] @ z for q in x["cands"]])
    f_rec, _ = ss.fit(tr, [0, 1, 4])
    for x in tr + te:
        x["F"] = np.c_[x["F"], x["dense"]]
    f_all, w = ss.fit(tr, [0, 1, 4, 5])
    L = ["# Dense snapshot similarity (zero-shot proxy for a learned intent encoder)", "",
         f"Transitions with a non-empty snapshot: train {len(tr)}, test {len(te)}; {len(files)} distinct candidate files.", ""]
    for nm, sub in [("all", te), ("re-opens", [x for x in te if not x["first"]]), ("first-time opens", [x for x in te if x["first"]])]:
        L += [f"## {nm} ({len(sub)})", "", "| ranker | top-1 | top-5 | MRR |", "|---|---|---|---|"]
        for name, fn in [("random", lambda F: np.zeros(len(F))), ("recency (LRU)", lambda F: -F[:, 0]),
                         ("dense snapshot-file similarity alone", lambda F: F[:, 5]),
                         ("logistic: recency + frequency + same dir", f_rec),
                         ("logistic: + dense similarity", f_all)]:
            a, b, c = ss.metrics(sub, fn)
            L.append(f"| {name} | {a:.2f} | {b:.2f} | {c:.2f} |")
        L.append("")
    L.append("Weights (standardized; recency, frequency, same dir, dense): " + ", ".join(f"{v:+.2f}" for v in w))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "dense_snapshot.md")
    open(out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
