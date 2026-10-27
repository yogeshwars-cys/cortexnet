"""Symbol-level include test on git history: does intent text pick the symbols that get changed?

Local repos only (aggregates are printed, nothing copied). A case is (commit, file): the file's parent version is large
enough for outline mode, and the diff changes lines inside at least one leaf symbol of it. Labels: leaf symbols whose
old-side lines the commit changed (a pure insertion counts the symbol around the insertion point). Query: the commit
message. Commit messages are written after the change and often name what changed, so this is an optimistic upper bound
for how well an intent query can choose up-front includes; the report gives the share of cases where it names a label.

Rankers fill the Reader's include budget (max(400, 25% of the file's tokens)). Logistic mixes are fitted on half of
the repos and tested on the other half (repo-level split, both directions averaged).
Usage: python -I bench/commit_symbols.py [root_with_repos]
"""
import os
import re
import subprocess
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortexnet import symbols, tokens  # noqa: E402
from cortexnet.learn import idents  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = (".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".java", ".kt",
       ".dart", ".swift", ".cs", ".gd")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, timeout=120)
    return r.stdout.decode("utf8", errors="replace") if r.returncode == 0 else None


def repo_cases(repo):
    log = git(repo, "log", "--no-merges", "-n", "400", "--format=%H%x00%P%x00%s%x00%b%x1e")
    out = []
    for rec in (log or "").split("\x1e"):
        parts = rec.strip("\n").split("\x00")
        if len(parts) < 4 or not parts[1].strip():
            continue
        h, parent, subj, body = parts[0].strip(), parts[1].split()[0], parts[2], parts[3]
        diff = git(repo, "diff", "-U0", "--no-color", "--no-renames", parent, h)
        if not diff:
            continue
        changed, cur = {}, None
        for line in diff.split("\n"):
            if line.startswith("--- "):
                cur = line[6:] if line.startswith("--- a/") else None
            elif line.startswith("@@") and cur and cur.lower().endswith(EXT):
                m = HUNK.match(line)
                if m:
                    a, n = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
                    changed.setdefault(cur, []).append((a, a + n - 1) if n > 0 else (max(a, 1), max(a, 1)))
        for path, spans in changed.items():
            text = git(repo, "show", f"{parent}:{path}")
            if not text or len(text) > 300_000 or tokens(text) <= 800:
                continue
            syms = [s for s in symbols(text, path) if not s.children]
            if len(syms) < 3:
                continue
            y = np.array([any(s.start <= b and s.end >= a for a, b in spans) for s in syms])
            if not y.any() or y.all():
                continue
            lines = text.split("\n")
            out.append(dict(repo=os.path.basename(repo), msg=(subj + "\n" + body)[:1500], syms=syms, y=y, lines=lines,
                            file_tok=tokens(text), path=path,
                            cost=np.array([tokens("\n".join(lines[s.start - 1:s.end])) for s in syms])))
    return out


def evaluate(cs, score):
    rec, anyhit, mrr = [], [], []
    for c in cs:
        s = score(c) + 1e-9 * np.random.default_rng(len(c["y"])).random(len(c["y"]))
        order = np.argsort(-s)
        budget = max(400, 0.25 * c["file_tok"])
        used, shown = 0, set()
        for j in order:
            if used + c["cost"][j] <= budget:
                shown.add(j)
                used += c["cost"][j]
        need = set(np.where(c["y"])[0])
        rec.append(len(need & shown) / len(need))
        anyhit.append(bool(need & shown))
        mrr.append(1 / (1 + int(np.where(c["y"][order])[0][0])))
    return np.mean(rec), np.mean(anyhit), np.mean(mrr)


def main():
    repos = sorted(os.path.join(ROOT, d) for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d, ".git")))
    cs = []
    for r in repos:
        got = repo_cases(r)
        cs += got
        print(f"{os.path.basename(r)}: {len(got)} cases", flush=True)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    for c in cs:
        bodies = ["\n".join(c["lines"][s.start - 1:s.end]) for s in c["syms"]]
        V = m.encode([s.head + "\n" + b[:1500] for s, b in zip(c["syms"], bodies)], batch_size=64, normalize_embeddings=True)
        z = m.encode([c["msg"]], normalize_embeddings=True)[0]
        q = idents(c["msg"])
        bid = [idents(b) for b in bodies]
        c["dense"] = V @ z
        c["lex"] = np.array([len(b & q) / max(len(b), 1) for b in bid])
        c["name"] = np.array([float(s.name.lower() in q or (bool(idents(s.name)) and idents(s.name) <= q))
                              for s in c["syms"]])
        c["F"] = np.c_[np.log1p(c["cost"]), c["lex"], c["name"], c["dense"]]
    names = sorted({c["repo"] for c in cs})
    groups = [set(names[0::2]), set(names[1::2])]
    rankers_fixed = [("random", lambda c: np.zeros(len(c["y"]))), ("smallest first (size only)", lambda c: -c["cost"]),
                     ("largest first", lambda c: c["cost"].astype(float)),
                     ("lexical overlap with message", lambda c: c["lex"]),
                     ("message names the symbol", lambda c: c["name"] - 1e-3 * c["cost"]),
                     ("dense similarity to message", lambda c: c["dense"])]
    res = {n: [] for n, _ in rankers_fixed}
    res.update({"logistic: size + lexical + name": [], "logistic: + dense": []})
    for g in groups:
        tr = [c for c in cs if c["repo"] not in g]
        te = [c for c in cs if c["repo"] in g]
        if not tr or not te:
            continue
        F = np.vstack([c["F"] for c in tr])
        y = np.concatenate([c["y"] for c in tr])
        mu, sd = F.mean(0), F.std(0) + 1e-6
        full = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit((F - mu) / sd, y)
        nod = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(((F - mu) / sd)[:, :3], y)
        for n, fn in rankers_fixed:
            res[n].append((len(te), evaluate(te, fn)))
        res["logistic: size + lexical + name"].append(
            (len(te), evaluate(te, lambda c: ((c["F"] - mu) / sd)[:, :3] @ nod.coef_[0])))
        res["logistic: + dense"].append((len(te), evaluate(te, lambda c: ((c["F"] - mu) / sd) @ full.coef_[0])))
    names_hit = np.mean([bool((c["name"] * c["y"]).any()) for c in cs])
    L = ["# Git-history include test: does intent text pick the symbols that get changed?", "",
         f"{len(cs)} (commit, file) cases from {len(names)} repos (files > 800 tokens; median "
         f"{int(np.median([len(c['y']) for c in cs]))} leaf symbols per file, {np.mean([c['y'].mean() for c in cs]):.0%} "
         f"changed on average). The commit message names a changed symbol in {names_hit:.0%} of cases (leakage: messages "
         "are written after the change). Budget = max(400, 25% of file tokens). Repo-level split, two folds, weighted mean.", "",
         "| ranker | recall of changed symbols in budget | cases with >= 1 changed symbol shown | MRR |", "|---|---|---|---|"]
    for n, rows in res.items():
        w = np.array([k for k, _ in rows], float)
        v = np.array([x for _, x in rows])
        a, b, c3 = (v * w[:, None]).sum(0) / w.sum()
        L.append(f"| {n} | {a:.0%} | {b:.0%} | {c3:.2f} |")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "commit_symbols.md")
    open(out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
